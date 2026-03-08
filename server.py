from __future__ import annotations

import asyncio
import base64
import gc
import io
import json
import os
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Awaitable, Callable
from uuid import uuid4

import av
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from qwen_asr import Qwen3ASRModel


PROJECT_ROOT = Path(__file__).resolve().parent
UI_DIR = PROJECT_ROOT
CACHE_ROOT = PROJECT_ROOT / ".cache"
RUNTIME_ROOT = PROJECT_ROOT / ".runtime"

os.environ.setdefault("HF_HOME", str(CACHE_ROOT / "hf"))
os.environ.setdefault(
    "HUGGINGFACE_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub")
)
os.environ.setdefault(
    "TRANSFORMERS_CACHE", str(Path(os.environ["HF_HOME"]) / "transformers")
)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg"))

MODEL_NAME = os.environ.get("QWEN_RT_MODEL_NAME", "qwen3-asr-rt")
MODEL_ID = os.environ.get("QWEN_RT_MODEL_ID", "Qwen/Qwen3-ASR-1.7B")
MODEL_PATH = os.environ.get("QWEN_RT_MODEL_PATH")
SAMPLE_RATE = 16000
COMMIT_SECONDS = float(os.environ.get("QWEN_RT_COMMIT_SECONDS", "8.0"))
PARTIAL_MIN_SECONDS = float(os.environ.get("QWEN_RT_PARTIAL_MIN_SECONDS", "2.0"))
PARTIAL_STEP_SECONDS = float(os.environ.get("QWEN_RT_PARTIAL_STEP_SECONDS", "1.0"))
UPLOAD_DIR = Path(os.environ.get("QWEN_RT_UPLOAD_DIR", str(RUNTIME_ROOT / "uploads")))
IDLE_UNLOAD_SECONDS = float(os.environ.get("QWEN_RT_IDLE_UNLOAD_SECONDS", "120"))
IDLE_CHECK_SECONDS = float(os.environ.get("QWEN_RT_IDLE_CHECK_SECONDS", "10"))
ASR_CONTEXT = os.environ.get(
    "QWEN_RT_ASR_CONTEXT",
    (
        "Transcribe the speech faithfully and follow these rules strictly. "
        "Write the output as Korean sentences by default. "
        "If an English word, acronym, product name, brand name, or technical term is clearly spoken, keep that word itself in original English spelling. "
        "Except for those English words, keep particles, endings, connectors, and overall sentence structure in Korean. "
        "Do not output full English sentences. "
        "Do not output any Chinese characters or Chinese words. "
        "If uncertain, do not guess Chinese; write conservatively in Korean instead. "
        "Do not translate, summarize, paraphrase, explain, or rewrite the speech. "
        "Only transcribe what is actually spoken."
    ),
)

COMMIT_SAMPLES = int(COMMIT_SECONDS * SAMPLE_RATE)
PARTIAL_MIN_SAMPLES = int(PARTIAL_MIN_SECONDS * SAMPLE_RATE)
PARTIAL_STEP_SAMPLES = int(PARTIAL_STEP_SECONDS * SAMPLE_RATE)

app = FastAPI()
app.mount("/assets", StaticFiles(directory=UI_DIR), name="assets")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_MODEL: Qwen3ASRModel | None = None
_MODEL_LOCK = threading.Lock()
_INFER_LOCK = threading.Lock()
_STATE_LOCK = threading.Lock()
_ACTIVE_INFERENCES = 0
_LAST_ACTIVITY_TS = time.monotonic()
_JOBS: dict[str, "FileJob"] = {}
_JOBS_LOCK = threading.Lock()


@dataclass(slots=True)
class EnergyVadOptions:
    min_silence_duration_ms: int = 700
    speech_pad_ms: int = 220
    max_speech_duration_s: float = 20.0
    frame_ms: int = 30
    hop_ms: int = 10
    min_speech_duration_ms: int = 220
    rms_floor: float = 0.004
    rms_ratio: float = 2.2


@dataclass(slots=True)
class RuntimeConfig:
    device_map: str
    dtype: str
    attn_implementation: str
    max_inference_batch_size: int
    local_files_only: bool


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def resolve_cached_model_path() -> str | None:
    if MODEL_PATH and Path(MODEL_PATH).exists():
        return MODEL_PATH

    hf_home = Path(os.environ.get("HF_HOME", str(CACHE_ROOT / "hf")))
    model_key = f"models--{MODEL_ID.replace('/', '--')}"
    candidate_roots = [
        hf_home / model_key,
        hf_home / "hub" / model_key,
    ]
    for repo_dir in candidate_roots:
        snapshots_dir = repo_dir / "snapshots"
        if not snapshots_dir.is_dir():
            continue
        snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
        if snapshots:
            return str(snapshots[-1])
    return None


@lru_cache(maxsize=1)
def select_runtime_config() -> RuntimeConfig:
    local_files_only = env_flag("QWEN_RT_LOCAL_FILES_ONLY", False)
    if not torch.cuda.is_available():
        return RuntimeConfig(
            device_map="cpu",
            dtype="float32",
            attn_implementation="eager",
            max_inference_batch_size=1,
            local_files_only=local_files_only,
        )

    major, minor = torch.cuda.get_device_capability(0)
    total_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    device_map = os.environ.get("QWEN_RT_DEVICE_MAP", "cuda:0")

    if major >= 8:
        dtype = os.environ.get(
            "QWEN_RT_MODEL_DTYPE",
            "bfloat16" if torch.cuda.is_bf16_supported() else "float16",
        )
        attn = os.environ.get("QWEN_RT_ATTN_IMPLEMENTATION", "sdpa")
        batch = 8 if total_gib >= 16 else 4
    elif major >= 7:
        dtype = os.environ.get("QWEN_RT_MODEL_DTYPE", "float16")
        attn = os.environ.get("QWEN_RT_ATTN_IMPLEMENTATION", "sdpa")
        batch = 6 if total_gib >= 16 else 3
    else:
        dtype = os.environ.get("QWEN_RT_MODEL_DTYPE", "float16")
        attn = os.environ.get("QWEN_RT_ATTN_IMPLEMENTATION", "eager")
        batch = 4 if total_gib >= 16 else 2

    batch = int(os.environ.get("QWEN_RT_MAX_INFERENCE_BATCH_SIZE", str(batch)))

    return RuntimeConfig(
        device_map=device_map,
        dtype=dtype,
        attn_implementation=attn,
        max_inference_batch_size=batch,
        local_files_only=local_files_only,
    )


def get_model() -> Qwen3ASRModel:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            runtime = select_runtime_config()
            model_source = resolve_cached_model_path() or MODEL_ID
            _MODEL = Qwen3ASRModel.from_pretrained(
                model_source,
                device_map=runtime.device_map,
                dtype=runtime.dtype,
                attn_implementation=runtime.attn_implementation,
                max_inference_batch_size=runtime.max_inference_batch_size,
                max_new_tokens=512,
                local_files_only=runtime.local_files_only,
            )
            touch_activity()
    return _MODEL


def touch_activity() -> None:
    global _LAST_ACTIVITY_TS
    with _STATE_LOCK:
        _LAST_ACTIVITY_TS = time.monotonic()


def inference_started() -> None:
    global _ACTIVE_INFERENCES, _LAST_ACTIVITY_TS
    with _STATE_LOCK:
        _ACTIVE_INFERENCES += 1
        _LAST_ACTIVITY_TS = time.monotonic()


def inference_finished() -> None:
    global _ACTIVE_INFERENCES, _LAST_ACTIVITY_TS
    with _STATE_LOCK:
        _ACTIVE_INFERENCES = max(0, _ACTIVE_INFERENCES - 1)
        _LAST_ACTIVITY_TS = time.monotonic()


def unload_model_if_idle() -> bool:
    global _MODEL
    with _STATE_LOCK:
        idle_for = time.monotonic() - _LAST_ACTIVITY_TS
        should_unload = (
            _MODEL is not None
            and _ACTIVE_INFERENCES == 0
            and idle_for >= IDLE_UNLOAD_SECONDS
        )
    if not should_unload:
        return False

    with _MODEL_LOCK:
        if _MODEL is None:
            return False
        with _STATE_LOCK:
            idle_for = time.monotonic() - _LAST_ACTIVITY_TS
            if _ACTIVE_INFERENCES != 0 or idle_for < IDLE_UNLOAD_SECONDS:
                return False
        _MODEL = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True


async def idle_unload_loop() -> None:
    while True:
        await asyncio.sleep(IDLE_CHECK_SECONDS)
        unloaded = await asyncio.to_thread(unload_model_if_idle)
        if unloaded:
            print(
                f"Idle unload completed after {IDLE_UNLOAD_SECONDS:.0f}s without requests.",
                flush=True,
            )


def merge_speech_chunks(
    speech_chunks: list[dict[str, int]], sr: int = SAMPLE_RATE
) -> list[dict[str, int]]:
    if not speech_chunks:
        return []
    merged: list[dict[str, int]] = []
    cur = {"start": int(speech_chunks[0]["start"]), "end": int(speech_chunks[0]["end"])}
    for seg in speech_chunks[1:]:
        gap_sec = (int(seg["start"]) - cur["end"]) / sr
        next_dur_sec = (int(seg["end"]) - cur["start"]) / sr
        if gap_sec <= 1.0 and next_dur_sec <= 20.0:
            cur["end"] = int(seg["end"])
        else:
            merged.append(cur)
            cur = {"start": int(seg["start"]), "end": int(seg["end"])}
    merged.append(cur)
    return merged


def get_speech_timestamps(
    audio: np.ndarray,
    vad_options: EnergyVadOptions,
    sampling_rate: int = SAMPLE_RATE,
) -> list[dict[str, int]]:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return []

    frame_samples = max(1, int(vad_options.frame_ms * sampling_rate / 1000))
    hop_samples = max(1, int(vad_options.hop_ms * sampling_rate / 1000))
    min_silence_frames = max(1, int(vad_options.min_silence_duration_ms / vad_options.hop_ms))
    min_speech_frames = max(1, int(vad_options.min_speech_duration_ms / vad_options.hop_ms))
    pad_samples = int(vad_options.speech_pad_ms * sampling_rate / 1000)
    max_speech_samples = int(vad_options.max_speech_duration_s * sampling_rate)

    rms_values: list[float] = []
    offsets: list[int] = []
    for start in range(0, max(1, audio.size - frame_samples + 1), hop_samples):
        frame = audio[start : start + frame_samples]
        rms_values.append(float(np.sqrt(np.mean(np.square(frame), dtype=np.float32))))
        offsets.append(start)

    if not rms_values:
        return []

    rms = np.asarray(rms_values, dtype=np.float32)
    noise_floor = float(np.percentile(rms, 35))
    threshold = max(vad_options.rms_floor, noise_floor * vad_options.rms_ratio)
    speech_mask = rms >= threshold

    chunks: list[dict[str, int]] = []
    start_frame: int | None = None
    silence_run = 0
    for idx, is_speech in enumerate(speech_mask):
        if is_speech:
            if start_frame is None:
                start_frame = idx
            silence_run = 0
            continue
        if start_frame is None:
            continue
        silence_run += 1
        if silence_run >= min_silence_frames:
            end_frame = idx - silence_run + 1
            if end_frame - start_frame >= min_speech_frames:
                start = max(0, offsets[start_frame] - pad_samples)
                end = min(audio.size, offsets[min(end_frame, len(offsets) - 1)] + frame_samples + pad_samples)
                chunks.append({"start": int(start), "end": int(end)})
            start_frame = None
            silence_run = 0

    if start_frame is not None:
        end_frame = len(offsets) - 1
        if end_frame - start_frame + 1 >= min_speech_frames:
            start = max(0, offsets[start_frame] - pad_samples)
            end = min(audio.size, offsets[end_frame] + frame_samples + pad_samples)
            chunks.append({"start": int(start), "end": int(end)})

    split_chunks: list[dict[str, int]] = []
    for chunk in chunks:
        start = int(chunk["start"])
        end = int(chunk["end"])
        while end - start > max_speech_samples:
            split_chunks.append({"start": start, "end": start + max_speech_samples})
            start += max_speech_samples
        split_chunks.append({"start": start, "end": end})
    return split_chunks


def transcribe_audio(audio: np.ndarray) -> str:
    clips = build_speech_clips(audio)
    if not clips:
        return ""
    texts = transcribe_clips(clips)
    return " ".join(texts).strip()


def build_speech_clips(audio: np.ndarray) -> list[tuple[np.ndarray, int]]:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return []

    vad_options = EnergyVadOptions(
        min_silence_duration_ms=700,
        speech_pad_ms=220,
        max_speech_duration_s=20.0,
    )
    speech_chunks = get_speech_timestamps(
        audio, vad_options=vad_options, sampling_rate=SAMPLE_RATE
    )
    merged = merge_speech_chunks(speech_chunks, sr=SAMPLE_RATE)
    if not merged:
        merged = [{"start": 0, "end": len(audio)}]

    clips: list[tuple[np.ndarray, int]] = []
    for item in merged:
        clip = audio[item["start"] : item["end"]]
        if clip.size < 800:
            continue
        clips.append((clip, SAMPLE_RATE))
    return clips


def transcribe_clips(clips: list[tuple[np.ndarray, int]]) -> list[str]:
    if not clips:
        return []
    with _INFER_LOCK:
        results = get_model().transcribe(clips, context=ASR_CONTEXT)

    texts: list[str] = []
    for result in results:
        text = (getattr(result, "text", "") or "").strip()
        if text:
            texts.append(text)
    return texts


def append_without_overlap(
    base_text: str, addition_text: str, min_overlap_chars: int = 12
) -> str:
    if not base_text:
        return addition_text
    if not addition_text:
        return base_text

    addition = addition_text.lstrip()
    max_overlap = min(len(base_text), len(addition))
    for overlap in range(max_overlap, min_overlap_chars - 1, -1):
        if base_text.endswith(addition[:overlap]):
            return base_text + addition[overlap:]
    if base_text.endswith(addition):
        return base_text
    if not base_text.endswith((" ", "\n")) and not addition.startswith((" ", "\n", ".", ",", "?", "!")):
        return f"{base_text} {addition}"
    return base_text + addition


def stable_delta(previous_text: str, next_text: str) -> tuple[str, str]:
    if not next_text:
        return previous_text, ""
    if next_text.startswith(previous_text):
        return next_text, next_text[len(previous_text) :]
    return next_text, ""


@dataclass
class SessionState:
    websocket: WebSocket | None = None
    event_sink: Callable[[dict], Awaitable[None]] | None = None
    session_id: str = field(default_factory=lambda: f"rt-{uuid4()}")
    model_name: str = MODEL_NAME
    audio: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    processed_samples: int = 0
    committed_text: str = ""
    visible_text: str = ""
    final_requested: bool = False
    decode_event: asyncio.Event = field(default_factory=asyncio.Event)
    closed: bool = False
    generation_task: asyncio.Task | None = None
    last_partial_audio_samples: int = 0

    async def send_json(self, payload: dict) -> None:
        if self.closed:
            return
        if self.websocket is not None:
            await self.websocket.send_text(json.dumps(payload, ensure_ascii=False))
            return
        if self.event_sink is not None:
            await self.event_sink(payload)

    async def send_visible_text(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        updated, delta = stable_delta(self.visible_text, text)
        self.visible_text = updated
        await self.send_json({"type": "transcription.partial", "text": text})
        if delta:
            await self.send_json({"type": "transcription.delta", "delta": delta})


@dataclass
class FileJob:
    job_id: str
    path: Path
    filename: str
    queue: asyncio.Queue[str] = field(default_factory=asyncio.Queue)
    text: str = ""
    done: bool = False
    error: str | None = None
    cancel_requested: bool = False
    total_segments: int = 0
    completed_segments: int = 0


def set_job(job: FileJob) -> None:
    with _JOBS_LOCK:
        _JOBS[job.job_id] = job


def get_job(job_id: str) -> FileJob | None:
    with _JOBS_LOCK:
        return _JOBS.get(job_id)


def decode_audio_mono_16k(audio_file: Path) -> np.ndarray:
    chunks: list[np.ndarray] = []
    with av.open(str(audio_file)) as container:
        stream = container.streams.audio[0]
        resampler = av.audio.resampler.AudioResampler(
            format="fltp",
            layout="mono",
            rate=SAMPLE_RATE,
        )
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                arr = np.asarray(out.to_ndarray(), dtype=np.float32)
                if arr.ndim == 2:
                    arr = arr[0]
                chunks.append(arr)
    if not chunks:
        raise ValueError("Failed to decode audio.")
    audio = np.concatenate(chunks).reshape(-1)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.0:
        audio = audio / peak
    return audio.astype(np.float32, copy=False)


def decode_audio_bytes_mono_16k(data: bytes) -> np.ndarray:
    chunks: list[np.ndarray] = []
    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.audio[0]
        resampler = av.audio.resampler.AudioResampler(
            format="fltp",
            layout="mono",
            rate=SAMPLE_RATE,
        )
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                arr = np.asarray(out.to_ndarray(), dtype=np.float32)
                if arr.ndim == 2:
                    arr = arr[0]
                chunks.append(arr)
    if not chunks:
        raise ValueError("Failed to decode audio.")
    audio = np.concatenate(chunks).reshape(-1)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.0:
        audio = audio / peak
    return audio.astype(np.float32, copy=False)


def build_transcription_response(
    *,
    text: str,
    response_format: str,
    language: str | None = None,
    duration_s: float | None = None,
) -> JSONResponse | PlainTextResponse:
    if response_format == "text":
        return PlainTextResponse(text)
    if response_format == "verbose_json":
        return JSONResponse(
            {
                "task": "transcribe",
                "language": language or "ko",
                "duration": duration_s,
                "text": text,
                "segments": [],
            }
        )
    if response_format in {"json", None}:
        return JSONResponse(
            {
                "text": text,
            }
        )
    raise HTTPException(status_code=400, detail=f"unsupported response_format: {response_format}")


async def emit_job_event(job: FileJob, payload: dict) -> None:
    await job.queue.put(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n")


async def run_file_job(job: FileJob) -> None:
    try:
        audio = await asyncio.to_thread(decode_audio_mono_16k, job.path)
        job.total_segments = max(1, int(np.ceil(audio.size / COMMIT_SAMPLES)))
        await emit_job_event(
            job,
            {
                "type": "file.progress",
                "job_id": job.job_id,
                "completed_segments": 0,
                "total_segments": job.total_segments,
                "text": "",
            },
        )
        async def sink(payload: dict) -> None:
            payload = dict(payload)
            payload["job_id"] = job.job_id
            if payload.get("type") == "transcription.partial":
                job.text = payload.get("text", "")
                payload["completed_segments"] = job.completed_segments
                payload["total_segments"] = job.total_segments
            elif payload.get("type") == "transcription.done":
                job.text = payload.get("text", "")
            await emit_job_event(job, payload)

        state = SessionState(event_sink=sink)
        state.generation_task = asyncio.create_task(generation_loop(state))

        feed_chunk_samples = 3200
        total_samples = int(audio.size)
        cursor = 0
        while cursor < total_samples:
            if job.cancel_requested:
                raise RuntimeError("cancelled")
            next_cursor = min(total_samples, cursor + feed_chunk_samples)
            state.audio = np.concatenate((state.audio, audio[cursor:next_cursor]), axis=0)
            cursor = next_cursor
            state.decode_event.set()
            job.completed_segments = min(
                job.total_segments,
                max(job.completed_segments, int(np.ceil(cursor / COMMIT_SAMPLES))),
            )
            await emit_job_event(
                job,
                {
                    "type": "file.progress",
                    "job_id": job.job_id,
                    "completed_segments": job.completed_segments,
                    "total_segments": job.total_segments,
                    "text": job.text,
                    "audio_progress": round(cursor / max(total_samples, 1), 4),
                },
            )
            await asyncio.sleep(0)

        state.final_requested = True
        state.decode_event.set()
        await state.generation_task
        job.text = state.visible_text
        job.done = True
        await emit_job_event(
            job,
            {
                "type": "file.progress",
                "job_id": job.job_id,
                "completed_segments": job.total_segments,
                "total_segments": job.total_segments,
                "text": job.text,
                "audio_progress": 1.0,
            },
        )
    except Exception as exc:
        job.error = str(exc)
        await emit_job_event(
            job,
            {
                "type": "error",
                "job_id": job.job_id,
                "error": job.error,
                "code": "processing_error",
            },
        )
    finally:
        job.done = True
        await job.queue.put("event: close\ndata: {}\n\n")


async def maybe_transcribe(audio: np.ndarray) -> str:
    if audio.size == 0:
        return ""
    return await asyncio.to_thread(_transcribe_with_activity, audio.copy())


def _transcribe_with_activity(audio: np.ndarray) -> str:
    inference_started()
    try:
        return transcribe_audio(audio)
    finally:
        inference_finished()


def _transcribe_clips_with_activity(clips: list[tuple[np.ndarray, int]]) -> list[str]:
    inference_started()
    try:
        return transcribe_clips(clips)
    finally:
        inference_finished()


async def generation_loop(state: SessionState) -> None:
    try:
        while not state.closed:
            await state.decode_event.wait()
            state.decode_event.clear()

            while True:
                available = state.audio.size - state.processed_samples
                if available < COMMIT_SAMPLES:
                    break

                segment = state.audio[
                    state.processed_samples : state.processed_samples + COMMIT_SAMPLES
                ]
                segment_text = await maybe_transcribe(segment)
                if segment_text:
                    state.committed_text = append_without_overlap(
                        state.committed_text, segment_text
                    )
                    await state.send_visible_text(state.committed_text)
                state.processed_samples += COMMIT_SAMPLES
                state.last_partial_audio_samples = state.processed_samples

            remaining = state.audio.size - state.processed_samples
            should_emit_partial = (
                remaining >= PARTIAL_MIN_SAMPLES
                and (
                    state.audio.size - state.last_partial_audio_samples >= PARTIAL_STEP_SAMPLES
                    or not state.visible_text
                )
            )
            if should_emit_partial:
                partial_audio = state.audio[state.processed_samples :]
                partial_text = await maybe_transcribe(partial_audio)
                visible = append_without_overlap(state.committed_text, partial_text)
                await state.send_visible_text(visible)
                state.last_partial_audio_samples = state.audio.size

            if state.final_requested:
                tail_audio = state.audio[state.processed_samples :]
                tail_text = await maybe_transcribe(tail_audio)
                final_text = append_without_overlap(state.committed_text, tail_text)
                await state.send_visible_text(final_text)
                await state.send_json(
                    {
                        "type": "transcription.done",
                        "text": final_text,
                        "usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": len(final_text),
                            "total_tokens": len(final_text),
                        },
                    }
                )
                return
    except Exception as exc:
        await state.send_json({"type": "error", "error": str(exc), "code": "processing_error"})


@app.on_event("startup")
async def startup_event() -> None:
    app.state.idle_task = asyncio.create_task(idle_unload_loop())


@app.on_event("shutdown")
async def shutdown_event() -> None:
    idle_task = getattr(app.state, "idle_task", None)
    if idle_task is not None:
        idle_task.cancel()
        try:
            await idle_task
        except asyncio.CancelledError:
            pass


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(UI_DIR / "index.html")


@app.get("/health")
@app.get("/healthz")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
async def models() -> JSONResponse:
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": MODEL_NAME,
                    "object": "model",
                    "owned_by": "local-qwen-asr",
                }
            ],
        }
    )


@app.post("/v1/audio/transcriptions", response_model=None)
async def create_audio_transcription(
    file: UploadFile = File(...),
    model: str = Form(default=MODEL_NAME),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    stream: str | None = Form(default=None),
    temperature: float | None = Form(default=None),
):
    del prompt, temperature
    if model != MODEL_NAME:
        raise HTTPException(status_code=404, detail=f"unknown model: {model}")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")

    audio = await asyncio.to_thread(decode_audio_bytes_mono_16k, data)
    duration_s = round(audio.size / SAMPLE_RATE, 3)
    stream_enabled = str(stream).lower() == "true" if stream is not None else False

    if not stream_enabled:
        text = await maybe_transcribe(audio)
        return build_transcription_response(
            text=text,
            response_format=response_format,
            language=language or "ko",
            duration_s=duration_s,
        )

    async def event_iter():
        clips = build_speech_clips(audio)
        if not clips:
            payload = {"type": "transcription.done", "text": "", "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return

        committed_text = ""
        total = len(clips)
        for idx, clip in enumerate(clips, start=1):
            texts = await asyncio.to_thread(_transcribe_clips_with_activity, [clip])
            segment_text = texts[0] if texts else ""
            if segment_text:
                committed_text = append_without_overlap(committed_text, segment_text)
                partial_payload = {
                    "type": "transcription.partial",
                    "text": committed_text,
                    "completed_segments": idx,
                    "total_segments": total,
                }
                yield f"data: {json.dumps(partial_payload, ensure_ascii=False)}\n\n"

        done_payload = {
            "type": "transcription.done",
            "text": committed_text,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": len(committed_text),
                "total_tokens": len(committed_text),
            },
        }
        yield f"data: {json.dumps(done_payload, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_iter(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.post("/v1/audio/translations")
async def create_audio_translation() -> JSONResponse:
    raise HTTPException(status_code=400, detail="translations are not supported for this backend")


@app.post("/v1/file-transcriptions")
async def create_file_transcription(file: UploadFile = File(...)) -> JSONResponse:
    suffix = Path(file.filename or "upload.bin").suffix or ".bin"
    job_id = f"file-{uuid4()}"
    path = UPLOAD_DIR / f"{job_id}{suffix}"
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")
    path.write_bytes(data)
    job = FileJob(job_id=job_id, path=path, filename=file.filename or path.name)
    set_job(job)
    asyncio.create_task(run_file_job(job))
    return JSONResponse(
        {
            "job_id": job.job_id,
            "filename": job.filename,
            "events_url": f"/v1/file-transcriptions/{job.job_id}/events",
        }
    )


@app.get("/v1/file-transcriptions/{job_id}/events")
async def file_transcription_events(job_id: str) -> StreamingResponse:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    async def event_iter():
        while True:
            chunk = await job.queue.get()
            yield chunk
            if job.done and chunk.startswith("event: close"):
                break

    return StreamingResponse(
        event_iter(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.post("/v1/file-transcriptions/{job_id}/cancel")
async def cancel_file_transcription(job_id: str) -> JSONResponse:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    job.cancel_requested = True
    return JSONResponse({"job_id": job_id, "status": "cancelling"})


@app.websocket("/v1/realtime")
async def realtime_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    state = SessionState(websocket=websocket)
    await state.send_json({"type": "session.created", "id": state.session_id})

    try:
        while True:
            message = json.loads(await websocket.receive_text())
            msg_type = message.get("type")
            if msg_type == "session.update":
                state.model_name = message.get("model") or MODEL_NAME
                continue

            if msg_type == "input_audio_buffer.append":
                audio_bytes = base64.b64decode(message["audio"])
                audio_array = (
                    np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                )
                if audio_array.size:
                    state.audio = np.concatenate((state.audio, audio_array), axis=0)
                    state.decode_event.set()
                continue

            if msg_type == "input_audio_buffer.commit":
                if state.generation_task is None:
                    state.generation_task = asyncio.create_task(generation_loop(state))
                if message.get("final"):
                    state.final_requested = True
                state.decode_event.set()
                continue

            await state.send_json(
                {"type": "error", "error": f"unknown event type: {msg_type}", "code": "unknown_event"}
            )
    except WebSocketDisconnect:
        pass
    finally:
        state.closed = True
        state.decode_event.set()
        if state.generation_task is not None:
            try:
                await asyncio.wait_for(state.generation_task, timeout=1.0)
            except Exception:
                state.generation_task.cancel()


def main() -> None:
    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "3003")),
        log_level="info",
    )


if __name__ == "__main__":
    main()
