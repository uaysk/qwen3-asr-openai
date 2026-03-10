# qwen3-asr-openai

Direct `Qwen3-ASR` server with:

- `GET /v1/models`
- `POST /v1/audio/transcriptions`
- `POST /v1/audio/transcriptions` with `stream=true`
- `WS /v1/realtime`
- bundled demo UI on `/`

## Install

```bash
cd /home/uaysk/llama-swap/qwen3-asr-openai
./install.sh
```

What the installer does:

- detects NVIDIA GPU, compute capability, and driver version
- chooses a matching PyTorch wheel family
- creates `.venv`
- installs server dependencies
- writes `.env.runtime` with runtime settings for this hardware
- downloads `Qwen/Qwen3-ASR-1.7B`
- downloads `Qwen/Qwen3-ForcedAligner-0.6B`
- if the ASR model is cached, switches runtime to local-files-only offline mode so `run.sh` does not fetch the model again
- if the aligner is cached, writes `QWEN_RT_ALIGNER_PATH` so offline timestamps work immediately

Useful options:

```bash
./install.sh --dry-run
./install.sh --skip-model-download
./install.sh --skip-aligner-download
```

## Run

```bash
cd /home/uaysk/llama-swap/qwen3-asr-openai
./run.sh
```

The server listens on `0.0.0.0:8000` by default.

Examples:

```bash
./run.sh --port 3010
./run.sh --port 3010 --idle-unload-seconds 120
./run.sh --host 127.0.0.1 --local-files-only
./run.sh --print-env --port 3010
```

List runtime flags:

```bash
./run.sh --help
```

Offline timestamps:

- `POST /v1/audio/transcriptions` supports `timestamps=true` for non-streaming requests.
- When enabled, `json` and `verbose_json` responses include `segments`, and `words` when word alignment is available.
- The default installer downloads `Qwen/Qwen3-ForcedAligner-0.6B` and writes `QWEN_RT_ALIGNER_PATH` automatically.
- `stream=true` and `WS /v1/realtime` do not support timestamps.

Timestamped transcription request:

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F "file=@sample.wav" \
  -F "model=qwen3-asr-rt" \
  -F "response_format=json" \
  -F "timestamps=true"
```

Output format:

- `response_format=json` returns `text` and, when `timestamps=true`, also `segments`, `words`, `language`, and `duration`.
- `response_format=verbose_json` returns `task`, `language`, `duration`, `text`, `segments`, and optional `words`.
- `POST /v1/file-transcriptions` with `timestamps=true` streams progress first, then emits timestamp data in the final `transcription.done` event.

Example `response_format=json` output:

```json
{
  "text": "Hello everyone. Let's get started.",
  "language": "English",
  "duration": 4.812,
  "segments": [
    {
      "start": 0.12,
      "end": 1.84,
      "text": "Hello everyone."
    },
    {
      "start": 2.01,
      "end": 4.62,
      "text": "Let's get started."
    }
  ],
  "words": [
    {
      "text": "Hello",
      "start": 0.12,
      "end": 0.56
    },
    {
      "text": "everyone",
      "start": 0.58,
      "end": 1.31
    }
  ]
}
```

Example `response_format=verbose_json` output:

```json
{
  "task": "transcribe",
  "language": "English",
  "duration": 4.812,
  "text": "Hello everyone. Let's get started.",
  "segments": [
    {
      "start": 0.12,
      "end": 1.84,
      "text": "Hello everyone."
    },
    {
      "start": 2.01,
      "end": 4.62,
      "text": "Let's get started."
    }
  ],
  "words": [
    {
      "text": "Hello",
      "start": 0.12,
      "end": 0.56
    },
    {
      "text": "everyone",
      "start": 0.58,
      "end": 1.31
    }
  ]
}
```

Example final file transcription SSE event:

```text
data: {"type":"transcription.done","job_id":"file-123","text":"Hello everyone. Let's get started.","language":"English","segments":[{"start":0.12,"end":1.84,"text":"Hello everyone."},{"start":2.01,"end":4.62,"text":"Let's get started."}],"words":[{"text":"Hello","start":0.12,"end":0.56}],"usage":{"prompt_tokens":0,"completion_tokens":36,"total_tokens":36}}
```

## Notes

- Pascal and other pre-Ampere GPUs default to `float16 + eager` mode.
- Ampere and newer GPUs default to `sdpa`, and prefer `bfloat16`.
- Blackwell-class GPUs use the `cu128` PyTorch path.
- CPU-only installs are allowed, but throughput will be much lower.
