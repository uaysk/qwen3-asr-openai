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

## API Usage

Base URL defaults to `http://127.0.0.1:8000`. If you run the systemd service configured earlier, replace it with `http://127.0.0.1:3003`.

Check health:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/healthz
```

List models:

```bash
curl http://127.0.0.1:8000/v1/models
```

List supported languages:

```bash
curl http://127.0.0.1:8000/v1/languages
```

Notes:

- `model` must match the server model name returned by `/v1/models`. The default is `qwen3-asr-rt`.
- `language` and `secondary_language` are optional.
- `prompt` is supported for offline and realtime requests as request-specific system context.
- Request `prompt` is appended after the server's built-in conservative transcription instructions. It does not replace the base transcription guardrails.
- `POST /v1/audio/translations` is not supported and returns `400`.

### Offline Transcription

Basic non-streaming transcription:

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F "file=@sample.wav" \
  -F "model=qwen3-asr-rt" \
  -F "response_format=json"
```

With explicit language hint:

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F "file=@sample.wav" \
  -F "model=qwen3-asr-rt" \
  -F "language=Korean" \
  -F "secondary_language=English" \
  -F "response_format=verbose_json"
```

With a request-specific prompt:

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F "file=@sample.wav" \
  -F "model=qwen3-asr-rt" \
  -F "language=Korean" \
  -F "prompt=Prefer CS terms in English spelling. Keep professor names exactly as spoken." \
  -F "response_format=json"
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

### Streaming Transcription Over HTTP

`stream=true` returns Server-Sent Events. This mode sends progressive text updates, but it does not support timestamps.

```bash
curl -N -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F "file=@sample.wav" \
  -F "model=qwen3-asr-rt" \
  -F "prompt=Keep product names in English spelling." \
  -F "stream=true"
```

Typical SSE events:

```text
data: {"type":"transcription.partial","text":"Hello every","completed_segments":1,"total_segments":3}

data: {"type":"transcription.partial","text":"Hello everyone. Let's get","completed_segments":2,"total_segments":3}

data: {"type":"transcription.done","text":"Hello everyone. Let's get started.","usage":{"prompt_tokens":0,"completion_tokens":36,"total_tokens":36}}

data: [DONE]
```

### File Transcription Jobs

`POST /v1/file-transcriptions` is the recommended path for long offline files and for the bundled web UI. The server creates a job, then you read progress from the job's SSE endpoint.

Create a job:

```bash
curl -X POST http://127.0.0.1:8000/v1/file-transcriptions \
  -F "file=@lecture.webm" \
  -F "language=Korean" \
  -F "secondary_language=English" \
  -F "prompt=Keep networking terms in English and preserve classroom filler words." \
  -F "timestamps=true"
```

Example response:

```json
{
  "job_id": "file-123",
  "filename": "lecture.webm",
  "events_url": "/v1/file-transcriptions/file-123/events"
}
```

Stream job events:

```bash
curl -N http://127.0.0.1:8000/v1/file-transcriptions/file-123/events
```

Typical event flow:

- `transcription.partial`: progressive combined text
- `file.progress`: audio processing progress
- `transcription.done`: final text and optional `segments` and `words`
- `error`: asynchronous job failure
- `event: close`: end of stream

Example final file transcription SSE event:

```text
data: {"type":"transcription.done","job_id":"file-123","text":"Hello everyone. Let's get started.","language":"English","segments":[{"start":0.12,"end":1.84,"text":"Hello everyone."},{"start":2.01,"end":4.62,"text":"Let's get started."}],"words":[{"text":"Hello","start":0.12,"end":0.56}],"usage":{"prompt_tokens":0,"completion_tokens":36,"total_tokens":36}}
```

Cancel a running job:

```bash
curl -X POST http://127.0.0.1:8000/v1/file-transcriptions/file-123/cancel
```

### Realtime WebSocket

Realtime transcription uses `WS /v1/realtime`. This is the path used by the demo UI for microphone input and browser-side file replay.

Minimal message flow:

1. Connect to `ws://127.0.0.1:8000/v1/realtime`
2. Receive `{"type":"session.created","id":"..."}`
3. Optionally send session settings:

```json
{"type":"session.update","model":"qwen3-asr-rt","language":"Korean","secondary_language":"English","prompt":"Keep technical terms in English spelling."}
```

4. Send PCM16 mono 16 kHz audio chunks:

```json
{"type":"input_audio_buffer.append","audio":"<base64 pcm16 bytes>"}
```

5. Ask the server to decode buffered audio:

```json
{"type":"input_audio_buffer.commit"}
```

6. When finished, send:

```json
{"type":"input_audio_buffer.commit","final":true}
```

Typical server events:

- `transcription.partial`
- `transcription.delta`
- `transcription.done`
- `error`

Realtime WebSocket mode does not provide timestamps.

Realtime notes:

- `session.update` accepts `prompt` and `context`. They are treated the same way.
- The request prompt applies only to the current WebSocket session.

## Web UI Usage

Open the demo UI at `http://127.0.0.1:8000/` or `http://127.0.0.1:3003/` if you are using the systemd service.

The three main actions use different backend paths:

- `Start Mic`: opens `WS /v1/realtime` and transcribes live microphone audio.
- `Play File`: opens `WS /v1/realtime`, decodes the selected file in the browser, and replays it into the realtime endpoint.
- `Transcribe File`: uploads the selected file to `POST /v1/file-transcriptions` with `timestamps=true`, shows progressive text, then renders the final full transcript plus a separate timestamp timeline.

Recommended WEBUI flow:

1. Open `/`
2. Choose primary and optional secondary language
3. Optionally fill `Request Prompt` if you want request-specific transcription instructions
4. Select an audio file if you plan to use `Play File` or `Transcribe File`
5. Use one of the buttons based on your goal:

- `Start Mic` for live microphone transcription
- `Play File` to simulate realtime transcription from a local file
- `Transcribe File` for full offline transcription with timestamps

Current WEBUI timestamp behavior:

- Timestamps are shown only for `Transcribe File`
- The exact final transcript is displayed separately from the timestamp timeline
- `Start Mic` and `Play File` do not show timestamps
- `Request Prompt` is sent to both realtime WebSocket sessions and offline file transcription jobs

## Notes

- Pascal and other pre-Ampere GPUs default to `float16 + eager` mode.
- Ampere and newer GPUs default to `sdpa`, and prefer `bfloat16`.
- Blackwell-class GPUs use the `cu128` PyTorch path.
- CPU-only installs are allowed, but throughput will be much lower.
