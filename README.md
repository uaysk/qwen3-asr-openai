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
- optionally downloads `Qwen/Qwen3-ASR-1.7B`
- if model download succeeds, switches runtime to local-files-only offline mode so `run.sh` does not fetch the model again

Useful options:

```bash
./install.sh --dry-run
./install.sh --skip-model-download
```

## Run

```bash
cd /home/uaysk/llama-swap/qwen3-asr-openai
./run.sh
```

The server listens on `0.0.0.0:3003` by default.

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

## Notes

- Pascal and other pre-Ampere GPUs default to `float16 + eager` mode.
- Ampere and newer GPUs default to `sdpa`, and prefer `bfloat16`.
- Blackwell-class GPUs use the `cu128` PyTorch path.
- CPU-only installs are allowed, but throughput will be much lower.
