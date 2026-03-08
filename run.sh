#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage:
  ./run.sh [options]

Options:
  --host HOST
  --port PORT
  --model-id MODEL_ID
  --model-name MODEL_NAME
  --device-map DEVICE_MAP
  --dtype DTYPE
  --attn-implementation NAME
  --max-inference-batch-size N
  --commit-seconds SECONDS
  --partial-min-seconds SECONDS
  --partial-step-seconds SECONDS
  --upload-dir PATH
  --idle-unload-seconds SECONDS
  --idle-check-seconds SECONDS
  --asr-context TEXT
  --local-files-only
  --allow-model-download
  --print-env
  -h, --help

Environment-backed options:
  HOST
  PORT
  QWEN_RT_MODEL_ID
  QWEN_RT_MODEL_NAME
  QWEN_RT_DEVICE_MAP
  QWEN_RT_MODEL_DTYPE
  QWEN_RT_ATTN_IMPLEMENTATION
  QWEN_RT_MAX_INFERENCE_BATCH_SIZE
  QWEN_RT_COMMIT_SECONDS
  QWEN_RT_PARTIAL_MIN_SECONDS
  QWEN_RT_PARTIAL_STEP_SECONDS
  QWEN_RT_UPLOAD_DIR
  QWEN_RT_IDLE_UNLOAD_SECONDS
  QWEN_RT_IDLE_CHECK_SECONDS
  QWEN_RT_ASR_CONTEXT
  QWEN_RT_LOCAL_FILES_ONLY
  HF_HUB_OFFLINE
  TRANSFORMERS_OFFLINE
EOF
}

PRINT_ENV=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      export HOST="$2"
      shift 2
      ;;
    --port)
      export PORT="$2"
      shift 2
      ;;
    --model-id)
      export QWEN_RT_MODEL_ID="$2"
      shift 2
      ;;
    --model-name)
      export QWEN_RT_MODEL_NAME="$2"
      shift 2
      ;;
    --device-map)
      export QWEN_RT_DEVICE_MAP="$2"
      shift 2
      ;;
    --dtype)
      export QWEN_RT_MODEL_DTYPE="$2"
      shift 2
      ;;
    --attn-implementation)
      export QWEN_RT_ATTN_IMPLEMENTATION="$2"
      shift 2
      ;;
    --max-inference-batch-size)
      export QWEN_RT_MAX_INFERENCE_BATCH_SIZE="$2"
      shift 2
      ;;
    --commit-seconds)
      export QWEN_RT_COMMIT_SECONDS="$2"
      shift 2
      ;;
    --partial-min-seconds)
      export QWEN_RT_PARTIAL_MIN_SECONDS="$2"
      shift 2
      ;;
    --partial-step-seconds)
      export QWEN_RT_PARTIAL_STEP_SECONDS="$2"
      shift 2
      ;;
    --upload-dir)
      export QWEN_RT_UPLOAD_DIR="$2"
      shift 2
      ;;
    --idle-unload-seconds)
      export QWEN_RT_IDLE_UNLOAD_SECONDS="$2"
      shift 2
      ;;
    --idle-check-seconds)
      export QWEN_RT_IDLE_CHECK_SECONDS="$2"
      shift 2
      ;;
    --asr-context)
      export QWEN_RT_ASR_CONTEXT="$2"
      shift 2
      ;;
    --local-files-only)
      export QWEN_RT_LOCAL_FILES_ONLY="true"
      shift
      ;;
    --allow-model-download)
      export QWEN_RT_LOCAL_FILES_ONLY="false"
      shift
      ;;
    --print-env)
      PRINT_ENV=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      echo >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -f "$ROOT_DIR/.env.runtime" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env.runtime"
fi

export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-3003}"
export QWEN_RT_LOCAL_FILES_ONLY="${QWEN_RT_LOCAL_FILES_ONLY:-false}"

if [[ "$PRINT_ENV" == "1" ]]; then
  cat <<EOF
HOST=$HOST
PORT=$PORT
QWEN_RT_MODEL_ID=${QWEN_RT_MODEL_ID:-}
QWEN_RT_MODEL_NAME=${QWEN_RT_MODEL_NAME:-}
QWEN_RT_DEVICE_MAP=${QWEN_RT_DEVICE_MAP:-}
QWEN_RT_MODEL_DTYPE=${QWEN_RT_MODEL_DTYPE:-}
QWEN_RT_ATTN_IMPLEMENTATION=${QWEN_RT_ATTN_IMPLEMENTATION:-}
QWEN_RT_MAX_INFERENCE_BATCH_SIZE=${QWEN_RT_MAX_INFERENCE_BATCH_SIZE:-}
QWEN_RT_COMMIT_SECONDS=${QWEN_RT_COMMIT_SECONDS:-}
QWEN_RT_PARTIAL_MIN_SECONDS=${QWEN_RT_PARTIAL_MIN_SECONDS:-}
QWEN_RT_PARTIAL_STEP_SECONDS=${QWEN_RT_PARTIAL_STEP_SECONDS:-}
QWEN_RT_UPLOAD_DIR=${QWEN_RT_UPLOAD_DIR:-}
QWEN_RT_IDLE_UNLOAD_SECONDS=${QWEN_RT_IDLE_UNLOAD_SECONDS:-}
QWEN_RT_IDLE_CHECK_SECONDS=${QWEN_RT_IDLE_CHECK_SECONDS:-}
QWEN_RT_LOCAL_FILES_ONLY=${QWEN_RT_LOCAL_FILES_ONLY:-}
HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-}
TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-}
EOF
fi

if [[ ! -x "$ROOT_DIR/.venv/bin/python" ]]; then
  echo "missing virtualenv: run ./install.sh first" >&2
  exit 1
fi

exec "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/server.py"
