#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export PYTHONPATH="${PYTHONPATH:-$PWD/src}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

usage() {
  cat <<'EOF'
Usage:
  ./run.sh web [--host 0.0.0.0] [--port 8000]
  ./run.sh transcribe <audio_or_video> [args...]
  ./run.sh summarize <text_file> [args...]
  ./run.sh download-model [model_id] [local_dir]
  ./run.sh check

Examples:
  ./run.sh web
  ./run.sh transcribe input.wav -o output.txt --language Chinese
  ./run.sh summarize output.txt -o output.summary.md
  ./run.sh download-model Qwen/Qwen3-ASR-1.7B models/Qwen3-ASR-1.7B
EOF
}

cmd="${1:-web}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$cmd" in
  web)
    exec .venv/bin/python -m qwen_asr_web.cli web "$@"
    ;;
  transcribe)
    exec .venv/bin/python -m qwen_asr_web.cli transcribe "$@"
    ;;
  summarize)
    exec .venv/bin/python -m qwen_asr_web.cli summarize "$@"
    ;;
  download-model)
    model_id="${1:-Qwen/Qwen3-ASR-1.7B}"
    local_dir="${2:-$PWD/models/Qwen3-ASR-1.7B}"
    .venv/bin/python -m qwen_asr_web.cli download-model "$model_id" "$local_dir"
    ;;
  check)
    exec .venv/bin/python -m qwen_asr_web.cli check
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown command: $cmd" >&2
    usage >&2
    exit 2
    ;;
esac
