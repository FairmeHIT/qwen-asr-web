from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import uvicorn

from . import __version__
from .asr import ASRService, default_checkpoint


def add_transcribe_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", help="Audio or video file path.")
    parser.add_argument("-o", "--output", help="Output .txt path. Defaults to <input>.txt")
    parser.add_argument("--json-output", help="Optional JSON output path.")
    parser.add_argument("--checkpoint", default=os.environ.get("ASR_CHECKPOINT") or default_checkpoint())
    parser.add_argument("--language", default=None, help="Optional language, e.g. Chinese or English.")
    parser.add_argument("--context", default="", help="Optional prompt/context text.")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--keep-extracted-audio", action="store_true")


def run_transcribe(args: argparse.Namespace) -> int:
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else input_path.with_suffix(".txt")
    json_output = Path(args.json_output).expanduser().resolve() if args.json_output else None

    service = ASRService(
        checkpoint=args.checkpoint,
        device_map=args.device_map,
        dtype=args.dtype,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    result = service.transcribe_file(
        input_path,
        language=args.language,
        context=args.context,
        keep_extracted_audio=args.keep_extracted_audio,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.text, encoding="utf-8")

    if json_output:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(result.to_json(), encoding="utf-8")

    print(f"Language: {result.language}")
    print(f"Text: {output_path}")
    if json_output:
        print(f"JSON: {json_output}")

    return 0


def add_web_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--reload", action="store_true")


def run_web(args: argparse.Namespace) -> int:
    uvicorn.run("qwen_asr_web.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def run_download_model(args: argparse.Namespace) -> int:
    from modelscope import snapshot_download

    local_dir = Path(args.local_dir).expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.model_id} to {local_dir}")
    snapshot_download(args.model_id, local_dir=str(local_dir))
    print("Done")
    return 0


def run_check(_: argparse.Namespace) -> int:
    import torch
    import transformers

    print("qwen_asr_web:", __version__)
    print("transformers:", transformers.__version__)
    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
    print("ffmpeg:", shutil.which("ffmpeg") or "not found")
    print("checkpoint:", default_checkpoint())
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="qwen3-asr",
        description="Qwen3-ASR local web and CLI toolkit.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

    web_parser = sub.add_parser("web", help="Start the Web UI.")
    add_web_args(web_parser)
    web_parser.set_defaults(func=run_web)

    transcribe_parser = sub.add_parser("transcribe", help="Transcribe an audio/video file.")
    add_transcribe_args(transcribe_parser)
    transcribe_parser.set_defaults(func=run_transcribe)

    download_parser = sub.add_parser("download-model", help="Download a model from ModelScope.")
    download_parser.add_argument("model_id", nargs="?", default="Qwen/Qwen3-ASR-1.7B")
    download_parser.add_argument("local_dir", nargs="?", default="models/Qwen3-ASR-1.7B")
    download_parser.set_defaults(func=run_download_model)

    check_parser = sub.add_parser("check", help="Print environment diagnostics.")
    check_parser.set_defaults(func=run_check)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        args = parser.parse_args(["web"])
    return args.func(args)


def web_main() -> int:
    parser = argparse.ArgumentParser(description="Start the Qwen3-ASR local web app.")
    add_web_args(parser)
    return run_web(parser.parse_args())


def transcribe_main() -> int:
    parser = argparse.ArgumentParser(description="Transcribe an audio/video file with Qwen3-ASR.")
    add_transcribe_args(parser)
    return run_transcribe(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
