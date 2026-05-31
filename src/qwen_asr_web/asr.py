from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


VIDEO_EXTS = {
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".flv",
    ".wmv",
    ".webm",
    ".m4v",
    ".mpeg",
    ".mpg",
}


@dataclass(frozen=True)
class TranscriptionResult:
    input: str
    audio: str
    checkpoint: str
    language: str
    text: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def project_root() -> Path:
    configured = os.environ.get("QWEN_ASR_WEB_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()

    cwd = Path.cwd().resolve()
    if (cwd / "pyproject.toml").is_file() or (cwd / "models").exists():
        return cwd

    return Path(__file__).resolve().parents[2]


def default_checkpoint(root: Optional[Path] = None) -> str:
    root = root or project_root()
    for local in (root / "models" / "Qwen3-ASR-1.7B", root / "models"):
        if (local / "config.json").is_file():
            return str(local)
    return "Qwen/Qwen3-ASR-1.7B"


def dtype_from_name(name: str):
    import torch

    value = name.lower().strip()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def extract_audio_if_video(input_path: Path, work_dir: Path) -> Path:
    if input_path.suffix.lower() not in VIDEO_EXTS:
        return input_path

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found. Install ffmpeg before transcribing video files.")

    out_path = work_dir / f"{input_path.stem}.16k.wav"
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return out_path


class ASRService:
    def __init__(
        self,
        checkpoint: Optional[str] = None,
        device_map: str = "cuda:0",
        dtype: str = "bfloat16",
        batch_size: int = 4,
        max_new_tokens: int = 512,
    ) -> None:
        root = project_root()
        self.checkpoint = checkpoint or os.environ.get("ASR_CHECKPOINT") or default_checkpoint(root)
        self.device_map = device_map
        self.dtype = dtype
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self._model = None

    def load(self):
        if self._model is None:
            from qwen_asr import Qwen3ASRModel

            self._model = Qwen3ASRModel.from_pretrained(
                self.checkpoint,
                dtype=dtype_from_name(self.dtype),
                device_map=self.device_map,
                max_inference_batch_size=self.batch_size,
                max_new_tokens=self.max_new_tokens,
            )
        return self._model

    def transcribe_file(
        self,
        input_path: Path,
        language: Optional[str] = None,
        context: str = "",
        keep_extracted_audio: bool = False,
    ) -> TranscriptionResult:
        input_path = input_path.expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(input_path)

        with tempfile.TemporaryDirectory(prefix="qwen_asr_") as tmp:
            temp_dir = Path(tmp)
            audio_path = extract_audio_if_video(input_path, temp_dir)
            model = self.load()
            result = model.transcribe(
                audio=str(audio_path),
                context=context,
                language=language,
            )[0]

            if keep_extracted_audio and audio_path.parent == temp_dir:
                kept = input_path.with_suffix(".16k.wav")
                shutil.copy2(audio_path, kept)
                audio_display = str(kept)
            else:
                audio_display = str(audio_path)

            return TranscriptionResult(
                input=str(input_path),
                audio=audio_display,
                checkpoint=self.checkpoint,
                language=result.language or "",
                text=result.text or "",
            )
