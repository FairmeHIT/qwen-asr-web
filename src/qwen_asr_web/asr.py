from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional


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
    configured = os.environ.get("ASR_CHECKPOINT")
    if configured:
        configured_path = Path(configured).expanduser()
        if configured_path.is_absolute() and (configured_path / "config.json").is_file():
            return str(configured_path)
        local_configured = root / configured_path
        if (local_configured / "config.json").is_file():
            return str(local_configured)

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
        self.checkpoint = self._resolve_checkpoint(checkpoint or os.environ.get("ASR_CHECKPOINT"), root)
        self.device_map = device_map
        self.dtype = dtype
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._lock = threading.Lock()

    def _resolve_checkpoint(self, checkpoint: Optional[str], root: Path) -> str:
        if checkpoint:
            path = Path(checkpoint).expanduser()
            candidates = [path] if path.is_absolute() else [root / path, path]
            for candidate in candidates:
                if (candidate / "config.json").is_file():
                    return str(candidate.resolve())
            fallback = default_checkpoint(root)
            if fallback != "Qwen/Qwen3-ASR-1.7B":
                return fallback
            return checkpoint
        return default_checkpoint(root)

    def load(self):
        if self._model is None:
            from qwen_asr import Qwen3ASRModel

            checkpoint_path = Path(self.checkpoint)
            if (
                not self.checkpoint.startswith("Qwen/")
                and not self.checkpoint.startswith("http")
                and not (checkpoint_path / "config.json").is_file()
            ):
                raise FileNotFoundError(
                    f"ASR checkpoint is invalid: {self.checkpoint}. "
                    "Set ASR_CHECKPOINT to a local model directory containing config.json."
                )

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
        progress_callback: Optional[Callable[[str, Optional[int]], None]] = None,
    ) -> TranscriptionResult:
        def report(message: str, progress: Optional[int] = None) -> None:
            if progress_callback:
                progress_callback(message, progress)

        input_path = input_path.expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        if input_path.stat().st_size == 0:
            raise ValueError(f"Input file is empty: {input_path}")

        with tempfile.TemporaryDirectory(prefix="qwen_asr_") as tmp:
            temp_dir = Path(tmp)
            report("检查输入文件", 25)
            audio_path = extract_audio_if_video(input_path, temp_dir)
            if audio_path != input_path:
                report("视频音轨已抽取为 16k 单声道 WAV", 35)
            else:
                report("输入文件按音频处理", 35)
            report("等待 ASR 推理资源", 40)
            with self._lock:
                report("加载 ASR 模型", 45)
                model = self.load()
                report("模型已加载，开始转写", 60)
                result = model.transcribe(
                    audio=str(audio_path),
                    context=context,
                    language=language,
                )[0]
            report("模型推理完成", 85)

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
