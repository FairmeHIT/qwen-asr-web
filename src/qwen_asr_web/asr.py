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

TRANSCODE_EXTS = VIDEO_EXTS | {
    ".m4a",
    ".mp3",
    ".aac",
    ".ogg",
    ".opus",
    ".wma",
    ".flac",
}

DIRECT_AUDIO_EXTS = {
    ".wav",
}


@dataclass(frozen=True)
class TranscriptionResult:
    input: str
    audio: str
    checkpoint: str
    language: str
    text: str
    input_duration_sec: Optional[float] = None
    audio_duration_sec: Optional[float] = None
    max_new_tokens: int = 4096

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


def media_duration_sec(input_path: Path) -> Optional[float]:
    try:
        import av

        with av.open(str(input_path)) as container:
            if container.duration is not None:
                return round(float(container.duration) / 1_000_000, 3)
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream and stream.duration and stream.time_base:
                return round(float(stream.duration * stream.time_base), 3)
    except Exception:
        return None
    return None


def audio_file_duration_sec(input_path: Path) -> Optional[float]:
    try:
        import soundfile as sf

        info = sf.info(str(input_path))
        return round(float(info.duration), 3)
    except Exception:
        return media_duration_sec(input_path)


def prepare_audio_for_asr(input_path: Path, work_dir: Path) -> tuple[Path, str]:
    suffix = input_path.suffix.lower()
    if suffix in DIRECT_AUDIO_EXTS:
        return input_path, "输入文件为 WAV，直接转写"
    if suffix not in TRANSCODE_EXTS:
        return input_path, "未知音频扩展名，直接交给 ASR 解码"

    out_path = work_dir / f"{input_path.stem}.16k.wav"
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
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
        return out_path, "已用 ffmpeg 转为 16k 单声道 WAV"

    transcode_with_pyav(input_path, out_path)
    return out_path, "已用 PyAV 转为 16k 单声道 WAV"


def transcode_with_pyav(input_path: Path, out_path: Path) -> None:
    import av
    import numpy as np
    import soundfile as sf

    chunks = []
    with av.open(str(input_path)) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise RuntimeError(f"No audio stream found in {input_path.name}.")

        resampler = av.audio.resampler.AudioResampler(
            format="fltp",
            layout="mono",
            rate=16000,
        )

        for frame in container.decode(stream):
            frames = resampler.resample(frame)
            if frames is None:
                continue
            if not isinstance(frames, list):
                frames = [frames]
            for resampled in frames:
                arr = resampled.to_ndarray()
                if arr.ndim == 2:
                    arr = arr[0]
                chunks.append(arr.astype("float32", copy=False))

    if not chunks:
        raise RuntimeError(f"Could not decode audio from {input_path.name}.")

    audio = np.concatenate(chunks)
    sf.write(str(out_path), audio, 16000)


class ASRService:
    def __init__(
        self,
        checkpoint: Optional[str] = None,
        device_map: str = "cuda:0",
        dtype: str = "bfloat16",
        batch_size: int = 4,
        max_new_tokens: int = 4096,
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
            input_duration = media_duration_sec(input_path)
            if input_duration is not None:
                report(f"输入时长：{input_duration:.1f}s", 28)
            audio_path, audio_message = prepare_audio_for_asr(input_path, temp_dir)
            audio_duration = audio_file_duration_sec(audio_path)
            if audio_path != input_path:
                report(audio_message, 35)
            else:
                report(audio_message, 35)
            if audio_duration is not None:
                report(f"ASR 音频时长：{audio_duration:.1f}s", 38)
            report("等待 ASR 推理资源", 40)
            with self._lock:
                report("加载 ASR 模型", 45)
                model = self.load()
                report(f"模型已加载，开始转写；max_new_tokens={self.max_new_tokens}", 60)
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
                input_duration_sec=input_duration,
                audio_duration_sec=audio_duration,
                max_new_tokens=self.max_new_tokens,
            )
