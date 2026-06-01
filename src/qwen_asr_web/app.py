from __future__ import annotations

import shutil
import uuid
import os
from importlib.resources import files
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .asr import ASRService, default_checkpoint, project_root
from .jobs import Job, JobStore
from .llm import LLMService


ROOT = project_root()
STATIC_DIR = Path(str(files("qwen_asr_web").joinpath("static")))
UPLOAD_DIR = ROOT / "data" / "uploads"
OUTPUT_DIR = ROOT / "data" / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

service = ASRService(
    max_new_tokens=int(os.environ.get("ASR_MAX_NEW_TOKENS", "4096")),
    chunk_seconds=int(os.environ.get("ASR_CHUNK_SECONDS", "60")),
)
llm_service = LLMService()
jobs = JobStore()
app = FastAPI(title="Qwen3-ASR Web", version=__version__)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    import torch
    import transformers

    return {
        "app": __version__,
        "checkpoint": service.checkpoint,
        "checkpoint_exists": (Path(service.checkpoint) / "config.json").is_file(),
        "default_checkpoint": default_checkpoint(ROOT),
        "model_loaded": service._model is not None,
        "asr_max_new_tokens": service.max_new_tokens,
        "asr_chunk_seconds": service.chunk_seconds,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "transformers": transformers.__version__,
        "ffmpeg": shutil.which("ffmpeg"),
        "llm_configured": llm_service.configured,
        "llm_provider": llm_service.provider,
        "llm_base_url": llm_service.base_url,
        "llm_model": llm_service.model,
    }


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form(""),
    context: str = Form(""),
) -> JSONResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    suffix = Path(file.filename).suffix
    safe_name = f"{uuid.uuid4().hex}{suffix}"
    upload_path = UPLOAD_DIR / safe_name
    job_id = upload_path.stem
    job = jobs.create(job_id, "transcribe")
    job.log(f"收到文件：{file.filename}")

    try:
        with upload_path.open("wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)
        job.progress = 10
        job.stage = "uploaded"
        job.log(f"文件已保存：{upload_path.name} ({upload_path.stat().st_size / 1024 / 1024:.1f} MB)")

        def run(job: Job) -> None:
            job.stage = "preparing"
            job.progress = 20
            job.log(f"ASR checkpoint: {service.checkpoint}")
            job.log("准备加载模型；首次运行可能需要较长时间")

            result = service.transcribe_file(
                upload_path,
                language=language.strip() or None,
                context=context.strip(),
                keep_extracted_audio=True,
                progress_callback=lambda message, progress=None: update_job(job, message, progress),
                text_callback=lambda text, index, total: append_job_text(job, text, index, total),
            )

            job.stage = "writing"
            job.progress = 90
            job.log("转写完成，正在写出结果文件")
            txt_path = OUTPUT_DIR / f"{job.id}.txt"
            json_path = OUTPUT_DIR / f"{job.id}.json"
            wav_path = OUTPUT_DIR / f"{job.id}.16k.wav"
            txt_path.write_text(result.text, encoding="utf-8")
            json_path.write_text(result.to_json(), encoding="utf-8")
            audio_source = Path(result.audio)
            audio_url = None
            if audio_source.is_file() and audio_source.suffix.lower() == ".wav":
                shutil.copy2(audio_source, wav_path)
                audio_url = f"/outputs/{wav_path.name}"
            job.result = {
                "id": job.id,
                "language": result.language,
                "text": result.text,
                "summary": None,
                "text_url": f"/outputs/{txt_path.name}",
                "json_url": f"/outputs/{json_path.name}",
                "audio_url": audio_url,
                "input_duration_sec": result.input_duration_sec,
                "audio_duration_sec": result.audio_duration_sec,
                "text_chars": len(result.text),
                "max_new_tokens": result.max_new_tokens,
                "chunk_seconds": result.chunk_seconds,
            }
            job.log(f"识别语言：{result.language or 'unknown'}")
            if result.input_duration_sec is not None and result.audio_duration_sec is not None:
                job.log(
                    f"时长核对：输入 {result.input_duration_sec:.1f}s / "
                    f"ASR {result.audio_duration_sec:.1f}s"
                )
            job.log(f"输出字符数：{len(result.text)}")

        jobs.start(job, run)
        return JSONResponse(
            {
                "id": job.id,
                "status": job.status,
                "progress": job.progress,
                "status_url": f"/api/jobs/{job.id}",
            }
        )
    except RuntimeError as exc:
        job.status = "failed"
        job.stage = "upload_failed"
        job.error = str(exc)
        job.log(f"上传失败：{exc}")
        raise HTTPException(status_code=500, detail={"message": str(exc), "job": job.snapshot()}) from exc
    except Exception as exc:
        job.status = "failed"
        job.stage = "upload_failed"
        job.error = f"Upload failed: {exc}"
        job.log(job.error)
        raise HTTPException(status_code=500, detail={"message": job.error, "job": job.snapshot()}) from exc


def update_job(job: Job, message: str, progress: int | None = None) -> None:
    if progress is not None:
        job.progress = max(job.progress, min(int(progress), 99))
    job.log(message)


def append_job_text(job: Job, text: str, index: int, total: int) -> None:
    job.partial_text = f"{job.partial_text}\n{text}".strip()
    job.progress = max(job.progress, 50 + int(35 * index / max(total, 1)))
    job.log(f"第 {index}/{total} 段完成，累计 {len(job.partial_text)} 字")


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    snapshot = jobs.snapshot(job_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JSONResponse(snapshot)


@app.post("/api/summarize")
async def summarize(payload: dict) -> JSONResponse:
    text = str(payload.get("text") or "")
    instruction = str(payload.get("instruction") or "")

    try:
        result = llm_service.summarize(text=text, instruction=instruction)
        output_id = uuid.uuid4().hex
        summary_path = OUTPUT_DIR / f"{output_id}.summary.md"
        json_path = OUTPUT_DIR / f"{output_id}.summary.json"
        summary_path.write_text(result.summary, encoding="utf-8")
        json_path.write_text(result.to_json(), encoding="utf-8")

        return JSONResponse(
            {
                "summary": result.summary,
                "provider": result.provider,
                "base_url": result.base_url,
                "model": result.model,
                "summary_url": f"/outputs/{summary_path.name}",
                "json_url": f"/outputs/{json_path.name}",
            }
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Summary failed: {exc}") from exc


app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
