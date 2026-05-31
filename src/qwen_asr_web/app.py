from __future__ import annotations

import shutil
import uuid
from importlib.resources import files
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .asr import ASRService, default_checkpoint, project_root
from .llm import LLMService


ROOT = project_root()
STATIC_DIR = Path(str(files("qwen_asr_web").joinpath("static")))
UPLOAD_DIR = ROOT / "data" / "uploads"
OUTPUT_DIR = ROOT / "data" / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

service = ASRService()
llm_service = LLMService()
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
        "default_checkpoint": default_checkpoint(ROOT),
        "model_loaded": service._model is not None,
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

    try:
        with upload_path.open("wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)

        result = service.transcribe_file(
            upload_path,
            language=language.strip() or None,
            context=context.strip(),
        )

        output_id = upload_path.stem
        txt_path = OUTPUT_DIR / f"{output_id}.txt"
        json_path = OUTPUT_DIR / f"{output_id}.json"
        txt_path.write_text(result.text, encoding="utf-8")
        json_path.write_text(result.to_json(), encoding="utf-8")

        return JSONResponse(
            {
                "id": output_id,
                "language": result.language,
                "text": result.text,
                "summary": None,
                "text_url": f"/outputs/{txt_path.name}",
                "json_url": f"/outputs/{json_path.name}",
            }
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc


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
