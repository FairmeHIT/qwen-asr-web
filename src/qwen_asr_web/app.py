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


ROOT = project_root()
STATIC_DIR = Path(str(files("qwen_asr_web").joinpath("static")))
UPLOAD_DIR = ROOT / "data" / "uploads"
OUTPUT_DIR = ROOT / "data" / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

service = ASRService()
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
                "text_url": f"/outputs/{txt_path.name}",
                "json_url": f"/outputs/{json_path.name}",
            }
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc


app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
