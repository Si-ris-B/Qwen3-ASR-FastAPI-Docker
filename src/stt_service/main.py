import io
import librosa
import time
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from .services.model_manager import qwen_manager
from .models.schemas import TranscriptionResponse, LoadModelRequest
from .core.config import settings
from .core.logging_config import setup_logging

# Initialize Production Logger
logger = setup_logging(settings.LOG_LEVEL, settings.APP_LOGGER_NAME)

app = FastAPI(title="Qwen3-ASR Production API")

@app.on_event("startup")
async def startup_event():
    logger.info("---- Qwen3-ASR API initialized (IDLE - No model loaded) ----")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": qwen_manager.current_model_id or "None",
        "device": qwen_manager.current_device or "None"
    }

@app.post("/load_model")
async def load_model(req: LoadModelRequest):
    logger.info(f"Loading Model: {req.model_id} on {req.device}")
    try:
        await qwen_manager.load_model(req.dict())
        return {"status": "success", "message": f"Model {req.model_id} is ready."}
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    prompt: str = Form(""),
    max_new_tokens: int = Form(512),
    max_inference_batch_size: int = Form(32),
    return_timestamps: bool = Form(True),
    max_gap_sec: float = Form(0.6),
    max_chars: int = Form(40),
    split_mode: str = Form("split_by_punctuation_or_pause_or_length")
):
    if not qwen_manager.model:
        raise HTTPException(status_code=503, detail="API is idle. Call /load_model first.")

    start_time = time.time()
    logger.info(f"Transcription Request: {file.filename} (Lang: {language})")

    try:
        content = await file.read()
        with io.BytesIO(content) as f:
            audio, _ = librosa.load(f, sr=16000)

        audio_duration = len(audio) / 16000

        result = qwen_manager.transcribe(
            audio=audio,
            language=language,
            hints=prompt,
            max_gap=max_gap_sec,
            max_chars=max_chars,
            split_mode=split_mode,
            return_ts=return_timestamps,
            max_new_tokens=max_new_tokens,
            batch_size=max_inference_batch_size
        )

        total_time = time.time() - start_time
        rtf = audio_duration / total_time if total_time > 0 else 0
        logger.info(f"Done in {total_time:.2f}s | Audio: {audio_duration:.2f}s | RTF: {rtf:.2f}x")

        return result
    except Exception as e:
        logger.error(f"Processing Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))