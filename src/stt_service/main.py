import io
import librosa
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from .services.model_manager import qwen_manager
from .models.schemas import TranscriptionResponse, LoadModelRequest
from .core.config import settings

# Setup logging
logger = logging.getLogger(settings.APP_LOGGER_NAME)

app = FastAPI(title="Qwen3-ASR Production API")

@app.get("/health")
async def health():
    """Check API status and currently loaded model."""
    return {
        "status": "ok",
        "model_loaded": qwen_manager.current_model_id or "None"
    }

@app.on_event("startup")
async def startup_event():
    """Load default model on startup."""
    try:
        logger.info(f"Starting up: Loading default model {settings.DEFAULT_MODEL}")
        await qwen_manager.load_model(settings.DEFAULT_MODEL, "bf16")
    except Exception as e:
        logger.error(f"Failed to load default model on startup: {e}")

@app.post("/load_model")
async def load_model(req: LoadModelRequest):
    """Switch models dynamically."""
    try:
        await qwen_manager.load_model(req.model_id, req.precision, req.use_aligner)
        return {"status": "success", "model": req.model_id}
    except Exception as e:
        logger.error(f"Manual model load failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model load failed: {str(e)}"
        )

@app.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    prompt: str = Form(""), # This is the 'hints' field
    max_gap_sec: float = Form(0.6),
    max_chars: int = Form(40),
    split_mode: str = Form("split_by_punctuation_or_pause_or_length")
):
    """OpenAI-compatible transcription endpoint with ComfyUI subtitle logic."""
    if not qwen_manager.model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Use /load_model first."
        )

    try:
        # Read file into memory
        audio_bytes = await file.read()

        # Qwen3 requires 16000Hz mono
        with io.BytesIO(audio_bytes) as f:
            audio, _ = librosa.load(f, sr=16000)

        # Call the manager (which uses the subtitle_utils logic)
        result = qwen_manager.transcribe(
            audio,
            language=language,
            hints=prompt,
            max_gap=max_gap_sec,
            max_chars=max_chars,
            split_mode=split_mode
        )
        return result

    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )