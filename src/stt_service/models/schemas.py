from pydantic import BaseModel
from typing import List, Optional

class LoadModelRequest(BaseModel):
    model_id: str = "Qwen/Qwen3-ASR-0.6B"
    precision: str = "bf16" # Options: bf16, fp16, fp32
    use_aligner: bool = True

class SubtitleSegment(BaseModel):
    index: int
    start: float
    end: float
    text: str

class TranscriptionResponse(BaseModel):
    text: str              # Full plain text
    srt: str               # Formatted .srt string
    language: str          # Detected language
    duration: float        # Audio length in seconds
    segments: List[SubtitleSegment] # Timestamped chunks