from pydantic import BaseModel
from typing import List, Optional

class LoadModelRequest(BaseModel):
    model_id: str = "Qwen/Qwen3-ASR-0.6B"
    device: str = "cuda"
    dtype: str = "bf16"
    use_aligner: bool = True
    attn_implementation: str = "sdpa"
    max_inference_batch_size: int = 32
    max_new_tokens: int = 512

class SubtitleSegment(BaseModel):
    index: int
    start: float
    end: float
    text: str

class TranscriptionResponse(BaseModel):
    text: str
    srt: Optional[str]
    language: str
    duration: float
    segments: List[SubtitleSegment]