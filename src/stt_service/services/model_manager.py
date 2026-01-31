import torch
import gc
import logging
import numpy as np
from typing import Optional
from qwen_asr import Qwen3ASRModel
from ..core.config import settings
from .subtitle_utils import format_srt_time, group_time_stamps

logger = logging.getLogger(settings.APP_LOGGER_NAME)


class QwenASRManager:
    def __init__(self):
        self.model: Optional[Qwen3ASRModel] = None
        self.current_model_id: Optional[str] = None

    async def load_model(self, model_id: str, precision: str, use_aligner: bool = True):
        if self.current_model_id == model_id and self.model: return
        await self.unload_model()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16

        aligner_id = "Qwen/Qwen3-ForcedAligner-0.6B" if use_aligner else None

        logger.info(f"Loading Qwen3-ASR: {model_id} (Using SDPA Attention)")

        # We set attn_implementation="sdpa" to skip flash-attn requirement
        self.model = Qwen3ASRModel.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=device,
            forced_aligner=aligner_id,
            forced_aligner_kwargs={
                "dtype": dtype,
                "device_map": device,
                "attn_implementation": "sdpa"
            },
            max_inference_batch_size=32,
            attn_implementation="sdpa"
        )
        self.current_model_id = model_id
        logger.info("Model loaded successfully.")

    async def unload_model(self):
        if self.model:
            self.model = None
            self.current_model_id = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def transcribe(self, audio, language, hints, max_gap, max_chars, split_mode):
        if not self.model: raise RuntimeError("Model not loaded.")

        results = self.model.transcribe(
            audio=(audio, 16000),
            language=None if language == "auto" else language,
            context=hints if hints else None,
            return_time_stamps=True if self.model.forced_aligner else False
        )

        res = results[0]
        groups = group_time_stamps(res.time_stamps, max_gap, max_chars, split_mode)

        srt_lines = []
        sub_list = []
        for i, g in enumerate(groups, 1):
            start_fmt = format_srt_time(g['start'])
            end_fmt = format_srt_time(g['end'])
            srt_lines.append(f"{i}\n{start_fmt} --> {end_fmt}\n{g['text']}\n")
            sub_list.append({"index": i, "start": g['start'], "end": g['end'], "text": g['text']})

        return {
            "text": res.text,
            "srt": "\n".join(srt_lines),
            "language": res.language,
            "duration": len(audio) / 16000,
            "segments": sub_list
        }


qwen_manager = QwenASRManager()