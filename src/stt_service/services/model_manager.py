import torch
import gc
import logging
import numpy as np
from typing import Optional, Dict, Any
from qwen_asr import Qwen3ASRModel
from ..core.logging_config import get_logger
from .subtitle_utils import format_srt_time, group_time_stamps

logger = get_logger("model_manager")

class QwenASRManager:
    def __init__(self):
        self.model: Optional[Qwen3ASRModel] = None
        self.current_model_id: Optional[str] = None
        self.current_device: Optional[str] = None

    async def load_model(self, params: Dict[str, Any]):
        # Unload existing to clear VRAM
        await self.unload_model()

        model_id = params.get("model_id")
        device = params.get("device", "cuda:0")
        dtype_str = params.get("dtype", "bf16")

        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        torch_dtype = dtype_map.get(dtype_str, torch.float32)

        # Forced Aligner settings derived from request
        use_aligner = params.get("use_aligner", True)
        aligner_id = "Qwen/Qwen3-ForcedAligner-0.6B" if use_aligner else None

        logger.info(f"Initializing Qwen3ASR: {model_id} on {device}")

        self.model = Qwen3ASRModel.from_pretrained(
            pretrained_model_name_or_path=model_id,
            device_map=device,
            torch_dtype=torch_dtype,
            attn_implementation=params.get("attn_implementation", "sdpa"),
            forced_aligner=aligner_id,
            forced_aligner_kwargs={
                "dtype": torch_dtype,
                "device_map": device,
                "attn_implementation": params.get("attn_implementation", "sdpa")
            } if aligner_id else None,
            max_inference_batch_size=params.get("max_inference_batch_size", 32),
            max_new_tokens=params.get("max_new_tokens", 512)
        )
        self.current_model_id = model_id
        self.current_device = device

    async def unload_model(self):
        if self.model:
            logger.warning(f"Unloading model {self.current_model_id} to free memory")
            self.model = None
            self.current_model_id = None
            self.current_device = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def transcribe(self, audio, language, hints, max_gap, max_chars, split_mode,
                  return_ts, max_new_tokens, batch_size):

        # DYNAMICALLY UPDATE ATTRIBUTES (From Qwen3ASRModel __init__)
        self.model.max_new_tokens = max_new_tokens
        self.model.max_inference_batch_size = batch_size

        # Determine if alignment is possible
        can_align = self.model.forced_aligner is not None
        do_align = return_ts and can_align

        if return_ts and not can_align:
            logger.warning("Subtitle timestamps requested but Aligner not loaded.")

        results = self.model.transcribe(
            audio=(audio, 16000),
            language=None if language == "auto" else language,
            context=hints if hints else None,
            return_time_stamps=do_align
        )

        res = results[0]
        output = {
            "text": res.text,
            "language": res.language,
            "duration": round(len(audio) / 16000, 2),
            "srt": None,
            "segments": []
        }

        if do_align and res.time_stamps:
            logger.debug("Applying linguistic subtitle grouping logic...")
            groups = group_time_stamps(res.time_stamps, max_gap, max_chars, split_mode)
            srt_lines = []
            for i, g in enumerate(groups, 1):
                srt_lines.append(f"{i}\n{format_srt_time(g['start'])} --> {format_srt_time(g['end'])}\n{g['text']}\n")
                g["index"] = i

            output["srt"] = "\n".join(srt_lines)
            output["segments"] = groups

        return output

qwen_manager = QwenASRManager()