"""LLaVA-1.6 inference wrapper for semiconductor defect Q&A.

Supports:
  - Base LLaVA-1.6 (from HuggingFace)
  - QLoRA fine-tuned adapter merged on top
  - 4-bit / 8-bit quantisation for GPU memory efficiency
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import (
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    BitsAndBytesConfig,
)

from src.config import settings

logger = logging.getLogger(__name__)

# Prompt template matching LLaVA-1.6 Mistral chat format
_SYSTEM_PROMPT = (
    "You are an expert semiconductor failure analysis engineer. "
    "Analyse the provided inspection image (SEM, optical, or wafer map) "
    "and answer questions accurately and concisely."
)

_CHAT_TEMPLATE = (
    "[INST] <<SYS>>\n{system}\n<</SYS>>\n\n"
    "<image>\n{user_query} [/INST]"
)


class LLaVAInferenceEngine:
    """
    Thin inference wrapper around LLaVA-1.6.

    Args:
        model_id: HuggingFace model ID or local path
        finetuned_adapter_path: optional path to QLoRA adapter directory
        load_in_4bit: quantise to 4-bit (requires bitsandbytes + CUDA)
    """

    def __init__(
        self,
        model_id: str | None = None,
        finetuned_adapter_path: str | None = None,
        load_in_4bit: bool = True,
    ) -> None:
        self._model_id = model_id or settings.llava_model_id
        self._adapter_path = finetuned_adapter_path or settings.llava_finetuned_path
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_in_4bit = load_in_4bit and self._device == "cuda"

        self._processor: LlavaNextProcessor | None = None
        self._model: LlavaNextForConditionalGeneration | None = None

    # ── Lazy loading ─────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load model weights (call once at startup)."""
        logger.info("Loading LLaVA processor from %s", self._model_id)
        self._processor = LlavaNextProcessor.from_pretrained(
            self._model_id, token=settings.hf_token or None
        )

        quant_config = None
        if self._load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        logger.info("Loading LLaVA model weights (4bit=%s)…", self._load_in_4bit)
        self._model = LlavaNextForConditionalGeneration.from_pretrained(
            self._model_id,
            quantization_config=quant_config,
            torch_dtype=torch.float16 if not self._load_in_4bit else None,
            device_map="auto",
            token=settings.hf_token or None,
        )

        # Load fine-tuned adapter if available
        adapter_dir = Path(self._adapter_path)
        if adapter_dir.exists() and (adapter_dir / "adapter_config.json").exists():
            logger.info("Loading QLoRA adapter from %s", adapter_dir)
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(self._model, str(adapter_dir))
            self._model = self._model.merge_and_unload()
            logger.info("Adapter merged.")

        self._model.eval()
        logger.info("LLaVA inference engine ready on %s", self._device)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ── Inference ────────────────────────────────────────────────────────────

    def query(
        self,
        image: Image.Image,
        user_query: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ) -> str:
        """
        Run a single image + text query through LLaVA.

        Returns the model's text response.
        """
        if not self.is_loaded:
            self.load()

        if image.mode != "RGB":
            image = image.convert("RGB")

        prompt = _CHAT_TEMPLATE.format(system=_SYSTEM_PROMPT, user_query=user_query)

        inputs = self._processor(
            text=prompt, images=image, return_tensors="pt"
        ).to(self._device)

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self._processor.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[0][input_len:]
        return self._processor.decode(generated, skip_special_tokens=True).strip()

    def batch_query(
        self,
        pairs: list[tuple[Image.Image, str]],
        **kwargs: Any,
    ) -> list[str]:
        return [self.query(img, q, **kwargs) for img, q in pairs]
