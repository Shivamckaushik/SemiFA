"""QLoRA fine-tuning of LLaVA-1.6 on semiconductor defect image-text pairs.

Strategy:
  - Load LLaVA-1.6 in 4-bit (NF4) via bitsandbytes
  - Apply LoRA adapters via get_peft_model (PEFT)
  - Train with HuggingFace Trainer — response-only loss (human prompts masked)
  - Save LoRA adapter checkpoints to models/llava-semiconductor-qlora/

Hardware:
  - Minimum: 1× A100 40GB  (batch_size=2, grad_accum=8, ~2-3 hrs for 790 ex × 3 ep)
  - Alternative: T4 16GB    (batch_size=1, grad_accum=16, ~8-10 hrs)

Usage:
  python training/qlora_finetune.py
  python training/qlora_finetune.py --epochs 1 --batch-size 1   # T4 / quick test
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    BitsAndBytesConfig,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    Trainer,
    TrainingArguments,
)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class FinetuneConfig:
    base_model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf"
    output_dir: str = "models/llava-semiconductor-qlora"
    hf_token: str = field(default_factory=lambda: os.getenv("HF_TOKEN", ""))

    train_jsonl: str = field(default_factory=lambda: os.getenv("TRAIN_JSONL", "data/processed/train.jsonl"))
    val_jsonl: str = field(default_factory=lambda: os.getenv("VAL_JSONL", "data/processed/val.jsonl"))
    # image paths in JSONL are relative to project root — set "." here
    image_root: str = "."

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    # Training
    num_epochs: int = 3
    batch_size: int = 2          # A100: 2 | T4: 1
    grad_accumulation: int = 8   # effective batch = batch_size × grad_accumulation
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    max_seq_length: int = 2048
    bf16: bool = True            # A100 supports bf16; T4 → set to False, fp16=True
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 200


# ── Data loading ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert semiconductor failure analysis engineer. "
    "Analyse inspection images and answer questions accurately and concisely."
)


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_prompt_and_image(
    record: dict,
    processor: LlavaNextProcessor,
    image_root: str,
) -> tuple[str, Image.Image]:
    """Convert one JSONL record to (formatted_text, PIL image)."""
    img_path = Path(image_root) / record["image"]
    if img_path.exists():
        image = Image.open(img_path).convert("RGB")
    else:
        # Fallback: grey placeholder so training doesn't crash on missing files
        print(f"  [WARN] Missing image: {img_path} — using placeholder")
        image = Image.new("RGB", (256, 256), color=(100, 100, 100))

    conversations = record.get("conversations", [])

    # Build messages in the format processor.apply_chat_template expects.
    # First human turn always carries the image.
    messages = []
    for i, turn in enumerate(conversations):
        role = "user" if turn["from"] == "human" else "assistant"
        text = turn["value"]

        if role == "user":
            # Strip the raw "<image>\n" prefix — the processor adds the image token
            text = text.replace("<image>\n", "").replace("<image>", "").strip()
            if i == 0:
                # First turn: attach image content block
                content = [
                    {"type": "image"},
                    {"type": "text", "text": text},
                ]
            else:
                content = [{"type": "text", "text": text}]
        else:
            content = [{"type": "text", "text": text}]

        messages.append({"role": role, "content": content})

    # apply_chat_template → string with special tokens, image placeholders, etc.
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Prepend system prompt if the template doesn't include one already
    if SYSTEM_PROMPT not in prompt:
        # Mistral format: [INST] <<SYS>> ... <</SYS>> prepended to first turn
        prompt = prompt.replace(
            "[INST]",
            f"[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n",
            1,
        )

    return prompt, image


class SemiFA_Dataset(torch.utils.data.Dataset):
    """Dataset that returns (text_prompt, image, label) tuples for SFTTrainer."""

    def __init__(
        self,
        records: list[dict],
        processor: LlavaNextProcessor,
        image_root: str,
        max_length: int = 2048,
    ) -> None:
        self.records = records
        self.processor = processor
        self.image_root = image_root
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        prompt, image = build_prompt_and_image(record, self.processor, self.image_root)

        # Do NOT truncate here — LLaVA-1.6 expands one image into up to 2928 tokens
        # (5 tiles × ~576 tokens). Truncating mid-image causes token count mismatch.
        # Padding is handled dynamically by SemiFA_Collator.
        encoding = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        pixel_values = encoding["pixel_values"].squeeze(0)

        # Response-only loss: mask everything up to the last [/INST] token.
        labels = input_ids.clone()
        inst_token_ids = self.processor.tokenizer.encode("[/INST]", add_special_tokens=False)
        last_inst_pos = -1
        for i in range(len(input_ids) - len(inst_token_ids), -1, -1):
            if input_ids[i:i + len(inst_token_ids)].tolist() == inst_token_ids:
                last_inst_pos = i + len(inst_token_ids)
                break
        if last_inst_pos > 0:
            labels[:last_inst_pos] = -100

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
        }
        if "image_sizes" in encoding:
            out["image_sizes"] = encoding["image_sizes"].squeeze(0)
        return out


# ── Collator ──────────────────────────────────────────────────────────────────

class SemiFA_Collator:
    """Dynamic left-padding collator for variable-length LLaVA sequences."""

    PAD_ID = 0  # will be overridden in __init__ with actual pad token id

    def __init__(self, pad_token_id: int = 0) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        max_len = max(b["input_ids"].shape[0] for b in batch)

        input_ids_padded = []
        attention_mask_padded = []
        labels_padded = []

        for b in batch:
            seq_len = b["input_ids"].shape[0]
            pad_len = max_len - seq_len
            # Pad on the right
            input_ids_padded.append(
                torch.nn.functional.pad(b["input_ids"], (0, pad_len), value=self.pad_token_id)
            )
            attention_mask_padded.append(
                torch.nn.functional.pad(b["attention_mask"], (0, pad_len), value=0)
            )
            labels_padded.append(
                torch.nn.functional.pad(b["labels"], (0, pad_len), value=-100)
            )

        # pixel_values: variable number of tiles per image (e.g. [5,3,336,336] vs [3,3,336,336])
        # Pad to the max tile count in this batch; model uses image_sizes to ignore padding.
        max_tiles = max(b["pixel_values"].shape[0] for b in batch)
        pixel_values_padded = []
        for b in batch:
            pv = b["pixel_values"]
            pad_tiles = max_tiles - pv.shape[0]
            if pad_tiles > 0:
                pv = torch.cat(
                    [pv, torch.zeros(pad_tiles, *pv.shape[1:], dtype=pv.dtype)], dim=0
                )
            pixel_values_padded.append(pv)

        out = {
            "input_ids": torch.stack(input_ids_padded),
            "attention_mask": torch.stack(attention_mask_padded),
            "labels": torch.stack(labels_padded),
            "pixel_values": torch.stack(pixel_values_padded),
        }

        if "image_sizes" in batch[0]:
            out["image_sizes"] = torch.stack([b["image_sizes"] for b in batch])

        return out


# ── Fine-tuning ───────────────────────────────────────────────────────────────

def finetune(cfg: FinetuneConfig) -> None:
    print(f"[INFO] Base model : {cfg.base_model_id}")
    print(f"[INFO] Output dir : {cfg.output_dir}")
    print(f"[INFO] bf16={cfg.bf16} | batch={cfg.batch_size} | grad_accum={cfg.grad_accumulation}")

    # 4-bit NF4 quantisation
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("[INFO] Loading processor ...")
    processor = LlavaNextProcessor.from_pretrained(
        cfg.base_model_id,
        token=cfg.hf_token or None,
    )
    # Ensure padding side is right for decoder-only models
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    print("[INFO] Loading model in 4-bit ...")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        cfg.base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=cfg.hf_token or None,
    )
    model.config.use_cache = False  # required for gradient checkpointing

    # LoRA — apply before creating Trainer
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.lora_target_modules,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    # Datasets
    print("[INFO] Building datasets ...")
    train_records = load_jsonl(cfg.train_jsonl)
    val_records = load_jsonl(cfg.val_jsonl)
    print(f"  Train: {len(train_records)} | Val: {len(val_records)}")

    train_ds = SemiFA_Dataset(train_records, processor, cfg.image_root, cfg.max_seq_length)
    val_ds = SemiFA_Dataset(val_records, processor, cfg.image_root, cfg.max_seq_length)

    collator = SemiFA_Collator(pad_token_id=processor.tokenizer.pad_token_id)

    # Compute warmup_steps from warmup_ratio
    steps_per_epoch = len(train_ds) // (cfg.batch_size * cfg.grad_accumulation)
    total_steps = steps_per_epoch * cfg.num_epochs
    warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))
    print(f"[INFO] Total steps: {total_steps} | Warmup steps: {warmup_steps}")

    # Training config — base Trainer avoids SFTTrainer's HF-Dataset-only assumptions
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accumulation,
        gradient_checkpointing=True,
        learning_rate=cfg.learning_rate,
        warmup_steps=warmup_steps,
        bf16=cfg.bf16,
        fp16=not cfg.bf16,
        logging_steps=cfg.logging_steps,
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    print(f"[INFO] Trainable parameters:")
    model.print_trainable_parameters()

    print("[INFO] Starting QLoRA fine-tuning ...")
    trainer.train()

    print(f"[INFO] Saving LoRA adapter → {cfg.output_dir}")
    model.save_pretrained(cfg.output_dir)
    processor.save_pretrained(cfg.output_dir)
    print("[DONE] Fine-tuning complete.")
    print(f"       Adapter saved to: {Path(cfg.output_dir).resolve()}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA fine-tune LLaVA-1.6 on SemiFA dataset")
    parser.add_argument("--base-model", default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--output-dir", default="models/llava-semiconductor-qlora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2,
                        help="2 for A100 40GB, 1 for T4 16GB")
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--no-bf16", action="store_true",
                        help="Use fp16 instead of bf16 (for T4 GPU)")
    parser.add_argument("--image-root", default=".",
                        help="Root directory for resolving relative image paths")
    args = parser.parse_args()

    cfg = FinetuneConfig(
        base_model_id=args.base_model,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        grad_accumulation=args.grad_accum,
        lora_r=args.lora_r,
        bf16=not args.no_bf16,
        image_root=args.image_root,
    )
    finetune(cfg)
