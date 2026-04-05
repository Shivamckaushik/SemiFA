"""Defect detection — DINOv2 feature extractor + lightweight detection head.

Architecture:
  DINOv2-base (frozen) → CLS token + patch features
  ↓
  MLP detection head → defect class logits
  ↓
  Anomaly score (cosine distance from in-distribution centroid)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from src.config import settings

logger = logging.getLogger(__name__)

# Defect classes derived from domain knowledge (SEMI standards + WM-811K)
DEFECT_CLASSES = [
    "scratch",
    "particle_contamination",
    "edge_crack",
    "center_cluster",
    "local_cluster",
    "ring_pattern",
    "random_defects",
    "near_full_wafer",
    "no_defect",
]


class DINOv2FeatureExtractor(nn.Module):
    """Frozen DINOv2 encoder that returns CLS + mean-pooled patch embeddings."""

    def __init__(self, model_id: str = "facebook/dinov2-base") -> None:
        super().__init__()
        self._processor = AutoImageProcessor.from_pretrained(model_id)
        self._backbone = AutoModel.from_pretrained(model_id)
        # Freeze all parameters
        for param in self._backbone.parameters():
            param.requires_grad = False
        self._backbone.eval()
        logger.info("DINOv2 encoder loaded: %s", model_id)

    @property
    def embedding_dim(self) -> int:
        return self._backbone.config.hidden_size  # 768 for dinov2-base

    def forward(self, images: list[Image.Image]) -> torch.Tensor:
        """
        Args:
            images: list of PIL Images (RGB)
        Returns:
            Tensor of shape (N, 768) — CLS tokens
        """
        inputs = self._processor(images=images, return_tensors="pt")
        inputs = {k: v.to(next(self._backbone.parameters()).device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._backbone(**inputs)
        # CLS token is the first token
        cls_tokens = outputs.last_hidden_state[:, 0, :]  # (N, 768)
        return cls_tokens


class DefectClassificationHead(nn.Module):
    """Lightweight MLP head trained on top of frozen DINOv2 features."""

    def __init__(self, input_dim: int = 768, num_classes: int = len(DEFECT_CLASSES)) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class DefectDetector:
    """
    End-to-end defect detector combining DINOv2 + classification head.

    Usage:
        detector = DefectDetector()
        result = detector.detect(pil_image)
        # result = {"defect_class": "scratch", "confidence": 0.94,
        #           "anomaly_score": 0.32, "embedding": [...]}
    """

    def __init__(self, weights_path: str | Path | None = None) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._encoder = DINOv2FeatureExtractor(settings.dinov2_model_id).to(self._device)
        self._head = DefectClassificationHead().to(self._device)
        self._head.eval()

        # In-distribution centroids for anomaly scoring (loaded from training)
        self._centroids: torch.Tensor | None = None

        if weights_path and Path(weights_path).exists():
            self._load_weights(weights_path)
            logger.info("Defect detector weights loaded from %s", weights_path)
        else:
            logger.warning(
                "No weights found at %s — using random head (fine-tune before production).",
                weights_path,
            )

    def _load_weights(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self._device)
        self._head.load_state_dict(state["head"])
        if "centroids" in state:
            self._centroids = state["centroids"].to(self._device)

    # ── Inference ────────────────────────────────────────────────────────────

    def detect(self, image: Image.Image) -> dict[str, Any]:
        """Run defect detection on a single image."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        embedding = self._encoder([image])  # (1, 768)

        with torch.no_grad():
            logits = self._head(embedding)  # (1, num_classes)
            probs = torch.softmax(logits, dim=-1).squeeze(0)

        top_idx = int(probs.argmax())
        confidence = float(probs[top_idx])
        anomaly_score = self._compute_anomaly_score(embedding)

        return {
            "defect_class": DEFECT_CLASSES[top_idx],
            "confidence": round(confidence, 4),
            "anomaly_score": round(anomaly_score, 4),
            "class_probabilities": {
                cls: round(float(p), 4)
                for cls, p in zip(DEFECT_CLASSES, probs.tolist())
            },
            "embedding": embedding.squeeze(0).cpu().numpy().tolist(),
        }

    def batch_detect(self, images: list[Image.Image]) -> list[dict[str, Any]]:
        return [self.detect(img) for img in images]

    def _compute_anomaly_score(self, embedding: torch.Tensor) -> float:
        """Cosine distance from nearest in-distribution centroid."""
        if self._centroids is None:
            return 0.0
        emb_norm = nn.functional.normalize(embedding, dim=-1)
        ctr_norm = nn.functional.normalize(self._centroids, dim=-1)
        sims = (emb_norm @ ctr_norm.T).squeeze(0)
        return float(1.0 - sims.max().item())
