"""Image pre-processing pipeline for semiconductor inspection images.

Handles three modalities:
  - SEM (Scanning Electron Microscope) — grayscale, high resolution
  - Optical — colour or grayscale
  - Wafer map — binary / pseudo-colour defect maps
"""

from __future__ import annotations

from enum import Enum
from io import BytesIO

import cv2
import numpy as np
from PIL import Image


class ImageModality(str, Enum):
    SEM = "sem"
    OPTICAL = "optical"
    WAFER_MAP = "wafer_map"


# Target sizes for each modality
_TARGET_SIZE: dict[ImageModality, tuple[int, int]] = {
    ImageModality.SEM: (512, 512),
    ImageModality.OPTICAL: (512, 512),
    ImageModality.WAFER_MAP: (256, 256),
}

# Normalization stats (ImageNet for optical; grayscale mean/std for SEM)
_NORM_MEAN: dict[ImageModality, list[float]] = {
    ImageModality.SEM: [0.5],
    ImageModality.OPTICAL: [0.485, 0.456, 0.406],
    ImageModality.WAFER_MAP: [0.5],
}
_NORM_STD: dict[ImageModality, list[float]] = {
    ImageModality.SEM: [0.5],
    ImageModality.OPTICAL: [0.229, 0.224, 0.225],
    ImageModality.WAFER_MAP: [0.5],
}


class InspectionImagePreprocessor:
    """Deterministic preprocessing — no random augmentation at inference."""

    def __init__(self, modality: ImageModality = ImageModality.OPTICAL) -> None:
        self.modality = modality
        self._size = _TARGET_SIZE[modality]

    # ── Public API ───────────────────────────────────────────────────────────

    def preprocess(self, source: bytes | str | np.ndarray | Image.Image) -> np.ndarray:
        """
        Preprocess an inspection image.

        Returns a float32 numpy array of shape (H, W, C) normalised to [0, 1]
        after channel-wise normalisation.
        """
        img = self._load(source)
        img = self._resize(img)
        img = self._ensure_channels(img)
        img = self._normalise(img)
        return img

    def preprocess_for_llava(
        self, source: bytes | str | np.ndarray | Image.Image
    ) -> Image.Image:
        """Return a PIL Image ready for LLaVA's image processor (RGB, no norm)."""
        img = self._load(source)
        img = self._resize(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    # ── Internals ────────────────────────────────────────────────────────────

    @staticmethod
    def _load(source: bytes | str | np.ndarray | Image.Image) -> Image.Image:
        if isinstance(source, Image.Image):
            return source
        if isinstance(source, np.ndarray):
            return Image.fromarray(source)
        if isinstance(source, bytes):
            return Image.open(BytesIO(source))
        return Image.open(source)

    def _resize(self, img: Image.Image) -> Image.Image:
        return img.resize(self._size, Image.LANCZOS)

    def _ensure_channels(self, img: Image.Image) -> np.ndarray:
        arr = np.array(img, dtype=np.float32) / 255.0
        if self.modality == ImageModality.SEM:
            if arr.ndim == 3 and arr.shape[2] == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            arr = arr[..., np.newaxis]  # (H, W, 1)
        elif self.modality == ImageModality.WAFER_MAP:
            if arr.ndim == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            arr = arr[..., np.newaxis]
        else:
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
        return arr

    def _normalise(self, arr: np.ndarray) -> np.ndarray:
        mean = np.array(_NORM_MEAN[self.modality], dtype=np.float32)
        std = np.array(_NORM_STD[self.modality], dtype=np.float32)
        return (arr - mean) / std


class AugmentationPipeline:
    """Training-time augmentation for SEM / optical images using albumentations."""

    def __init__(self, modality: ImageModality = ImageModality.OPTICAL) -> None:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        common = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=15, p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.2),
        ]

        if modality == ImageModality.OPTICAL:
            colour = [
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, p=0.4),
                A.Normalize(
                    mean=_NORM_MEAN[modality], std=_NORM_STD[modality]
                ),
            ]
        else:
            colour = [
                A.RandomGamma(p=0.3),
                A.Normalize(
                    mean=_NORM_MEAN[modality], std=_NORM_STD[modality]
                ),
            ]

        self._transform = A.Compose(common + colour + [ToTensorV2()])

    def __call__(self, image: np.ndarray) -> "torch.Tensor":  # noqa: F821
        return self._transform(image=image)["image"]
