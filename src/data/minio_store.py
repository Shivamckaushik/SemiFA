"""MinIO object storage — inspection image upload / download."""

from __future__ import annotations

import io
import os
from datetime import timedelta
from pathlib import Path

from minio import Minio
from minio.error import S3Error

from src.config import settings


class ImageStore:
    """Thin wrapper around MinIO for inspection image management."""

    def __init__(self) -> None:
        self._client = Minio(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        self._bucket = settings.minio_bucket
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        if not self._client.bucket_exists(self._bucket):
            self._client.make_bucket(self._bucket)

    # ── Upload ───────────────────────────────────────────────────────────────

    def upload_image(
        self,
        object_name: str,
        image_bytes: bytes,
        content_type: str = "image/png",
    ) -> str:
        """Upload raw image bytes; returns the object name (key)."""
        data = io.BytesIO(image_bytes)
        self._client.put_object(
            bucket_name=self._bucket,
            object_name=object_name,
            data=data,
            length=len(image_bytes),
            content_type=content_type,
        )
        return object_name

    def upload_file(self, local_path: str | Path, object_name: str | None = None) -> str:
        local_path = Path(local_path)
        object_name = object_name or local_path.name
        self._client.fput_object(
            bucket_name=self._bucket,
            object_name=object_name,
            file_path=str(local_path),
        )
        return object_name

    # ── Download ─────────────────────────────────────────────────────────────

    def download_image(self, object_name: str) -> bytes:
        response = self._client.get_object(
            bucket_name=self._bucket, object_name=object_name
        )
        return response.read()

    def download_to_file(self, object_name: str, local_path: str | Path) -> None:
        self._client.fget_object(
            bucket_name=self._bucket,
            object_name=object_name,
            file_path=str(local_path),
        )

    # ── Presigned URL ─────────────────────────────────────────────────────────

    def presigned_url(self, object_name: str, expires_hours: int = 1) -> str:
        return self._client.presigned_get_object(
            bucket_name=self._bucket,
            object_name=object_name,
            expires=timedelta(hours=expires_hours),
        )

    def list_objects(self, prefix: str = "") -> list[str]:
        return [
            obj.object_name
            for obj in self._client.list_objects(
                self._bucket, prefix=prefix, recursive=True
            )
        ]
