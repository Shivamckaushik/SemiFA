"""Qdrant vector store — defect embedding storage and similarity retrieval."""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from src.config import settings


VECTOR_DIM = 768  # DINOv2-base embedding size


class DefectVectorStore:
    """Manages semiconductor defect embeddings in Qdrant."""

    def __init__(self) -> None:
        self._client = QdrantClient(
            host=settings.qdrant_host, port=settings.qdrant_port
        )
        self._collection = settings.qdrant_collection
        self._ensure_collection()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )

    # ── Write ────────────────────────────────────────────────────────────────

    def upsert_defect(
        self,
        embedding: list[float],
        metadata: dict[str, Any],
        point_id: str | None = None,
    ) -> str:
        """Store a defect embedding with its metadata payload."""
        point_id = point_id or str(uuid.uuid4())
        self._client.upsert(
            collection_name=self._collection,
            points=[PointStruct(id=point_id, vector=embedding, payload=metadata)],
        )
        return point_id

    def batch_upsert(
        self, records: list[dict[str, Any]], batch_size: int = 128
    ) -> None:
        """Batch upsert — each record must have 'embedding' and 'metadata' keys."""
        points = [
            PointStruct(
                id=r.get("id", str(uuid.uuid4())),
                vector=r["embedding"],
                payload=r["metadata"],
            )
            for r in records
        ]
        for i in range(0, len(points), batch_size):
            self._client.upsert(
                collection_name=self._collection, points=points[i : i + batch_size]
            )

    # ── Read ─────────────────────────────────────────────────────────────────

    def search_similar(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        defect_type_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return top-k historically similar defects."""
        query_filter = None
        if defect_type_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="defect_type",
                        match=MatchValue(value=defect_type_filter),
                    )
                ]
            )

        try:
            # qdrant-client >= 1.7.0
            hits = self._client.search(
                collection_name=self._collection,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )
            return [{"score": h.score, "id": h.id, **h.payload} for h in hits]
        except AttributeError:
            # qdrant-client >= 1.10 renamed to query_points
            results = self._client.query_points(
                collection_name=self._collection,
                query=query_embedding,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )
            return [{"score": h.score, "id": h.id, **h.payload} for h in results.points]

    def get_defect_by_id(self, point_id: str) -> dict[str, Any] | None:
        results = self._client.retrieve(
            collection_name=self._collection,
            ids=[point_id],
            with_payload=True,
            with_vectors=False,
        )
        return results[0].payload if results else None
