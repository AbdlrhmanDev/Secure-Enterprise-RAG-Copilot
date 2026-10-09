"""Qdrant-backed dense index. ACL and collection filters are applied inside the vector search itself."""

from __future__ import annotations

import threading
import uuid
from contextlib import nullcontext
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client import models as qm


@dataclass
class VectorRecord:
    chunk_id: str
    vector: list[float]
    document_id: str
    collection_id: str
    allowed_roles: list[str]


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantVectorStore:
    def __init__(self, *, url: str | None, api_key: str | None, path: str, collection: str, dim: int):
        if url:
            self._client = QdrantClient(url=url, api_key=api_key)
            self._lock = nullcontext()
        else:
            # Embedded mode: single process only, and not thread-safe.
            try:
                self._client = QdrantClient(location=":memory:") if path == ":memory:" else QdrantClient(path=path)
            except RuntimeError as exc:
                if "already accessed" not in str(exc):
                    raise
                raise SystemExit(
                    f"The local Qdrant index at {path} is in use by another process, most likely the API server "
                    "(uvicorn). Embedded Qdrant allows one process at a time: stop the server and run this again, "
                    "or set QDRANT_URL to a Qdrant server to allow concurrent access."
                ) from None
            self._lock = threading.RLock()
        self._collection = collection
        self._dim = dim
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        with self._lock:
            if self._client.collection_exists(self._collection):
                return
            self._client.create_collection(
                self._collection,
                vectors_config=qm.VectorParams(size=self._dim, distance=qm.Distance.COSINE),
            )
            for field in ("collection_id", "document_id", "allowed_roles"):
                self._client.create_payload_index(
                    self._collection, field_name=field, field_schema=qm.PayloadSchemaType.KEYWORD
                )

    def upsert(self, records: list[VectorRecord], batch_size: int = 256) -> None:
        for start in range(0, len(records), batch_size):
            points = [
                qm.PointStruct(
                    id=_point_id(r.chunk_id),
                    vector=r.vector,
                    payload={
                        "chunk_id": r.chunk_id,
                        "document_id": r.document_id,
                        "collection_id": r.collection_id,
                        "allowed_roles": r.allowed_roles,
                    },
                )
                for r in records[start : start + batch_size]
            ]
            with self._lock:
                self._client.upsert(self._collection, points=points, wait=True)

    @staticmethod
    def _document_filter(document_id: str) -> qm.Filter:
        return qm.Filter(must=[qm.FieldCondition(key="document_id", match=qm.MatchValue(value=document_id))])

    def delete_document(self, document_id: str) -> None:
        with self._lock:
            self._client.delete(
                self._collection,
                points_selector=qm.FilterSelector(filter=self._document_filter(document_id)),
                wait=True,
            )

    def set_document_roles(self, document_id: str, allowed_roles: list[str]) -> None:
        with self._lock:
            self._client.set_payload(
                self._collection,
                payload={"allowed_roles": allowed_roles},
                points=self._document_filter(document_id),
                wait=True,
            )

    def search(
        self, vector: list[float], *, collection_id: str, roles: list[str] | None, limit: int
    ) -> list[tuple[str, float]]:
        """Nearest chunks in `collection_id`. `roles=None` means no ACL restriction (admin)."""
        must = [qm.FieldCondition(key="collection_id", match=qm.MatchValue(value=collection_id))]
        if roles is not None:
            must.append(qm.FieldCondition(key="allowed_roles", match=qm.MatchAny(any=list(roles))))
        with self._lock:
            result = self._client.query_points(
                self._collection,
                query=vector,
                query_filter=qm.Filter(must=must),
                limit=limit,
                with_payload=["chunk_id"],
            )
        return [(point.payload["chunk_id"], float(point.score)) for point in result.points]
