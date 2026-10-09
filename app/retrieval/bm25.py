"""Lexical retrieval (BM25). One index per set of documents the caller is authorized to read."""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict

import numpy as np
from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import Chunk, Document
from app.text import content_tokens


def index_text(title: str | None, section: str | None, text: str) -> str:
    """Text used for indexing/reranking: the passage prefixed with its document title and section."""
    header = " - ".join(part for part in (title, section) if part)
    return f"{header}\n{text}" if header else text


class _Index:
    def __init__(self, chunk_ids: list[str], corpus: list[list[str]]):
        self.chunk_ids = chunk_ids
        self.bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        tokens = content_tokens(query)
        if self.bm25 is None or not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top = np.argsort(scores)[::-1][:limit]
        return [(self.chunk_ids[i], float(scores[i])) for i in top if scores[i] > 0]


class BM25Retriever:
    """Builds indexes lazily from the metadata store and caches them.

    The index only ever contains chunks of documents the requesting user may read, so
    unauthorized text cannot influence scores (no leakage through term statistics either).
    The cache key covers document ids, versions and update times, so re-ingestion or an ACL
    change made by any process invalidates it.
    """

    def __init__(self, max_indexes: int = 16):
        self._cache: OrderedDict[str, _Index] = OrderedDict()
        self._max = max_indexes
        self._lock = threading.Lock()

    @staticmethod
    def _signature(collection_id: str, documents: dict[str, Document]) -> str:
        parts = [f"{d.id}:{d.version}:{d.updated_at.isoformat()}" for d in documents.values()]
        return hashlib.sha1("|".join([collection_id, *sorted(parts)]).encode()).hexdigest()

    def search(
        self, db: Session, query: str, collection_id: str, documents: dict[str, Document], limit: int
    ) -> list[tuple[str, float]]:
        """`documents` must already be filtered to what the user is authorized to read."""
        if not documents:
            return []
        key = self._signature(collection_id, documents)
        with self._lock:
            index = self._cache.get(key)
            if index is not None:
                self._cache.move_to_end(key)
        if index is None:
            index = self._build(db, documents)
            with self._lock:
                self._cache[key] = index
                while len(self._cache) > self._max:
                    self._cache.popitem(last=False)
        return index.search(query, limit)

    @staticmethod
    def _build(db: Session, documents: dict[str, Document]) -> _Index:
        rows = db.execute(
            select(Chunk.id, Chunk.document_id, Chunk.section, Chunk.text)
            .where(Chunk.document_id.in_(list(documents)))
            .order_by(Chunk.id)
        ).all()
        chunk_ids = [row.id for row in rows]
        corpus = [
            content_tokens(index_text(documents[row.document_id].meta.get("title"), row.section, row.text))
            for row in rows
        ]
        return _Index(chunk_ids, corpus)
