"""Rerankers. Scores are normalized to [0, 1] so thresholds and citation scores are comparable."""

from __future__ import annotations

import math
import threading
from typing import Protocol

from app.text import content_tokens


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: list[str]) -> list[float]: ...


class FastEmbedReranker:
    """Cross-encoder (ONNX). Reads query and passage jointly, so it is far more precise than
    the bi-encoder used for first-stage retrieval, at the cost of one forward pass per candidate."""

    def __init__(self, model_name: str, cache_dir: str):
        self.name = model_name
        self._cache_dir = cache_dir
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                self._model = TextCrossEncoder(model_name=self.name, cache_dir=self._cache_dir)
        return self._model

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        logits = self._load().rerank(query, passages)
        return [1.0 / (1.0 + math.exp(-float(x))) for x in logits]


class LexicalReranker:
    """Query-term coverage. Deterministic stand-in for tests and offline runs."""

    name = "lexical-overlap"

    def score(self, query: str, passages: list[str]) -> list[float]:
        query_terms = set(content_tokens(query))
        if not query_terms:
            return [0.0] * len(passages)
        return [len(query_terms & set(content_tokens(p))) / len(query_terms) for p in passages]


def build_reranker(provider: str, model_name: str, cache_dir: str) -> Reranker:
    if provider == "lexical":
        return LexicalReranker()
    return FastEmbedReranker(model_name, cache_dir)
