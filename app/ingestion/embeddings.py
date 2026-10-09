"""Embedding providers: FastEmbed (ONNX, CPU-friendly) and a deterministic hashing embedder for tests."""

from __future__ import annotations

import hashlib
import math
import threading
from typing import Protocol

from app.text import content_tokens


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    def __init__(self, model_name: str, cache_dir: str):
        self.name = model_name
        self._cache_dir = cache_dir
        self._model = None
        self._lock = threading.Lock()
        self._dim: int | None = None

    def _load(self):
        with self._lock:
            if self._model is None:
                from fastembed import TextEmbedding

                self._model = TextEmbedding(model_name=self.name, cache_dir=self._cache_dir)
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed_query("dimension probe"))
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._load().embed(texts, batch_size=32)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._load().query_embed(text))).tolist()


class HashEmbedder:
    """Feature-hashing bag of words. No model download; useful for tests and offline smoke runs."""

    def __init__(self, dim: int = 384):
        self.name = f"hash-{dim}"
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = content_tokens(text)
        for feature in tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]:
            digest = hashlib.md5(feature.encode()).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            vector[index] += 1.0 if digest[4] & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def build_embedder(provider: str, model_name: str, cache_dir: str) -> Embedder:
    if provider == "hash":
        return HashEmbedder()
    return FastEmbedEmbedder(model_name, cache_dir)
