"""Process-wide singletons: embedder, vector store, reranker, LLM, cache, BM25 indexes."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache

from app.cache import Cache, build_cache
from app.config import Settings, get_settings
from app.generation.llm import LLM, build_llm
from app.ingestion.embeddings import Embedder, build_embedder
from app.reranking.reranker import Reranker, build_reranker
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.vector_store import QdrantVectorStore


@dataclass
class Services:
    settings: Settings
    embedder: Embedder
    vector_store: QdrantVectorStore
    reranker: Reranker
    llm: LLM
    cache: Cache
    bm25: BM25Retriever

    def embed_query(self, query: str) -> list[float]:
        if not self.settings.cache_enabled:
            return self.embedder.embed_query(query)
        key = "emb:" + hashlib.sha256(f"{self.embedder.name}|{query}".encode()).hexdigest()
        vector = self.cache.get(key)
        if vector is None:
            vector = self.embedder.embed_query(query)
            self.cache.set(key, vector, ttl=3600)
        return vector


@lru_cache
def get_services() -> Services:
    settings = get_settings()
    embedder = build_embedder(settings.embedding_provider, settings.embedding_model, settings.model_cache_dir)
    # One Qdrant collection per embedding model, so switching models can never mix vector spaces.
    suffix = re.sub(r"[^a-z0-9]+", "_", embedder.name.lower()).strip("_")
    vector_store = QdrantVectorStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        path=settings.qdrant_path,
        collection=f"{settings.qdrant_collection}_{suffix}",
        dim=embedder.dim,
    )
    return Services(
        settings=settings,
        embedder=embedder,
        vector_store=vector_store,
        reranker=build_reranker(settings.reranker_provider, settings.reranker_model, settings.model_cache_dir),
        llm=build_llm(settings),
        cache=build_cache(settings.redis_url),
        bm25=BM25Retriever(),
    )
