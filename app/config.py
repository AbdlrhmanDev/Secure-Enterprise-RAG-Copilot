"""Application settings. Everything is driven by environment variables (or a local .env file)."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

DEV_JWT_SECRET = "dev-only-secret-change-me-before-deploying-anywhere"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    # Document text never reaches the logs unless this is switched on explicitly.
    log_document_text: bool = False

    # Storage
    database_url: str = "sqlite:///./data/rag.db"
    qdrant_url: str | None = None  # e.g. http://qdrant:6333; unset = embedded local mode
    qdrant_api_key: str | None = None
    qdrant_path: str = "./data/qdrant"  # ":memory:" for tests
    qdrant_collection: str = "chunks"
    redis_url: str | None = None
    storage_dir: str = "./data/uploads"
    max_upload_mb: int = 50

    # Auth (simulated users, real signed JWTs)
    jwt_secret: str = DEV_JWT_SECRET
    jwt_algorithm: str = "HS256"
    jwt_ttl_minutes: int = 480

    # Ingestion
    chunk_size: int = 220  # approximate tokens per chunk
    chunk_overlap: int = 40
    ingest_backend: Literal["inline", "background", "arq"] = "background"
    ingest_max_retries: int = 3
    default_collection: str = "policies"

    # Models
    embedding_provider: Literal["fastembed", "hash"] = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_provider: Literal["fastembed", "lexical"] = "fastembed"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    model_cache_dir: str = "./data/models"

    # Retrieval defaults (all overridable per request / per eval experiment)
    retrieval_mode: Literal["bm25", "dense", "hybrid"] = "hybrid"
    top_k: int = 20
    fusion: Literal["rrf", "weighted"] = "rrf"
    rrf_k: int = 60
    dense_weight: float = 0.5
    bm25_weight: float = 0.5
    rerank: bool = True
    rerank_candidates: int = 20
    final_context_k: int = 5
    # If reranking is on and the best reranked chunk scores below this (0..1), the
    # pipeline refuses without calling the LLM.
    min_relevance_score: float = 0.0

    # Generation
    llm_provider: Literal["auto", "openai", "extractive"] = "auto"
    openai_api_key: str | None = None  # from the environment or .env
    openai_model: str = "gpt-5-mini"
    openai_reasoning_effort: str = ""  # e.g. "low"; empty = model default (only reasoning models accept it)
    llm_max_tokens: int = 4000
    # Used for the cost estimate only. Set these to the current list price of OPENAI_MODEL.
    llm_input_price_per_mtok: float = 0.25
    llm_output_price_per_mtok: float = 2.0
    judge_model: str | None = None  # defaults to openai_model

    # Caching
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300

    def resolved_llm_provider(self) -> str:
        if self.llm_provider != "auto":
            return self.llm_provider
        return "openai" if (self.openai_api_key or os.getenv("OPENAI_API_KEY")) else "extractive"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    if settings.env == "prod" and settings.jwt_secret == DEV_JWT_SECRET:
        raise RuntimeError("JWT_SECRET must be set to a non-default value when ENV=prod")
    return settings
