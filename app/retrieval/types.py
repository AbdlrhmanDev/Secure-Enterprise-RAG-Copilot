"""Retrieval configuration and result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from app.config import Settings


class RetrievalConfig(BaseModel):
    mode: Literal["bm25", "dense", "hybrid"] = "hybrid"
    top_k: int = Field(20, ge=1, le=200, description="Candidates fetched from each first-stage retriever")
    fusion: Literal["rrf", "weighted"] = "rrf"
    rrf_k: int = Field(60, ge=1)
    dense_weight: float = Field(0.5, ge=0)
    bm25_weight: float = Field(0.5, ge=0)
    rerank: bool = True
    rerank_candidates: int = Field(20, ge=1, le=200, description="Fused candidates passed to the reranker")
    final_context_k: int = Field(5, ge=1, le=50, description="Chunks placed in the LLM context")

    @classmethod
    def from_settings(cls, settings: Settings) -> RetrievalConfig:
        return cls(
            mode=settings.retrieval_mode,
            top_k=settings.top_k,
            fusion=settings.fusion,
            rrf_k=settings.rrf_k,
            dense_weight=settings.dense_weight,
            bm25_weight=settings.bm25_weight,
            rerank=settings.rerank,
            rerank_candidates=settings.rerank_candidates,
            final_context_k=settings.final_context_k,
        )


class RetrievalOverrides(BaseModel):
    """Per-request overrides; unset fields fall back to the server defaults."""

    mode: Literal["bm25", "dense", "hybrid"] | None = None
    top_k: int | None = Field(None, ge=1, le=200)
    fusion: Literal["rrf", "weighted"] | None = None
    rrf_k: int | None = Field(None, ge=1)
    dense_weight: float | None = Field(None, ge=0)
    bm25_weight: float | None = Field(None, ge=0)
    rerank: bool | None = None
    rerank_candidates: int | None = Field(None, ge=1, le=200)
    final_context_k: int | None = Field(None, ge=1, le=50)

    def apply(self, base: RetrievalConfig) -> RetrievalConfig:
        return RetrievalConfig(**{**base.model_dump(), **self.model_dump(exclude_none=True)})


@dataclass
class ScoredChunk:
    chunk_id: str
    document_id: str
    filename: str
    title: str | None
    text: str
    page: int | None
    section: str | None
    score: float  # final ranking score: reranker score if reranked, otherwise the first-stage score
    retrieval_score: float = 0.0
    rerank_score: float | None = None


@dataclass
class RetrievalResult:
    ranked: list[ScoredChunk] = field(default_factory=list)  # full ranked list, used for metrics
    context: list[ScoredChunk] = field(default_factory=list)  # the top `final_context_k` given to the LLM
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    acl_dropped: int = 0
