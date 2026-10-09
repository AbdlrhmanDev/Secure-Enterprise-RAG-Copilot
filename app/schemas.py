"""Request/response models for the HTTP API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.retrieval.types import RetrievalConfig, RetrievalOverrides


class TokenRequest(BaseModel):
    user_id: str = Field(examples=["u_employee"])


class UserOut(BaseModel):
    id: str
    name: str
    roles: list[str]


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserOut


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    collection_id: str
    filename: str
    source_uri: str
    checksum: str
    version: int
    status: str
    allowed_roles: list[str]
    error: str | None
    attempts: int
    page_count: int
    chunk_count: int
    uploaded_by: str | None
    created_at: datetime
    updated_at: datetime


class UploadResponse(BaseModel):
    document: DocumentOut
    deduplicated: bool = Field(description="True when identical content was already ingested (no work queued)")


class DocumentUpdate(BaseModel):
    allowed_roles: list[str] = Field(min_length=1)


class ChunkOut(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    collection_id: str
    page: int | None
    section: str | None
    token_count: int
    text: str


class CollectionOut(BaseModel):
    id: str
    document_count: int


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000, examples=["What is the refund approval policy?"])
    collection_id: str | None = Field(None, examples=["policies"])
    user_id: str | None = Field(
        None, description="Optional. Identity always comes from the bearer token; if set it must match it."
    )
    retrieval: RetrievalOverrides | None = None


class Citation(BaseModel):
    index: int = Field(description="The [n] marker used in the answer")
    document_id: str
    chunk_id: str
    filename: str
    page: int | None
    section: str | None
    score: float
    text: str


class Timings(BaseModel):
    retrieval_ms: float
    rerank_ms: float
    generation_ms: float


class Usage(BaseModel):
    model: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    cited_chunk_ids: list[str]
    insufficient_evidence: bool
    latency_ms: float
    timings: Timings
    usage: Usage
    retrieval: RetrievalConfig
    trace_id: str
    cached: bool = False


class FeedbackRequest(BaseModel):
    trace_id: str = Field(min_length=1, max_length=64)
    rating: int = Field(ge=-1, le=1)
    comment: str | None = Field(None, max_length=2000)


class EvalRunRequest(BaseModel):
    name: str = Field("adhoc", max_length=128)
    retrieval: RetrievalOverrides | None = None
    k: int = Field(10, ge=1, le=50, description="Cut-off for Recall@K / Precision@K / nDCG@K")
    generate: bool = Field(True, description="Also run generation and score answers")
    judge: str = Field("heuristic", pattern="^(heuristic|llm)$")
    limit: int | None = Field(None, ge=1, description="Evaluate only the first N cases")


class EvalRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    status: str
    config_json: dict[str, Any]
    recall_at_k: float | None
    precision_at_k: float | None
    mrr: float | None
    ndcg_at_k: float | None
    groundedness: float | None
    p95_latency_ms: float | None
    cost_per_query: float | None
    metrics_json: dict[str, Any]
    error: str | None
    created_at: datetime
    finished_at: datetime | None


class IngestionStatus(BaseModel):
    counts: dict[str, int]
    total_pages: int
    total_chunks: int
    failed: list[DocumentOut]
