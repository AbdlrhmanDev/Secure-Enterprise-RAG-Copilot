"""The query path: retrieve -> gate -> generate -> map citations. Shared by the API and the evaluator."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.auth.users import User
from app.generation.citations import extract_citations, strip_citations
from app.generation.llm import Generation
from app.generation.prompts import INSUFFICIENT_EVIDENCE_ANSWER
from app.observability import current_trace_id
from app.retrieval.pipeline import authorized_documents, retrieve
from app.retrieval.types import RetrievalConfig, RetrievalResult
from app.schemas import Citation, QueryResponse, Timings, Usage
from app.services import Services

logger = logging.getLogger(__name__)


@dataclass
class QueryOutput:
    response: QueryResponse
    retrieval: RetrievalResult = field(default_factory=RetrievalResult)


def _cache_key(db: Session, services: Services, query: str, user: User, collection_id: str, cfg: RetrievalConfig) -> str:
    # The key covers who is asking (roles) and exactly which document versions/ACLs they can
    # see, so a cached answer can never be served across an ACL boundary or after re-ingestion.
    documents = authorized_documents(db, collection_id, user)
    visible = sorted(f"{d.id}:{d.version}:{d.updated_at.isoformat()}" for d in documents.values())
    material = json.dumps(
        [query.strip(), collection_id, sorted(user.roles), cfg.model_dump(), services.llm.name, visible]
    )
    return "answer:" + hashlib.sha256(material.encode()).hexdigest()


def answer_query(
    db: Session,
    services: Services,
    *,
    query: str,
    user: User,
    collection_id: str,
    config: RetrievalConfig,
    use_cache: bool = True,
    generate: bool = True,
) -> QueryOutput:
    settings = services.settings
    started = time.perf_counter()
    use_cache = use_cache and settings.cache_enabled and generate

    cache_key = _cache_key(db, services, query, user, collection_id, config) if use_cache else None
    if cache_key:
        cached = services.cache.get(cache_key)
        if cached is not None:
            response = QueryResponse(**cached)
            response.cached = True
            response.trace_id = current_trace_id()
            response.latency_ms = round((time.perf_counter() - started) * 1000, 1)
            return QueryOutput(response)

    retrieval = retrieve(db, services, query=query, user=user, collection_id=collection_id, config=config)
    contexts = retrieval.context

    generation_started = time.perf_counter()
    below_threshold = config.rerank and bool(contexts) and contexts[0].score < settings.min_relevance_score
    if not generate or not contexts or below_threshold:
        # Nothing authorized and relevant was found: refuse without spending an LLM call.
        generation = Generation(INSUFFICIENT_EVIDENCE_ANSWER, True, services.llm.name)
    else:
        generation = services.llm.generate(query, contexts)
    generation_ms = (time.perf_counter() - generation_started) * 1000

    answer, cited_numbers = extract_citations(generation.answer, len(contexts))
    if generation.insufficient_evidence:
        # A refusal has no supporting passages; models sometimes list what they looked at.
        answer, cited_numbers = re.sub(r"\s+([.,;])", r"\1", strip_citations(answer)).strip(), []
    citations = [
        Citation(
            index=number,
            document_id=contexts[number - 1].document_id,
            chunk_id=contexts[number - 1].chunk_id,
            filename=contexts[number - 1].filename,
            page=contexts[number - 1].page,
            section=contexts[number - 1].section,
            score=round(contexts[number - 1].score, 4),
            text=contexts[number - 1].text,
        )
        for number in cited_numbers
    ]
    cost = (
        generation.input_tokens * settings.llm_input_price_per_mtok
        + generation.output_tokens * settings.llm_output_price_per_mtok
    ) / 1_000_000
    response = QueryResponse(
        answer=answer,
        citations=citations,
        cited_chunk_ids=[c.chunk_id for c in citations],
        insufficient_evidence=generation.insufficient_evidence,
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
        timings=Timings(
            retrieval_ms=round(retrieval.retrieval_ms, 1),
            rerank_ms=round(retrieval.rerank_ms, 1),
            generation_ms=round(generation_ms, 1),
        ),
        usage=Usage(
            model=generation.model,
            input_tokens=generation.input_tokens,
            output_tokens=generation.output_tokens,
            estimated_cost_usd=round(cost, 6),
        ),
        retrieval=config,
        trace_id=current_trace_id(),
    )

    # Identifiers, sizes and timings only: no query or document text.
    log_fields = {
        "user_id": user.id,
        "collection_id": collection_id,
        "query_chars": len(query),
        "mode": config.mode,
        "candidates": len(retrieval.ranked),
        "context_chunk_ids": [c.chunk_id for c in contexts],
        "cited_chunk_ids": response.cited_chunk_ids,
        "insufficient_evidence": response.insufficient_evidence,
        "latency_ms": response.latency_ms,
        **response.timings.model_dump(),
        "input_tokens": generation.input_tokens,
        "output_tokens": generation.output_tokens,
    }
    if settings.log_document_text:
        log_fields["query"] = query
    logger.info("query answered", extra=log_fields)

    if cache_key:
        services.cache.set(cache_key, response.model_dump(mode="json"), ttl=settings.cache_ttl_seconds)
    return QueryOutput(response, retrieval)
