"""Query-time retrieval: ACL scoping -> BM25 / dense -> fusion -> ACL re-check -> rerank."""

from __future__ import annotations

import logging
import time

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.acl import can_access
from app.auth.users import User
from app.db import Chunk, Document
from app.retrieval.bm25 import index_text
from app.retrieval.fusion import Ranked, reciprocal_rank_fusion, weighted_score_fusion
from app.retrieval.types import RetrievalConfig, RetrievalResult, ScoredChunk
from app.services import Services

logger = logging.getLogger(__name__)


def authorized_documents(db: Session, collection_id: str, user: User) -> dict[str, Document]:
    """Ready documents in the collection that `user` may read. Everything downstream is scoped to this."""
    rows = db.scalars(
        select(Document).where(Document.collection_id == collection_id, Document.status == "ready")
    ).all()
    return {doc.id: doc for doc in rows if can_access(user, doc.allowed_roles)}


def retrieve(
    db: Session, services: Services, *, query: str, user: User, collection_id: str, config: RetrievalConfig
) -> RetrievalResult:
    started = time.perf_counter()
    documents = authorized_documents(db, collection_id, user)
    if not documents:
        return RetrievalResult(retrieval_ms=(time.perf_counter() - started) * 1000)

    rankings: dict[str, Ranked] = {}
    if config.mode in ("bm25", "hybrid"):
        rankings["bm25"] = services.bm25.search(db, query, collection_id, documents, config.top_k)
    if config.mode in ("dense", "hybrid"):
        rankings["dense"] = services.vector_store.search(
            services.embed_query(query),
            collection_id=collection_id,
            roles=None if user.is_admin else list(user.roles),
            limit=config.top_k,
        )

    if config.mode == "hybrid":
        weights = {"bm25": config.bm25_weight, "dense": config.dense_weight}
        if config.fusion == "rrf":
            fused = reciprocal_rank_fusion(rankings, k=config.rrf_k, weights=weights)
        else:
            fused = weighted_score_fusion(rankings, weights)
    else:
        fused = rankings[config.mode]

    candidates = fused[: config.rerank_candidates if config.rerank else config.top_k]
    rows = {
        row.id: row
        for row in db.scalars(select(Chunk).where(Chunk.id.in_([chunk_id for chunk_id, _ in candidates])))
    }

    # Defence in depth: the retrievers already filter by ACL, but nothing reaches the reranker
    # or the LLM unless its document is in the authorized set computed from the metadata store.
    ranked: list[ScoredChunk] = []
    dropped = 0
    for chunk_id, score in candidates:
        row = rows.get(chunk_id)
        if row is None:
            continue  # vector exists for a chunk that has since been re-ingested
        document = documents.get(row.document_id)
        if document is None:
            dropped += 1
            continue
        ranked.append(
            ScoredChunk(
                chunk_id=row.id,
                document_id=row.document_id,
                filename=document.filename,
                title=document.meta.get("title"),
                text=row.text,
                page=row.page,
                section=row.section,
                score=score,
                retrieval_score=score,
            )
        )
    if dropped:
        logger.warning(
            "acl post-filter dropped candidates",
            extra={"user_id": user.id, "collection_id": collection_id, "dropped": dropped},
        )
    retrieval_ms = (time.perf_counter() - started) * 1000

    rerank_ms = 0.0
    if config.rerank and ranked:
        rerank_started = time.perf_counter()
        scores = services.reranker.score(query, [index_text(c.title, c.section, c.text) for c in ranked])
        for chunk, score in zip(ranked, scores, strict=True):
            chunk.rerank_score = chunk.score = score
        ranked.sort(key=lambda c: -c.score)
        rerank_ms = (time.perf_counter() - rerank_started) * 1000

    return RetrievalResult(
        ranked=ranked,
        context=ranked[: config.final_context_k],
        retrieval_ms=retrieval_ms,
        rerank_ms=rerank_ms,
        acl_dropped=dropped,
    )
