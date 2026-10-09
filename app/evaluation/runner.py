"""Runs an evaluation configuration over the labeled dataset and stores the results."""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.acl import can_access
from app.auth.users import USERS
from app.db import Chunk, Document, EvalRun, new_id, session_scope, utcnow
from app.evaluation import metrics as m
from app.evaluation.dataset import DEFAULT_DATASET, sync_eval_cases
from app.generation.service import answer_query
from app.retrieval.types import RetrievalConfig
from app.services import Services, get_services

logger = logging.getLogger(__name__)

# Case tags. "answerable": evidence exists and the asking user may read it.
# "unanswerable": nothing in the corpus answers it. "acl": the evidence exists but the asking
# user is not authorized, so the only correct behaviours are zero leakage and a refusal.
ANSWERABLE, UNANSWERABLE, ACL = "answerable", "unanswerable", "acl"


class EvalConfig(BaseModel):
    name: str = "adhoc"
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    k: int = Field(10, ge=1, le=50)
    generate: bool = True
    judge: str = "heuristic"  # or "llm"
    limit: int | None = None
    dataset_path: str = str(DEFAULT_DATASET)


def create_run(db: Session, config: EvalConfig) -> EvalRun:
    run = EvalRun(id=new_id("run"), name=config.name, status="running", config_json=config.model_dump())
    db.add(run)
    db.commit()
    return run


def execute_run(run_id: str, services: Services | None = None) -> None:
    """Entry point for background execution: never raises, records failure on the run row."""
    services = services or get_services()
    with session_scope() as db:
        run = db.get(EvalRun, run_id)
        config = EvalConfig(**run.config_json)
    try:
        with session_scope() as db:
            results = evaluate(db, services, config)
        status, error = "completed", None
    except Exception as exc:
        logger.exception("evaluation failed", extra={"run_id": run_id})
        results, status, error = {}, "failed", f"{type(exc).__name__}: {str(exc)[:300]}"
    with session_scope() as db:
        run = db.get(EvalRun, run_id)
        summary = results.get("summary", {})
        run.status = status
        run.error = error
        run.metrics_json = results
        run.recall_at_k = summary.get("recall_at_k")
        run.precision_at_k = summary.get("precision_at_k")
        run.mrr = summary.get("mrr")
        run.ndcg_at_k = summary.get("ndcg_at_k")
        run.groundedness = summary.get("groundedness")
        run.p95_latency_ms = summary.get("latency_p95_ms")
        run.cost_per_query = summary.get("cost_per_query_usd")
        run.finished_at = utcnow()


def evaluate(db: Session, services: Services, config: EvalConfig) -> dict:
    cases = sync_eval_cases(db, Path(config.dataset_path))
    if config.limit:
        cases = cases[: config.limit]
    k = config.k

    # Independent ACL oracle: chunk -> document ACL straight from the metadata store.
    chunk_document = dict(db.execute(select(Chunk.id, Chunk.document_id)).all())
    document_roles = dict(db.execute(select(Document.id, Document.allowed_roles)).all())

    judge = None
    if config.generate and config.judge == "llm":
        from app.evaluation.judge import LLMJudge

        judge = LLMJudge(services.settings)

    per_case = []
    retrieval_scores: dict[str, list[float]] = {"recall": [], "precision": [], "rr": [], "ndcg": [], "hit": []}
    correctness, grounded, citation, refusals, false_refusals = [], [], [], [], []
    latency = {"retrieval": [], "rerank": [], "generation": [], "total": []}
    input_tokens, output_tokens, costs = [], [], []
    leaked_chunks = 0
    unlabeled = 0

    for case in cases:
        user = USERS[case.user_id]
        output = answer_query(
            db,
            services,
            query=case.query,
            user=user,
            collection_id=case.collection_id,
            config=config.retrieval,
            use_cache=False,
            generate=config.generate,
        )
        response, retrieval = output.response, output.retrieval
        ranked = [c.chunk_id for c in retrieval.ranked]
        relevant = set(case.relevant_chunk_ids)
        tags = set(case.tags)
        record: dict = {"id": case.id, "tags": case.tags, "user_id": case.user_id, "ranked": ranked[:k]}

        # Zero-leakage check over everything the retriever returned, for every case.
        leaks = [c for c in ranked if not can_access(user, document_roles.get(chunk_document.get(c), []))]
        leaked_chunks += len(leaks)
        record["leaked_chunks"] = len(leaks)

        if ANSWERABLE in tags:
            if not relevant:
                unlabeled += 1  # evidence document not ingested: excluded rather than scored as a miss
            else:
                scores = {
                    "recall": m.recall_at_k(ranked, relevant, k),
                    "precision": m.precision_at_k(ranked, relevant, k),
                    "rr": m.reciprocal_rank(ranked, relevant),
                    "ndcg": m.ndcg_at_k(ranked, relevant, k),
                    "hit": 1.0 if relevant & set(ranked[:k]) else 0.0,
                }
                for key, value in scores.items():
                    retrieval_scores[key].append(value)
                record.update({f"{key}_at_{k}" if key != "rr" else key: round(v, 4) for key, v in scores.items()})

        latency["retrieval"].append(response.timings.retrieval_ms)
        latency["rerank"].append(response.timings.rerank_ms)

        if config.generate:
            latency["generation"].append(response.timings.generation_ms)
            latency["total"].append(response.latency_ms)
            input_tokens.append(response.usage.input_tokens)
            output_tokens.append(response.usage.output_tokens)
            costs.append(response.usage.estimated_cost_usd)
            record["refused"] = response.insufficient_evidence
            record["cited"] = response.cited_chunk_ids

            if ANSWERABLE in tags and relevant:
                false_refusals.append(1.0 if response.insufficient_evidence else 0.0)
                cited_texts = [c.text for c in response.citations]
                if judge and not response.insufficient_evidence:
                    grade = judge.grade(
                        question=case.query,
                        expected=case.expected_answer,
                        answer=response.answer,
                        cited_texts=cited_texts,
                    )
                    correct, faithful = float(grade["correct"]), float(grade["faithful"])
                elif response.insufficient_evidence:
                    correct, faithful = 0.0, None
                else:
                    correct = 1.0 if m.answer_token_recall(response.answer, case.expected_answer) >= 0.8 else 0.0
                    faithful = m.groundedness(response.answer, cited_texts)
                correctness.append(correct)
                record["correct"] = correct
                if faithful is not None:
                    grounded.append(faithful)
                    citation.append(m.citation_precision(response.cited_chunk_ids, relevant))
                    record["groundedness"] = round(faithful, 4)
            elif tags & {UNANSWERABLE, ACL}:
                refusals.append(1.0 if response.insufficient_evidence else 0.0)

        per_case.append(record)

    def rounded(value: float | None, digits: int = 4) -> float | None:
        return round(value, digits) if value is not None else None

    total_key = "total" if config.generate else "retrieval"
    totals = latency[total_key] if config.generate else [
        a + b for a, b in zip(latency["retrieval"], latency["rerank"], strict=True)
    ]
    summary = {
        "k": k,
        "cases": len(cases),
        "answerable_cases": len(retrieval_scores["recall"]),
        "unlabeled_cases": unlabeled,
        "recall_at_k": rounded(m.mean(retrieval_scores["recall"])),
        "precision_at_k": rounded(m.mean(retrieval_scores["precision"])),
        "mrr": rounded(m.mean(retrieval_scores["rr"])),
        "ndcg_at_k": rounded(m.mean(retrieval_scores["ndcg"])),
        "hit_rate_at_k": rounded(m.mean(retrieval_scores["hit"])),
        "acl_leaked_chunks": leaked_chunks,
        "answer_correctness": rounded(m.mean(correctness)),
        "groundedness": rounded(m.mean(grounded)),
        "citation_correctness": rounded(m.mean(citation)),
        "refusal_accuracy": rounded(m.mean(refusals)),
        "false_refusal_rate": rounded(m.mean(false_refusals)),
        "refusal_cases": len(refusals),
        "retrieval_p50_ms": rounded(m.percentile(latency["retrieval"], 50), 1),
        "retrieval_p95_ms": rounded(m.percentile(latency["retrieval"], 95), 1),
        "rerank_p50_ms": rounded(m.percentile(latency["rerank"], 50), 1),
        "rerank_p95_ms": rounded(m.percentile(latency["rerank"], 95), 1),
        "generation_p50_ms": rounded(m.percentile(latency["generation"], 50), 1),
        "generation_p95_ms": rounded(m.percentile(latency["generation"], 95), 1),
        "latency_p50_ms": rounded(m.percentile(totals, 50), 1),
        "latency_p95_ms": rounded(m.percentile(totals, 95), 1),
        "latency_scope": "end_to_end" if config.generate else "retrieval_and_rerank",
        "mean_input_tokens": rounded(m.mean(input_tokens), 1),
        "mean_output_tokens": rounded(m.mean(output_tokens), 1),
        "cost_per_query_usd": rounded(m.mean(costs), 6),
        "generator": services.llm.name if config.generate else None,
        "judge": (judge.model if judge else "heuristic") if config.generate else None,
        "embedder": services.embedder.name,
        "reranker": services.reranker.name if config.retrieval.rerank else None,
    }
    return {"summary": summary, "cases": per_case}
