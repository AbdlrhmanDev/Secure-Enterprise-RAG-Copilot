"""Evaluation dataset loading and resolution of evidence quotes to chunk ids.

Cases are labeled with (source file, quote) pairs rather than hard-coded chunk ids, so the same
labels stay valid when chunk size or overlap changes. Chunk ids are resolved against whatever is
currently indexed and stored on the eval_cases rows.
"""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import Chunk, Document, EvalCase
from app.text import words

DEFAULT_DATASET = Path("evals/dataset.jsonl")


def load_dataset(path: Path = DEFAULT_DATASET) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _contains(haystack: list[str], needle: list[str]) -> bool:
    return f" {' '.join(needle)} " in f" {' '.join(haystack)} "


def resolve_relevant_chunks(db: Session, collection_id: str, evidence: list[dict]) -> list[str]:
    """Chunks that contain an evidence quote (compared on word tokens, so layout differences don't matter)."""
    relevant: list[str] = []
    for item in evidence:
        document = db.scalar(
            select(Document).where(Document.collection_id == collection_id, Document.filename == item["source"])
        )
        if document is None:
            continue
        quote = words(item["quote"])
        chunks = db.scalars(select(Chunk).where(Chunk.document_id == document.id).order_by(Chunk.ordinal)).all()
        matches = [c.id for c in chunks if _contains(words(c.text), quote)]
        if not matches and chunks:
            # The quote straddles a chunk boundary: fall back to the chunk covering most of it.
            quote_set = set(quote)
            best = max(chunks, key=lambda c: len(quote_set & set(words(c.text))))
            if len(quote_set & set(words(best.text))) / len(quote_set) >= 0.6:
                matches = [best.id]
        relevant.extend(m for m in matches if m not in relevant)
    return relevant


def sync_eval_cases(db: Session, path: Path = DEFAULT_DATASET) -> list[EvalCase]:
    """Upsert the dataset into eval_cases, resolving relevant_chunk_ids against the current index."""
    cases = []
    for record in load_dataset(path):
        case = db.get(EvalCase, record["id"]) or EvalCase(id=record["id"])
        case.query = record["query"]
        case.expected_answer = record.get("expected_answer", "")
        case.tags = record.get("tags", [])
        case.collection_id = record["collection_id"]
        case.user_id = record["user_id"]
        case.evidence = record.get("evidence", [])
        case.relevant_chunk_ids = resolve_relevant_chunks(db, case.collection_id, case.evidence)
        db.add(case)
        cases.append(case)
    db.commit()
    return cases
