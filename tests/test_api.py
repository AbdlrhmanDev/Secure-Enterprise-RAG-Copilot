"""Integration tests through the HTTP API: ingestion lifecycle, querying, errors, evaluation, logging."""

from __future__ import annotations

import json
import logging
import time

from app.db import Chunk, session_scope
from app.evaluation.runner import EvalConfig, evaluate
from app.retrieval.types import RetrievalConfig
from app.services import get_services
from tests.conftest import HANDBOOK, upload


def test_health_docs_and_ui_are_served(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/docs").status_code == 200
    assert "/query" in client.get("/openapi.json").json()["paths"]
    assert "<title>" in client.get("/ui/").text


def test_query_returns_grounded_answer_with_working_citations(client, auth, seeded):
    response = client.post(
        "/query", headers=auth("u_employee"), json={"query": "How many vacation days do employees receive?", "collection_id": "kb"}
    )
    assert response.status_code == 200
    body = response.json()
    assert "24 vacation days" in body["answer"] and "[1]" in body["answer"]
    citation = body["citations"][0]
    assert citation["document_id"] == seeded["handbook.md"]["id"]
    assert citation["section"] == "Vacation"
    assert body["cited_chunk_ids"] == [c["chunk_id"] for c in body["citations"]]
    assert body["latency_ms"] > 0 and body["trace_id"] == response.headers["X-Trace-Id"]

    # The citation can be opened: the chunk endpoint returns the supporting passage and metadata.
    chunk = client.get(f"/chunks/{citation['chunk_id']}", headers=auth("u_employee")).json()
    assert chunk["text"] == citation["text"] and chunk["filename"] == "handbook.md"


def test_insufficient_evidence_is_stated(client, auth, seeded):
    body = client.post(
        "/query", headers=auth("u_employee"), json={"query": "Which satellite orbits Neptune fastest?", "collection_id": "kb"}
    ).json()
    assert body["insufficient_evidence"] is True
    assert body["citations"] == []


def test_retrieval_overrides_are_applied_and_validated(client, auth, seeded):
    payload = {"query": "meal reimbursement", "collection_id": "kb", "retrieval": {"mode": "bm25", "rerank": False}}
    body = client.post("/query", headers=auth("u_employee"), json=payload).json()
    assert body["retrieval"]["mode"] == "bm25" and body["retrieval"]["rerank"] is False

    payload["retrieval"] = {"mode": "telepathy"}
    error = client.post("/query", headers=auth("u_employee"), json=payload)
    assert error.status_code == 422
    assert error.json()["error"]["code"] == "validation_error"
    assert error.json()["error"]["trace_id"]


def test_reingestion_is_idempotent_and_versions_replace_old_chunks(client, auth):
    headers = auth("u_employee")
    v1 = b"# Parking\n\n## Fees\n\nThe parking fee is 40 USD per month.\n"
    first = upload(client, headers, "parking.md", v1, "employee")
    assert first.status_code == 202 and first.json()["document"]["version"] == 1

    same = upload(client, headers, "parking.md", v1, "employee")
    assert same.status_code == 200 and same.json()["deduplicated"] is True
    assert same.json()["document"]["id"] == first.json()["document"]["id"]
    assert same.json()["document"]["version"] == 1

    v2 = b"# Parking\n\n## Fees\n\nThe parking fee is 55 USD per month.\n"
    second = upload(client, headers, "parking.md", v2, "employee")
    assert second.status_code == 202 and second.json()["document"]["version"] == 2

    with session_scope() as db:
        texts = [c.text for c in db.query(Chunk).filter(Chunk.document_id == second.json()["document"]["id"])]
    assert any("55 USD" in t for t in texts) and not any("40 USD" in t for t in texts)

    answer = client.post("/query", headers=headers, json={"query": "What is the parking fee per month?", "collection_id": "kb"}).json()
    assert "55 USD" in answer["answer"] and "40 USD" not in answer["answer"]


def test_unsupported_type_and_empty_file_return_structured_errors(client, auth):
    response = upload(client, auth("u_employee"), "notes.exe", b"binary", "employee")
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media_type"
    assert upload(client, auth("u_employee"), "empty.txt", b"", "employee").status_code == 400
    assert client.get("/documents/doc_missing", headers=auth("u_employee")).json()["error"]["code"] == "not_found"


def test_failed_ingestion_is_visible_to_admins_and_retryable(client, auth):
    response = upload(client, auth("u_employee"), "scan.pdf", b"%PDF-1.4 corrupted", "employee")
    document = response.json()["document"]
    assert response.status_code == 202 and document["status"] == "failed"
    assert document["error"] and document["attempts"] == 1

    status = client.get("/admin/ingestion", headers=auth("u_admin")).json()
    assert status["counts"]["failed"] >= 1
    assert document["id"] in {d["id"] for d in status["failed"]}

    retried = client.post(f"/documents/{document['id']}/retry", headers=auth("u_admin"))
    assert retried.status_code == 202 and retried.json()["status"] == "failed"  # same bytes fail again

    # A corrected upload under the same name recovers the document.
    fixed = upload(client, auth("u_employee"), "scan.pdf", _valid_pdf(), "employee")
    assert fixed.json()["document"]["status"] == "ready"


def _valid_pdf() -> bytes:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 6, "The scanner on floor two is serviced every quarter.")
    return bytes(pdf.output())


def test_collections_and_document_listing(client, auth, seeded):
    collections = {c["id"]: c["document_count"] for c in client.get("/collections", headers=auth("u_admin")).json()}
    assert collections["kb"] >= 3
    visible = client.get("/documents", headers=auth("u_employee"), params={"collection_id": "kb"}).json()
    assert "handbook.md" in {d["filename"] for d in visible}


def test_feedback_is_recorded(client, auth):
    response = client.post("/feedback", headers=auth("u_employee"), json={"trace_id": "abc123", "rating": 1})
    assert response.status_code == 201 and response.json()["id"].startswith("fb_")


def test_document_text_is_not_logged(client, auth, seeded, caplog):
    with caplog.at_level(logging.DEBUG):
        upload(client, auth("u_employee"), "secret-note.txt", b"The vault combination is 8-6-7-5-3-0-9.", "employee")
        client.post("/query", headers=auth("u_employee"), json={"query": "What is the vault combination?", "collection_id": "kb"})
    logged = " ".join(f"{r.getMessage()} {json.dumps(r.__dict__, default=str)}" for r in caplog.records)
    assert "query answered" in logged and "document ingested" in logged
    assert "8-6-7-5-3-0-9" not in logged and "vault combination" not in logged


def test_evaluation_scores_a_labeled_dataset(seeded, tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    cases = [
        {"id": "t1", "query": "How many vacation days do employees receive?", "expected_answer": "24 vacation days",
         "evidence": [{"source": "handbook.md", "quote": "Employees receive 24 vacation days per year"}],
         "tags": ["answerable"], "collection_id": "kb", "user_id": "u_employee"},
        {"id": "t2", "query": "What is the salary band for senior analysts?", "expected_answer": "91000 to 118000 USD",
         "evidence": [{"source": "salary-bands.txt", "quote": "ranges from 91000 to 118000 USD"}],
         "tags": ["answerable"], "collection_id": "kb", "user_id": "u_finance"},
        {"id": "t3", "query": "What is the salary band for senior analysts?", "expected_answer": "",
         "evidence": [{"source": "salary-bands.txt", "quote": "ranges from 91000 to 118000 USD"}],
         "tags": ["acl"], "collection_id": "kb", "user_id": "u_employee"},
        {"id": "t4", "query": "Which satellite orbits Neptune fastest?", "expected_answer": "", "evidence": [],
         "tags": ["unanswerable"], "collection_id": "kb", "user_id": "u_employee"},
    ]
    dataset.write_text("\n".join(json.dumps(c) for c in cases), encoding="utf-8")
    config = EvalConfig(name="test", retrieval=RetrievalConfig(mode="hybrid", rerank=True), k=5, dataset_path=str(dataset))
    with session_scope() as db:
        summary = evaluate(db, get_services(), config)["summary"]
    assert summary["answerable_cases"] == 2 and summary["refusal_cases"] == 2
    assert summary["recall_at_k"] == 1.0 and summary["mrr"] == 1.0
    assert summary["acl_leaked_chunks"] == 0
    assert summary["answer_correctness"] == 1.0 and summary["citation_correctness"] == 1.0
    assert summary["refusal_accuracy"] == 1.0
    assert summary["latency_p95_ms"] > 0


def test_eval_endpoints_run_in_background_and_report_metrics(client, auth, seeded):
    started = client.post("/eval/run", headers=auth("u_engineer"), json={"name": "api-test", "limit": 5, "k": 5})
    assert started.status_code == 202
    run_id = started.json()["id"]
    for _ in range(50):
        run = client.get(f"/eval/runs/{run_id}", headers=auth("u_engineer")).json()
        if run["status"] != "running":
            break
        time.sleep(0.1)
    assert run["status"] == "completed", run
    assert run["config_json"]["retrieval"]["mode"] == "hybrid"
    assert run["metrics_json"]["summary"]["acl_leaked_chunks"] == 0
    listed = client.get("/eval/runs", headers=auth("u_engineer")).json()
    assert run_id in {r["id"] for r in listed} and "cases" not in listed[0]["metrics_json"]


def test_handbook_fixture_is_what_the_tests_assume():
    assert b"24 vacation days" in HANDBOOK
