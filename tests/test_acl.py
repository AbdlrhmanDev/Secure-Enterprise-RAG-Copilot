"""Security tests: unauthorized chunks must never reach the candidate list, the LLM context,
the citations, the cache or any read endpoint."""

from __future__ import annotations

import pytest

from app.auth.acl import can_access
from app.auth.users import USERS
from app.db import Chunk, Document, session_scope
from app.generation.service import answer_query
from app.retrieval.pipeline import retrieve
from app.retrieval.types import RetrievalConfig
from app.services import get_services
from tests.conftest import SALARY_BANDS, upload

SALARY_QUESTION = "What is the salary band for senior analysts?"
ROLLBACK_QUESTION = "When must a rollback of the payment service start?"
CONFIGS = [
    RetrievalConfig(mode=mode, rerank=rerank, top_k=50, rerank_candidates=50, final_context_k=10)
    for mode in ("bm25", "dense", "hybrid")
    for rerank in (False, True)
]


def _retrieved_documents(user_id: str, query: str, config: RetrievalConfig) -> set[str]:
    with session_scope() as db:
        result = retrieve(db, get_services(), query=query, user=USERS[user_id], collection_id="kb", config=config)
        return {chunk.document_id for chunk in result.ranked}


@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: f"{c.mode}-rerank={c.rerank}")
@pytest.mark.parametrize(
    "user_id, query, forbidden",
    [
        ("u_employee", SALARY_QUESTION, "salary-bands.txt"),
        ("u_engineer", SALARY_QUESTION, "salary-bands.txt"),
        ("u_employee", ROLLBACK_QUESTION, "runbook.html"),
        ("u_finance", ROLLBACK_QUESTION, "runbook.html"),
    ],
)
def test_no_retriever_returns_unauthorized_chunks(seeded, config, user_id, query, forbidden):
    retrieved = _retrieved_documents(user_id, query, config)
    assert seeded[forbidden]["id"] not in retrieved
    # ... and nothing else the user may not read slipped in either.
    with session_scope() as db:
        for document_id in retrieved:
            assert can_access(USERS[user_id], db.get(Document, document_id).allowed_roles)


@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: f"{c.mode}-rerank={c.rerank}")
@pytest.mark.parametrize("user_id", ["u_finance", "u_admin"])
def test_authorized_users_do_retrieve_the_restricted_document(seeded, config, user_id):
    assert seeded["salary-bands.txt"]["id"] in _retrieved_documents(user_id, SALARY_QUESTION, config)


def test_answer_for_unauthorized_user_contains_no_restricted_content(client, auth, seeded):
    response = client.post("/query", headers=auth("u_employee"), json={"query": SALARY_QUESTION, "collection_id": "kb"})
    body = response.json()
    assert response.status_code == 200
    assert "91000" not in body["answer"] and "Marigold" not in body["answer"]
    assert all(c["document_id"] != seeded["salary-bands.txt"]["id"] for c in body["citations"])
    assert all("91000" not in c["text"] for c in body["citations"])


def test_authorized_user_gets_the_answer_with_a_citation(client, auth, seeded):
    response = client.post("/query", headers=auth("u_finance"), json={"query": SALARY_QUESTION, "collection_id": "kb"})
    body = response.json()
    assert "91000" in body["answer"]
    assert body["citations"][0]["document_id"] == seeded["salary-bands.txt"]["id"]
    assert not body["insufficient_evidence"]


def test_cached_answer_is_not_served_across_an_acl_boundary(client, auth, seeded):
    payload = {"query": "What is the confidential codename for the planned acquisition?", "collection_id": "kb"}
    first = client.post("/query", headers=auth("u_finance"), json=payload).json()
    assert "Marigold" in first["answer"]
    again = client.post("/query", headers=auth("u_finance"), json=payload).json()
    assert again["cached"] is True
    other = client.post("/query", headers=auth("u_employee"), json=payload).json()
    assert other["cached"] is False
    assert "Marigold" not in other["answer"]
    assert all("Marigold" not in c["text"] for c in other["citations"])


def test_post_filter_blocks_chunks_when_the_vector_index_acl_is_stale(seeded):
    """Defence in depth: even if the vector payload wrongly grants access, the metadata store wins."""
    services = get_services()
    document_id = seeded["salary-bands.txt"]["id"]
    services.vector_store.set_document_roles(document_id, ["employee"])  # simulate a corrupted index
    try:
        config = RetrievalConfig(mode="dense", rerank=False, top_k=50)
        with session_scope() as db:
            result = retrieve(db, services, query=SALARY_QUESTION, user=USERS["u_employee"], collection_id="kb", config=config)
        assert result.acl_dropped > 0
        assert document_id not in {c.document_id for c in result.ranked}
        with session_scope() as db:
            output = answer_query(
                db, services, query=SALARY_QUESTION, user=USERS["u_employee"], collection_id="kb", config=config, use_cache=False
            )
        assert "91000" not in output.response.answer
    finally:
        services.vector_store.set_document_roles(document_id, ["finance"])


def test_read_endpoints_hide_restricted_documents_and_chunks(client, auth, seeded):
    document_id = seeded["salary-bands.txt"]["id"]
    with session_scope() as db:
        chunk_id = db.query(Chunk).filter(Chunk.document_id == document_id).first().id

    employee = auth("u_employee")
    assert client.get(f"/documents/{document_id}", headers=employee).status_code == 404
    assert client.get(f"/chunks/{chunk_id}", headers=employee).status_code == 404
    assert document_id not in {d["id"] for d in client.get("/documents", headers=employee).json()}

    finance = auth("u_finance")
    assert client.get(f"/documents/{document_id}", headers=finance).status_code == 200
    assert "91000" in client.get(f"/chunks/{chunk_id}", headers=finance).json()["text"]


def test_unauthorized_user_cannot_overwrite_a_restricted_document(client, auth, seeded):
    response = upload(client, auth("u_employee"), "salary-bands.txt", b"All salaries are public now.", "employee")
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "forbidden"
    finance = client.post("/query", headers=auth("u_finance"), json={"query": SALARY_QUESTION, "collection_id": "kb"})
    assert "91000" in finance.json()["answer"]


def test_users_cannot_assign_roles_they_do_not_hold(client, auth):
    response = upload(client, auth("u_employee"), "leak.txt", b"Some text here.", "finance")
    assert response.status_code == 403


def test_acl_change_takes_effect_immediately(client, auth, seeded):
    response = upload(client, auth("u_finance"), "bonus-memo.txt", b"The retention bonus for analysts is 7500 USD.", "finance")
    document_id = response.json()["document"]["id"]
    payload = {"query": "What is the retention bonus for analysts?", "collection_id": "kb"}

    assert "7500" not in client.post("/query", headers=auth("u_employee"), json=payload).json()["answer"]

    forbidden = client.patch(f"/documents/{document_id}", headers=auth("u_finance"), json={"allowed_roles": ["employee"]})
    assert forbidden.status_code == 403
    granted = client.patch(
        f"/documents/{document_id}", headers=auth("u_admin"), json={"allowed_roles": ["finance", "employee"]}
    )
    assert granted.status_code == 200
    assert "7500" in client.post("/query", headers=auth("u_employee"), json=payload).json()["answer"]

    client.patch(f"/documents/{document_id}", headers=auth("u_admin"), json={"allowed_roles": ["finance"]})
    revoked = client.post("/query", headers=auth("u_employee"), json=payload).json()
    assert "7500" not in revoked["answer"]
    assert all("7500" not in c["text"] for c in revoked["citations"])


def test_requests_without_a_valid_token_are_rejected(client, auth):
    assert client.post("/query", json={"query": "hi"}).status_code == 401
    bad = client.post("/query", headers={"Authorization": "Bearer not-a-jwt"}, json={"query": "hi"})
    assert bad.status_code == 401 and bad.json()["error"]["code"] == "unauthorized"


def test_body_user_id_cannot_impersonate_another_user(client, auth, seeded):
    response = client.post(
        "/query", headers=auth("u_employee"), json={"query": SALARY_QUESTION, "collection_id": "kb", "user_id": "u_finance"}
    )
    assert response.status_code == 403


def test_non_admins_cannot_use_admin_endpoints(client, auth, seeded):
    assert client.get("/admin/ingestion", headers=auth("u_employee")).status_code == 403
    assert client.delete(f"/documents/{seeded['handbook.md']['id']}", headers=auth("u_engineer")).status_code == 403
    assert client.post("/eval/run", headers=auth("u_employee"), json={}).status_code == 403


def test_salary_fixture_is_what_the_tests_assume():
    assert b"91000" in SALARY_BANDS and b"Marigold" in SALARY_BANDS
