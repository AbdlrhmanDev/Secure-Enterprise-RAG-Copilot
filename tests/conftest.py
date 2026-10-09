"""Test setup: an isolated, fully offline stack (SQLite, in-memory Qdrant, hashing embedder,
lexical reranker, extractive generator). Environment is configured before the app is imported."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="rag-tests-"))
os.environ.update(
    {
        "ENV": "test",
        "DATABASE_URL": f"sqlite:///{(_TMP / 'test.db').as_posix()}",
        "QDRANT_URL": "",
        "QDRANT_PATH": ":memory:",
        "REDIS_URL": "",
        "STORAGE_DIR": str(_TMP / "uploads"),
        "EMBEDDING_PROVIDER": "hash",
        "RERANKER_PROVIDER": "lexical",
        "LLM_PROVIDER": "extractive",
        "INGEST_BACKEND": "inline",
        "INGEST_MAX_RETRIES": "1",
        "DEFAULT_COLLECTION": "kb",
        "MIN_RELEVANCE_SCORE": "0",
        "LOG_DOCUMENT_TEXT": "false",
    }
)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

HANDBOOK = b"""# Employee Handbook

## Vacation

Employees receive 24 vacation days per year. Vacation requests are approved by the direct manager.

## Expenses

Meals are reimbursed up to 60 USD per day. Receipts must be uploaded within 30 days.
"""

SALARY_BANDS = b"""Salary Bands
============

Compensation
------------

The salary band for senior analysts ranges from 91000 to 118000 USD. The confidential codename
for the planned acquisition is Project Marigold.
"""

RUNBOOK = b"""<html><head><title>Deploy Runbook</title></head><body><h1>Deploy Runbook</h1>
<h2>Rollback</h2><p>A rollback of the payment service must start within 15 minutes of a failed deploy.
The rollback command is owned by the platform team.</p></body></html>"""


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def auth(client):
    cache: dict[str, dict] = {}

    def headers(user_id: str) -> dict:
        if user_id not in cache:
            response = client.post("/auth/token", json={"user_id": user_id})
            assert response.status_code == 200, response.text
            cache[user_id] = {"Authorization": f"Bearer {response.json()['access_token']}"}
        return cache[user_id]

    return headers


def upload(client, headers, name: str, data: bytes, roles: str, collection: str = "kb"):
    return client.post(
        "/documents",
        headers=headers,
        files={"file": (name, data)},
        data={"collection_id": collection, "allowed_roles": roles},
    )


@pytest.fixture(scope="session")
def seeded(client, auth):
    """Three documents with different ACLs in collection `kb`. Returns {filename: document}."""
    documents = {}
    for name, data, roles, user in (
        ("handbook.md", HANDBOOK, "employee", "u_employee"),
        ("salary-bands.txt", SALARY_BANDS, "finance", "u_finance"),
        ("runbook.html", RUNBOOK, "engineering", "u_engineer"),
    ):
        response = upload(client, auth(user), name, data, roles)
        assert response.status_code == 202, response.text
        document = response.json()["document"]
        assert document["status"] == "ready", document
        documents[name] = document
    return documents
