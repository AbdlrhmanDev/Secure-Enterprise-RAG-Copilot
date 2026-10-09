"""Arq worker for background ingestion. Run with: arq app.ingestion.worker.WorkerSettings"""

from __future__ import annotations

import asyncio

from arq import Retry, create_pool
from arq.connections import RedisSettings

from app.config import get_settings
from app.db import init_db
from app.ingestion.pipeline import ingest_document, is_retryable
from app.observability import configure_logging, new_trace_id, trace_id_var


async def ingest_task(ctx: dict, document_id: str, trace_id: str | None = None) -> str:
    trace_id_var.set(trace_id or new_trace_id())
    try:
        # Parsing and embedding are CPU-bound and synchronous: keep them off the event loop.
        await asyncio.to_thread(ingest_document, document_id)
    except Exception as exc:
        if is_retryable(exc) and ctx["job_try"] < WorkerSettings.max_tries:
            raise Retry(defer=ctx["job_try"] * 5) from exc
        return "failed"
    return "ready"


async def _startup(ctx: dict) -> None:
    configure_logging(get_settings().log_level)
    init_db()


def _redis_settings() -> RedisSettings:
    return RedisSettings.from_dsn(get_settings().redis_url or "redis://localhost:6379")


class WorkerSettings:
    functions = [ingest_task]
    on_startup = _startup
    redis_settings = _redis_settings()
    max_tries = get_settings().ingest_max_retries
    job_timeout = 900


async def _enqueue(document_id: str, trace_id: str) -> None:
    pool = await create_pool(_redis_settings())
    try:
        await pool.enqueue_job("ingest_task", document_id, trace_id)
    finally:
        await pool.aclose()


def enqueue_ingestion(document_id: str, trace_id: str) -> None:
    """Called from synchronous request handlers (which run in a worker thread, without an event loop)."""
    asyncio.run(_enqueue(document_id, trace_id))
