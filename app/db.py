"""Database engine, session handling and ORM models (PostgreSQL in Docker, SQLite for local dev/tests)."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.config import get_settings


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("collection_id", "filename", name="uq_document_collection_filename"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    collection_id: Mapped[str] = mapped_column(String(128), index=True)
    filename: Mapped[str] = mapped_column(String(512))
    source_uri: Mapped[str] = mapped_column(String(1024))
    checksum: Mapped[str] = mapped_column(String(64), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    allowed_roles: Mapped[list[str]] = mapped_column(JSON, default=list)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    uploaded_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(96), primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    ordinal: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section: Mapped[str | None] = mapped_column(String(512), nullable=True)
    token_count: Mapped[int] = mapped_column(Integer)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)


class EvalCase(Base):
    __tablename__ = "eval_cases"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    query: Mapped[str] = mapped_column(Text)
    relevant_chunk_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    expected_answer: Mapped[str] = mapped_column(Text, default="")
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    collection_id: Mapped[str] = mapped_column(String(128))
    user_id: Mapped[str] = mapped_column(String(64))
    evidence: Mapped[list[dict[str, str]]] = mapped_column(JSON, default=list)


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(16), default="running")
    config_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    recall_at_k: Mapped[float | None] = mapped_column(Float, nullable=True)
    precision_at_k: Mapped[float | None] = mapped_column(Float, nullable=True)
    mrr: Mapped[float | None] = mapped_column(Float, nullable=True)
    ndcg_at_k: Mapped[float | None] = mapped_column(Float, nullable=True)
    groundedness: Mapped[float | None] = mapped_column(Float, nullable=True)
    p95_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    cost_per_query: Mapped[float | None] = mapped_column(Float, nullable=True)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    trace_id: Mapped[str] = mapped_column(String(64), index=True)
    user_id: Mapped[str] = mapped_column(String(64))
    rating: Mapped[int] = mapped_column(Integer)  # +1 / -1
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


@lru_cache
def get_engine() -> Engine:
    url = get_settings().database_url
    kwargs: dict[str, Any] = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
        path = url.split("///", 1)[-1]
        if path and path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url, **kwargs)


@lru_cache
def get_sessionmaker() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def init_db() -> None:
    try:
        Base.metadata.create_all(get_engine())
    except DBAPIError:
        # API and worker start together and can race to create the same tables; whoever
        # loses just needs to look again.
        Base.metadata.create_all(get_engine())


def get_db() -> Iterator[Session]:
    """FastAPI dependency."""
    db = get_sessionmaker()()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    db = get_sessionmaker()()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
