"""Ingestion: register an upload (idempotently), then parse -> clean -> chunk -> embed -> index."""

from __future__ import annotations

import hashlib
import logging
import re
import time
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.auth.acl import assignable_roles, can_access
from app.auth.users import ROLES, User
from app.config import get_settings
from app.db import Chunk, Document, session_scope, utcnow
from app.errors import AppError, ForbiddenError, PayloadTooLargeError, UnsupportedMediaError
from app.ingestion.chunker import chunk_blocks
from app.ingestion.parsers import SUPPORTED_EXTENSIONS, ParseError, parse_document
from app.retrieval.bm25 import index_text
from app.retrieval.vector_store import VectorRecord
from app.services import Services, get_services

logger = logging.getLogger(__name__)

_COLLECTION_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def document_id_for(collection_id: str, filename: str) -> str:
    """Stable id per (collection, filename): re-uploading a file updates it instead of duplicating it."""
    return "doc_" + hashlib.sha1(f"{collection_id}/{filename}".encode()).hexdigest()[:12]


def register_upload(
    db: Session,
    *,
    filename: str,
    data: bytes,
    collection_id: str,
    allowed_roles: list[str],
    user: User,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> tuple[Document, bool]:
    """Store the file and create/update its document row.

    Returns (document, needs_ingestion). Identical content with identical chunking settings is a
    no-op, which makes re-running a sync idempotent.
    """
    settings = get_settings()
    filename = Path(filename).name  # never trust client-supplied paths
    if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise UnsupportedMediaError(
            f"Unsupported file type. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    if not data:
        raise AppError("Uploaded file is empty")
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise PayloadTooLargeError(f"File exceeds the {settings.max_upload_mb} MB limit")
    if not _COLLECTION_RE.match(collection_id):
        raise AppError("collection_id must match [a-z0-9][a-z0-9_-]{0,63}")
    unknown = [r for r in allowed_roles if r not in ROLES]
    if unknown:
        raise AppError(f"Unknown roles: {', '.join(unknown)}", details={"valid_roles": list(ROLES)})
    roles = assignable_roles(user, allowed_roles)
    if not roles:
        raise ForbiddenError("You can only restrict a document to roles you hold yourself")

    chunking = {
        "chunk_size": chunk_size or settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap if chunk_overlap is None else chunk_overlap,
    }
    if not 0 <= chunking["chunk_overlap"] < chunking["chunk_size"]:
        raise AppError("chunk_overlap must be >= 0 and smaller than chunk_size")

    checksum = hashlib.sha256(data).hexdigest()
    doc_id = document_id_for(collection_id, filename)
    document = db.get(Document, doc_id)

    if document is not None:
        if not can_access(user, document.allowed_roles):
            # Without this, anyone could overwrite (or learn about) a restricted document.
            raise ForbiddenError("A document with this name exists and you are not allowed to replace it")
        same_chunking = all(document.meta.get(k) == v for k, v in chunking.items())
        if document.checksum == checksum and same_chunking and document.status != "failed":
            if sorted(document.allowed_roles) != sorted(roles):
                update_document_roles(db, get_services(), document, roles)
            return document, False
        version = document.version + (1 if document.checksum != checksum else 0)
    else:
        version = 1

    target = Path(settings.storage_dir) / doc_id / f"v{version}_{filename}"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)

    if document is None:
        document = Document(id=doc_id, collection_id=collection_id, filename=filename)
        db.add(document)
    document.source_uri = str(target)
    document.checksum = checksum
    document.version = version
    document.status = "pending"
    document.allowed_roles = roles
    document.error = None
    document.attempts = 0
    document.uploaded_by = user.id
    document.meta = {**(document.meta or {}), **chunking}
    db.commit()
    return document, True


def update_document_roles(db: Session, services: Services, document: Document, roles: list[str]) -> None:
    """Change a document's ACL in both stores. `updated_at` changes, which invalidates BM25 and answer caches."""
    services.vector_store.set_document_roles(document.id, roles)
    document.allowed_roles = roles
    document.updated_at = utcnow()
    db.commit()


def delete_document(db: Session, services: Services, document: Document) -> None:
    services.vector_store.delete_document(document.id)
    db.execute(delete(Chunk).where(Chunk.document_id == document.id))
    db.delete(document)
    db.commit()


def ingest_document(document_id: str, services: Services | None = None) -> None:
    """Run the ingestion pipeline for one document. Raises on failure after recording it on the row."""
    services = services or get_services()
    started = time.perf_counter()

    with session_scope() as db:
        document = db.get(Document, document_id)
        if document is None:
            raise ParseError(f"document {document_id} no longer exists")
        document.status = "processing"
        document.attempts += 1
        document.error = None
        filename, source_uri, version = document.filename, document.source_uri, document.version
        collection_id, roles, meta = document.collection_id, list(document.allowed_roles), dict(document.meta)

    try:
        parsed = parse_document(filename, Path(source_uri).read_bytes())
        drafts = chunk_blocks(parsed.blocks, meta["chunk_size"], meta["chunk_overlap"])
        if not drafts:
            raise ParseError("document contains no extractable text")
        vectors = services.embedder.embed_documents(
            [index_text(parsed.title, d.section, d.text) for d in drafts]
        )
        chunk_ids = [f"{document_id}-{i:04d}" for i in range(len(drafts))]

        with session_scope() as db:
            document = db.get(Document, document_id)
            if document is None or document.version != version:
                return  # superseded by a newer upload while we were working
            # Replace, never append: old chunks of a previous version must not stay retrievable.
            db.execute(delete(Chunk).where(Chunk.document_id == document_id))
            services.vector_store.delete_document(document_id)
            db.add_all(
                Chunk(
                    id=chunk_id,
                    document_id=document_id,
                    ordinal=i,
                    text=draft.text,
                    page=draft.page,
                    section=draft.section,
                    token_count=draft.token_count,
                    meta={"version": version},
                )
                for i, (chunk_id, draft) in enumerate(zip(chunk_ids, drafts, strict=True))
            )
            # Use the ACL as it is now, in case it changed while the document was being parsed.
            roles = list(document.allowed_roles)
            services.vector_store.upsert(
                [
                    VectorRecord(chunk_id, vector, document_id, collection_id, roles)
                    for chunk_id, vector in zip(chunk_ids, vectors, strict=True)
                ]
            )
            document.status = "ready"
            document.page_count = parsed.page_count
            document.chunk_count = len(drafts)
            document.meta = {**meta, "title": parsed.title, **parsed.meta}
            document.updated_at = utcnow()
    except Exception as exc:
        # The message is built from exception type and parser diagnostics, never document text.
        message = str(exc) if isinstance(exc, ParseError) else f"{type(exc).__name__}: {str(exc)[:300]}"
        with session_scope() as db:
            document = db.get(Document, document_id)
            if document is not None:
                document.status = "failed"
                document.error = message
        logger.error(
            "ingestion failed",
            extra={"document_id": document_id, "error_type": type(exc).__name__, "retryable": is_retryable(exc)},
        )
        raise

    logger.info(
        "document ingested",
        extra={
            "document_id": document_id,
            "version": version,
            "pages": parsed.page_count,
            "chunks": len(drafts),
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        },
    )


def is_retryable(exc: Exception) -> bool:
    """Parse errors are deterministic; everything else (network, model, store) may succeed on retry."""
    return not isinstance(exc, ParseError)


def ingest_with_retries(document_id: str) -> None:
    """In-process ingestion with backoff. Used by the `inline` and `background` backends."""
    settings = get_settings()
    for attempt in range(1, settings.ingest_max_retries + 1):
        try:
            ingest_document(document_id)
            return
        except Exception as exc:
            if not is_retryable(exc) or attempt == settings.ingest_max_retries:
                return  # the failure is recorded on the document row
            time.sleep(min(2**attempt, 10))


def pending_document_ids(db: Session) -> list[str]:
    return list(db.scalars(select(Document.id).where(Document.status.in_(("pending", "processing")))))
