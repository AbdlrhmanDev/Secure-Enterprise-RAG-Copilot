from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Response, UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.auth.acl import can_access
from app.auth.jwt import get_current_user, require_roles
from app.auth.users import ROLES, User
from app.config import get_settings
from app.db import Chunk, Document, get_db
from app.errors import AppError, ForbiddenError, NotFoundError
from app.ingestion.pipeline import (
    delete_document,
    ingest_with_retries,
    register_upload,
    update_document_roles,
)
from app.observability import current_trace_id
from app.schemas import (
    ChunkOut,
    CollectionOut,
    DocumentOut,
    DocumentUpdate,
    IngestionStatus,
    UploadResponse,
)
from app.services import get_services

router = APIRouter(tags=["documents"])


def _dispatch_ingestion(document_id: str, background: BackgroundTasks) -> None:
    backend = get_settings().ingest_backend
    if backend == "arq":
        from app.ingestion.worker import enqueue_ingestion

        enqueue_ingestion(document_id, current_trace_id())
    elif backend == "inline":
        ingest_with_retries(document_id)
    else:
        background.add_task(ingest_with_retries, document_id)


def _visible_document(db: Session, document_id: str, user: User) -> Document:
    document = db.get(Document, document_id)
    # 404 rather than 403 so that restricted documents are not discoverable by id.
    if document is None or not can_access(user, document.allowed_roles):
        raise NotFoundError("Document not found")
    return document


@router.post("/documents", response_model=UploadResponse, status_code=202, summary="Upload a document")
def upload_document(
    background: BackgroundTasks,
    response: Response,
    file: UploadFile = File(...),
    collection_id: str = Form(None),
    allowed_roles: str = Form("employee", description="Comma-separated roles allowed to read the document"),
    chunk_size: int | None = Form(None, ge=32, le=2000),
    chunk_overlap: int | None = Form(None, ge=0, le=1000),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UploadResponse:
    """Accepts PDF, DOCX, Markdown, HTML and TXT. Ingestion runs in the background; poll
    `GET /documents/{id}` for status. Re-uploading identical content is a no-op."""
    settings = get_settings()
    limit = settings.max_upload_mb * 1024 * 1024
    data = file.file.read(limit + 1)
    document, needs_ingestion = register_upload(
        db,
        filename=file.filename or "",
        data=data,
        collection_id=collection_id or settings.default_collection,
        allowed_roles=[r.strip() for r in allowed_roles.split(",") if r.strip()],
        user=user,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if needs_ingestion:
        _dispatch_ingestion(document.id, background)
        db.refresh(document)
    else:
        response.status_code = 200
    return UploadResponse(document=DocumentOut.model_validate(document), deduplicated=not needs_ingestion)


@router.get("/documents", response_model=list[DocumentOut], summary="List documents you may read")
def list_documents(
    collection_id: str | None = None,
    status: str | None = None,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[Document]:
    query = select(Document).order_by(Document.collection_id, Document.filename)
    if collection_id:
        query = query.where(Document.collection_id == collection_id)
    if status:
        query = query.where(Document.status == status)
    return [d for d in db.scalars(query) if can_access(user, d.allowed_roles)]


@router.get("/documents/{document_id}", response_model=DocumentOut, summary="Document metadata and ingestion status")
def get_document(document_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> Document:
    return _visible_document(db, document_id, user)


@router.post("/documents/{document_id}/retry", response_model=DocumentOut, status_code=202, summary="Retry a failed ingestion")
def retry_document(
    document_id: str,
    background: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Document:
    document = _visible_document(db, document_id, user)
    if not user.is_admin and document.uploaded_by != user.id:
        raise ForbiddenError("Only an admin or the uploader can retry ingestion")
    if document.status != "failed":
        raise AppError(f"Document is '{document.status}', only failed documents can be retried")
    document.status = "pending"
    document.attempts = 0
    db.commit()
    _dispatch_ingestion(document.id, background)
    db.refresh(document)
    return document


@router.patch("/documents/{document_id}", response_model=DocumentOut, summary="Change a document's ACL (admin)")
def update_document(
    document_id: str,
    body: DocumentUpdate,
    _: User = Depends(require_roles()),
    db: Session = Depends(get_db),
) -> Document:
    document = db.get(Document, document_id)
    if document is None:
        raise NotFoundError("Document not found")
    unknown = [r for r in body.allowed_roles if r not in ROLES]
    if unknown:
        raise AppError(f"Unknown roles: {', '.join(unknown)}", details={"valid_roles": list(ROLES)})
    update_document_roles(db, get_services(), document, list(dict.fromkeys(body.allowed_roles)))
    return document


@router.delete("/documents/{document_id}", status_code=204, summary="Delete a document and its chunks (admin)")
def remove_document(document_id: str, _: User = Depends(require_roles()), db: Session = Depends(get_db)) -> None:
    document = db.get(Document, document_id)
    if document is None:
        raise NotFoundError("Document not found")
    delete_document(db, get_services(), document)


@router.get("/chunks/{chunk_id}", response_model=ChunkOut, summary="A source passage with its document metadata")
def get_chunk(chunk_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> ChunkOut:
    chunk = db.get(Chunk, chunk_id)
    document = db.get(Document, chunk.document_id) if chunk else None
    if chunk is None or document is None or not can_access(user, document.allowed_roles):
        raise NotFoundError("Chunk not found")
    return ChunkOut(
        chunk_id=chunk.id,
        document_id=document.id,
        filename=document.filename,
        collection_id=document.collection_id,
        page=chunk.page,
        section=chunk.section,
        token_count=chunk.token_count,
        text=chunk.text,
    )


@router.get("/collections", response_model=list[CollectionOut], summary="Collections containing documents you may read")
def list_collections(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> list[CollectionOut]:
    counts: dict[str, int] = {}
    for document in db.scalars(select(Document)):
        if can_access(user, document.allowed_roles):
            counts[document.collection_id] = counts.get(document.collection_id, 0) + 1
    return [CollectionOut(id=name, document_count=count) for name, count in sorted(counts.items())]


@router.get("/admin/ingestion", response_model=IngestionStatus, tags=["admin"], summary="Ingestion status and failed documents (admin)")
def ingestion_status(_: User = Depends(require_roles()), db: Session = Depends(get_db)) -> IngestionStatus:
    counts = dict(db.execute(select(Document.status, func.count()).group_by(Document.status)).all())
    pages, chunks = db.execute(
        select(func.coalesce(func.sum(Document.page_count), 0), func.coalesce(func.sum(Document.chunk_count), 0)).where(
            Document.status == "ready"
        )
    ).one()
    failed = db.scalars(select(Document).where(Document.status == "failed").order_by(Document.updated_at.desc())).all()
    return IngestionStatus(
        counts=counts,
        total_pages=int(pages),
        total_chunks=int(chunks),
        failed=[DocumentOut.model_validate(d) for d in failed],
    )
