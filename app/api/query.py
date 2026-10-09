from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.auth.jwt import get_current_user
from app.auth.users import User
from app.config import get_settings
from app.db import Feedback, get_db, new_id
from app.errors import ForbiddenError
from app.generation.service import answer_query
from app.retrieval.types import RetrievalConfig
from app.schemas import FeedbackRequest, QueryRequest, QueryResponse
from app.services import get_services

router = APIRouter(tags=["query"])


@router.get("/config", response_model=RetrievalConfig, summary="Server default retrieval configuration")
def retrieval_defaults(_: User = Depends(get_current_user)) -> RetrievalConfig:
    return RetrievalConfig.from_settings(get_settings())


@router.post("/query", response_model=QueryResponse, summary="Ask a question, get a cited answer")
def query(body: QueryRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> QueryResponse:
    """Retrieves only from documents the caller may read, then answers strictly from that context.

    The answer contains inline `[n]` markers; `citations` maps each marker to its source chunk.
    When the accessible documents do not contain the answer, `insufficient_evidence` is true.
    """
    # Identity comes from the signed token. A body user_id is accepted for API compatibility
    # but can never be used to act as someone else.
    if body.user_id is not None and body.user_id != user.id:
        raise ForbiddenError("user_id does not match the authenticated user")
    settings = get_settings()
    config = RetrievalConfig.from_settings(settings)
    if body.retrieval:
        config = body.retrieval.apply(config)
    output = answer_query(
        db,
        get_services(),
        query=body.query,
        user=user,
        collection_id=body.collection_id or settings.default_collection,
        config=config,
    )
    return output.response


@router.post("/feedback", status_code=201, summary="Rate an answer (thumbs up/down)")
def feedback(body: FeedbackRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    row = Feedback(id=new_id("fb"), trace_id=body.trace_id, user_id=user.id, rating=body.rating, comment=body.comment)
    db.add(row)
    db.commit()
    return {"id": row.id}
