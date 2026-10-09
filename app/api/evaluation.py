from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.jwt import require_roles
from app.auth.users import User
from app.config import get_settings
from app.db import EvalRun, get_db
from app.errors import AppError, NotFoundError
from app.evaluation.runner import EvalConfig, create_run, execute_run
from app.retrieval.types import RetrievalConfig
from app.schemas import EvalRunOut, EvalRunRequest

router = APIRouter(prefix="/eval", tags=["evaluation"])

_evaluator = require_roles("engineering")


@router.post("/run", response_model=EvalRunOut, status_code=202, summary="Start an evaluation run")
def run_evaluation(
    body: EvalRunRequest,
    background: BackgroundTasks,
    _: User = Depends(_evaluator),
    db: Session = Depends(get_db),
) -> EvalRun:
    """Runs the labeled dataset through the given retrieval configuration in the background.
    Poll `GET /eval/runs/{id}` until `status` is `completed`."""
    settings = get_settings()
    if body.judge == "llm" and settings.resolved_llm_provider() != "openai":
        raise AppError("judge='llm' requires OPENAI_API_KEY")
    retrieval = RetrievalConfig.from_settings(settings)
    if body.retrieval:
        retrieval = body.retrieval.apply(retrieval)
    config = EvalConfig(
        name=body.name, retrieval=retrieval, k=body.k, generate=body.generate, judge=body.judge, limit=body.limit
    )
    run = create_run(db, config)
    background.add_task(execute_run, run.id)
    return run


@router.get("/runs", response_model=list[EvalRunOut], summary="Recent runs (summary metrics only)")
def list_runs(_: User = Depends(_evaluator), db: Session = Depends(get_db)) -> list[EvalRunOut]:
    runs = db.scalars(select(EvalRun).order_by(EvalRun.created_at.desc()).limit(100))
    outputs = [EvalRunOut.model_validate(run) for run in runs]
    for output in outputs:
        # Per-case results can be large; they are served by GET /eval/runs/{id}.
        output.metrics_json = {"summary": output.metrics_json.get("summary", {})}
    return outputs


@router.get("/runs/{run_id}", response_model=EvalRunOut, summary="Metrics and configuration of a run")
def get_run(run_id: str, _: User = Depends(_evaluator), db: Session = Depends(get_db)) -> EvalRun:
    run = db.get(EvalRun, run_id)
    if run is None:
        raise NotFoundError("Evaluation run not found")
    return run
