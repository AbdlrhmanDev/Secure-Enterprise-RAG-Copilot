"""FastAPI application: API, auth, structured errors, trace ids and the bundled web UI."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api import auth, documents, evaluation, query
from app.config import get_settings
from app.db import init_db
from app.errors import install_error_handlers
from app.observability import configure_logging, new_trace_id, trace_id_var

logger = logging.getLogger("app.request")
WEB_DIR = Path(__file__).parent / "web"


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    init_db()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Secure Enterprise RAG Copilot",
        version="0.1.0",
        description=(
            "Hybrid retrieval (BM25 + dense), cross-encoder reranking, grounded answers with citations, "
            "document-level ACLs and an evaluation suite. Get a token from `POST /auth/token`, then "
            "use **Authorize**."
        ),
        lifespan=lifespan,
    )
    install_error_handlers(app)

    @app.middleware("http")
    async def trace_requests(request: Request, call_next):
        incoming = request.headers.get("x-trace-id", "")
        trace_id = incoming if incoming.isalnum() and len(incoming) <= 64 else new_trace_id()
        # Each request runs in its own context, so this never leaks into another request.
        trace_id_var.set(trace_id)
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Trace-Id"] = trace_id
        if not request.url.path.startswith("/ui"):
            logger.info(
                "request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                },
            )
        return response

    @app.get("/health", tags=["ops"])
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/ui/")

    for router in (auth.router, documents.router, query.router, evaluation.router):
        app.include_router(router)
    app.mount("/ui", StaticFiles(directory=WEB_DIR, html=True), name="ui")
    return app


app = create_app()
