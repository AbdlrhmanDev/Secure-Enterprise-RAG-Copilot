"""Structured JSON logging with a per-request trace id."""

from __future__ import annotations

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime

trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")

_RESERVED = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


def new_trace_id() -> str:
    return uuid.uuid4().hex


def current_trace_id() -> str:
    return trace_id_var.get()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "trace_id": trace_id_var.get(),
            "message": record.getMessage(),
        }
        # Anything passed via `extra=` is emitted as a structured field.
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpcore", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
