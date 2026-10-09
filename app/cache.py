"""Tiny cache abstraction: Redis when REDIS_URL is set, in-process otherwise. Failures never break a request."""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class Cache(Protocol):
    def get(self, key: str) -> Any | None: ...
    def set(self, key: str, value: Any, ttl: int) -> None: ...


class MemoryCache:
    def __init__(self, max_items: int = 2048):
        self._items: dict[str, tuple[float, Any]] = {}
        self._max_items = max_items
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._items.get(key)
            if entry is None:
                return None
            if entry[0] < time.monotonic():
                del self._items[key]
                return None
            return entry[1]

    def set(self, key: str, value: Any, ttl: int) -> None:
        with self._lock:
            if len(self._items) >= self._max_items:
                self._items.pop(next(iter(self._items)))
            self._items[key] = (time.monotonic() + ttl, value)


class RedisCache:
    def __init__(self, url: str, prefix: str = "rag:"):
        import redis

        self._client = redis.Redis.from_url(url, socket_timeout=1.0, socket_connect_timeout=1.0)
        self._prefix = prefix

    def get(self, key: str) -> Any | None:
        try:
            raw = self._client.get(self._prefix + key)
        except Exception as exc:
            logger.warning("cache get failed", extra={"error_type": type(exc).__name__})
            return None
        return json.loads(raw) if raw is not None else None

    def set(self, key: str, value: Any, ttl: int) -> None:
        try:
            self._client.set(self._prefix + key, json.dumps(value), ex=ttl)
        except Exception as exc:
            logger.warning("cache set failed", extra={"error_type": type(exc).__name__})


def build_cache(redis_url: str | None) -> Cache:
    return RedisCache(redis_url) if redis_url else MemoryCache()
