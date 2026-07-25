"""Optional, dependency-free API key authorization and per-key rate limiting.

Both primitives are opt-in (disabled by default) so the demo keeps working out of the
box. No FastAPI/starlette imports here; the web layer wires these in as middleware.
"""
from __future__ import annotations

import secrets
import threading
import time
from collections import deque
from collections.abc import Iterable

_BEARER_PREFIX = "bearer "


class ApiKeyGuard:
    """Authorizes requests against a single static API key.

    Disabled (always authorized) when `api_key` is empty. `exempt_paths` bypass the
    check entirely (e.g. /health) even when the guard is enabled.
    """

    def __init__(self, api_key: str = "", exempt_paths: Iterable[str] = ()) -> None:
        self._api_key = api_key
        self._exempt_paths = frozenset(exempt_paths)

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    def is_exempt(self, path: str) -> bool:
        return path in self._exempt_paths

    def authorize(self, path: str, header_value: str | None, bearer_value: str | None = None) -> bool:
        """True when disabled, the path is exempt, or a supplied key matches (constant-time)."""
        if not self.enabled or self.is_exempt(path):
            return True
        candidate = self._extract_key(header_value, bearer_value)
        if not candidate:
            return False
        return secrets.compare_digest(candidate, self._api_key)

    def _extract_key(self, header_value: str | None, bearer_value: str | None) -> str:
        if header_value:
            return header_value.strip()
        if bearer_value:
            value = bearer_value.strip()
            if value.lower().startswith(_BEARER_PREFIX):
                return value[len(_BEARER_PREFIX):].strip()
        return ""


class RateLimiter:
    """Sliding 60-second window rate limiter, keyed by an arbitrary string (e.g. API key or IP).

    Disabled (always allows) when `per_minute` <= 0. Thread-safe via a simple lock; the
    caller is expected to invoke `allow` from a single asyncio event loop, so contention
    is negligible.
    """

    _WINDOW_SECONDS = 60.0

    def __init__(self, per_minute: int = 0) -> None:
        self._per_minute = per_minute
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._per_minute > 0

    def allow(self, key: str, now: float | None = None) -> bool:
        """Record a hit for `key` and return whether it stays within the per-minute limit."""
        if not self.enabled:
            return True
        current = now if now is not None else time.monotonic()
        cutoff = current - self._WINDOW_SECONDS
        with self._lock:
            hits = self._hits.get(key)
            if hits is not None:
                while hits and hits[0] <= cutoff:
                    hits.popleft()
                if not hits:
                    # Prune keys with no recent activity so the map does not grow unbounded.
                    del self._hits[key]
                    hits = None
            if hits is None:
                hits = deque()
                self._hits[key] = hits
            if len(hits) >= self._per_minute:
                return False
            hits.append(current)
            return True

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()
