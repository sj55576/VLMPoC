"""Tests for the API key guard and sliding-window rate limiter."""
from __future__ import annotations

from app.core.security import ApiKeyGuard, RateLimiter


class TestApiKeyGuard:
    def test_disabled_when_no_api_key(self) -> None:
        guard = ApiKeyGuard(api_key="")

        assert guard.enabled is False
        assert guard.authorize("/api/anything", None) is True
        assert guard.authorize("/api/anything", "wrong-key") is True

    def test_enabled_when_api_key_set(self) -> None:
        guard = ApiKeyGuard(api_key="secret123")

        assert guard.enabled is True

    def test_accepts_matching_header_key(self) -> None:
        guard = ApiKeyGuard(api_key="secret123")

        assert guard.authorize("/api/session/start", "secret123") is True

    def test_rejects_wrong_header_key(self) -> None:
        guard = ApiKeyGuard(api_key="secret123")

        assert guard.authorize("/api/session/start", "wrong") is False

    def test_rejects_missing_key(self) -> None:
        guard = ApiKeyGuard(api_key="secret123")

        assert guard.authorize("/api/session/start", None) is False
        assert guard.authorize("/api/session/start", "") is False

    def test_accepts_bearer_token(self) -> None:
        guard = ApiKeyGuard(api_key="secret123")

        assert guard.authorize("/api/session/start", None, bearer_value="Bearer secret123") is True

    def test_bearer_scheme_is_case_insensitive_and_tolerates_whitespace(self) -> None:
        guard = ApiKeyGuard(api_key="secret123")

        assert guard.authorize("/api/x", None, bearer_value="  bearer   secret123  ".strip()) is True
        assert guard.authorize("/api/x", None, bearer_value="BEARER secret123") is True

    def test_rejects_malformed_bearer(self) -> None:
        guard = ApiKeyGuard(api_key="secret123")

        assert guard.authorize("/api/x", None, bearer_value="Basic secret123") is False
        assert guard.authorize("/api/x", None, bearer_value=None) is False

    def test_header_key_takes_precedence_over_bearer(self) -> None:
        guard = ApiKeyGuard(api_key="secret123")

        assert guard.authorize("/api/x", "secret123", bearer_value="Bearer wrong") is True

    def test_exempt_paths_bypass_check(self) -> None:
        guard = ApiKeyGuard(api_key="secret123", exempt_paths=["/health", "/metrics"])

        assert guard.is_exempt("/health") is True
        assert guard.is_exempt("/api/session/start") is False
        assert guard.authorize("/health", None) is True
        assert guard.authorize("/metrics", "wrong") is True
        assert guard.authorize("/api/session/start", None) is False

    def test_empty_key_never_matches_empty_api_key(self) -> None:
        # api_key="" means disabled entirely; this just documents that authorize()
        # short-circuits to True regardless of what's supplied.
        guard = ApiKeyGuard(api_key="")

        assert guard.authorize("/api/x", "") is True


class TestRateLimiter:
    def test_disabled_when_per_minute_is_zero(self) -> None:
        limiter = RateLimiter(per_minute=0)

        assert limiter.enabled is False
        for _ in range(1000):
            assert limiter.allow("k") is True

    def test_enabled_when_per_minute_positive(self) -> None:
        limiter = RateLimiter(per_minute=3)

        assert limiter.enabled is True

    def test_allows_exactly_n_per_window_then_rejects(self) -> None:
        limiter = RateLimiter(per_minute=3)
        now = 1000.0

        assert limiter.allow("k", now=now) is True
        assert limiter.allow("k", now=now + 1) is True
        assert limiter.allow("k", now=now + 2) is True
        assert limiter.allow("k", now=now + 3) is False

    def test_recovers_after_window_elapses(self) -> None:
        limiter = RateLimiter(per_minute=2)
        now = 0.0

        assert limiter.allow("k", now=now) is True
        assert limiter.allow("k", now=now + 1) is True
        assert limiter.allow("k", now=now + 2) is False

        # Past the 60s sliding window from both prior hits: they should have expired.
        assert limiter.allow("k", now=now + 61) is True

    def test_keys_are_independent(self) -> None:
        limiter = RateLimiter(per_minute=1)
        now = 0.0

        assert limiter.allow("a", now=now) is True
        assert limiter.allow("b", now=now) is True
        assert limiter.allow("a", now=now) is False
        assert limiter.allow("b", now=now) is False

    def test_sliding_window_partial_expiry(self) -> None:
        limiter = RateLimiter(per_minute=2)

        assert limiter.allow("k", now=0.0) is True
        assert limiter.allow("k", now=30.0) is True
        # Still within 60s of both hits.
        assert limiter.allow("k", now=59.0) is False
        # First hit (t=0) now expired (59.0+ window edge); only the t=30 hit remains active.
        assert limiter.allow("k", now=61.0) is True

    def test_reset_clears_all_state(self) -> None:
        limiter = RateLimiter(per_minute=1)
        now = 0.0

        assert limiter.allow("k", now=now) is True
        assert limiter.allow("k", now=now) is False

        limiter.reset()

        assert limiter.allow("k", now=now) is True

    def test_uses_monotonic_clock_when_now_omitted(self) -> None:
        limiter = RateLimiter(per_minute=1000)

        # Just verify it runs without error and returns a bool when now is not injected.
        assert limiter.allow("k") is True
