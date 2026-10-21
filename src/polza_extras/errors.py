from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal


class PolzaError(Exception):
    """Any error returned by the Polza API, parsed from its error body."""

    def __init__(
        self,
        status: int,
        message: str,
        code: str | None = None,
        trace_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.status = status
        self.message = message
        self.code = code
        self.trace_id = trace_id
        self.metadata = metadata or {}
        self.details = details or {}

    def __str__(self) -> str:
        parts = [str(self.status)]
        if self.code:
            parts.append(self.code)
        text = f"[{' '.join(parts)}] {self.message}"
        if self.trace_id:
            text += f" (trace_id={self.trace_id})"
        return text


class AuthError(PolzaError):
    """401: missing, invalid or revoked API key."""


class InsufficientFundsError(PolzaError):
    """402: balance or spending limit exhausted (or a parallel-request reserve conflict)."""


class ForbiddenError(PolzaError):
    """403: access denied, including requests rejected by moderation."""


class RateLimitError(PolzaError):
    """429. `scope` is "own" for Polza's per-organization limit (has Retry-After), else "provider"."""

    @property
    def scope(self) -> Literal["own", "provider"]:
        return "own" if self.retry_after is not None else "provider"


class ServerError(PolzaError):
    """408, 5xx: provider timeout, bad gateway, unavailable."""


class RequestError(PolzaError):
    """Other 4xx: bad request, not found, payload too large."""


_BY_STATUS: dict[int, type[PolzaError]] = {
    401: AuthError,
    402: InsufficientFundsError,
    403: ForbiddenError,
    408: ServerError,
}


def _header(headers: Mapping[str, str], name: str) -> str | None:
    for key, value in headers.items():
        if key.lower() == name.lower():
            return value
    return None


def _retry_after(headers: Mapping[str, str]) -> float | None:
    raw = _header(headers, "retry-after")
    try:
        return float(raw) if raw is not None else None
    except ValueError:
        return None


def error_from_response(status: int, body: Any, headers: Mapping[str, str]) -> PolzaError:
    error = body.get("error") if isinstance(body, dict) else None
    error = error if isinstance(error, dict) else {}
    top_trace = body.get("trace_id") if isinstance(body, dict) else None
    kwargs: dict[str, Any] = {
        "message": str(error.get("message") or f"Polza API returned HTTP {status}"),
        "code": error.get("code"),
        "trace_id": error.get("trace_id") or top_trace,
        "metadata": error.get("metadata"),
        "details": error.get("details"),
    }
    kwargs["retry_after"] = _retry_after(headers)
    if status == 429:
        return RateLimitError(status, **kwargs)
    if status in _BY_STATUS:
        return _BY_STATUS[status](status, **kwargs)
    if status >= 500:
        return ServerError(status, **kwargs)
    return RequestError(status, **kwargs)
