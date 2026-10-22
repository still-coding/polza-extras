from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ._models import Usage


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
        raw = self.metadata.get("raw")
        if raw and str(raw) not in self.message:
            text += f": {raw}"
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


class NotFoundError(RequestError):
    """404: unknown model or endpoint (a model id without the provider/ prefix is a 404 too)."""


class PolzaConnectionError(PolzaError):
    """The request never got an HTTP response (DNS, connect, timeout)."""

    def __init__(self, message: str) -> None:
        super().__init__(0, message)


class StreamError(PolzaError):
    """An error event inside a 200 stream (Polza sends `error` + `finish_reason: "error"`).

    Text received before it was already yielded; the request is not retried mid-stream.
    """


class EmptyResponseError(PolzaError):
    """A 2xx response without usable text: no choices, or empty content.

    `reasoning` carries the model's reasoning text, if any; with `finish_reason == "length"` it
    usually means `max_tokens` was spent on thinking. `usage` and `raw` (the response body) are
    kept because the request is still billed.
    """

    def __init__(
        self,
        message: str,
        finish_reason: str | None = None,
        reasoning: str | None = None,
        usage: Usage | None = None,
        raw: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(200, message)
        self.finish_reason = finish_reason
        self.reasoning = reasoning
        self.usage = usage
        self.raw = raw or {}


_BY_STATUS: dict[int, type[PolzaError]] = {
    401: AuthError,
    402: InsufficientFundsError,
    403: ForbiddenError,
    404: NotFoundError,
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
