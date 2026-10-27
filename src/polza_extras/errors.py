from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

from ._json import obj

if TYPE_CHECKING:
    from decimal import Decimal

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
    """429. `scope` is "own" for Polza's per-organization limit (has Retry-After), or "provider"."""

    @property
    def scope(self) -> Literal["own", "provider"]:
        return "own" if self.retry_after is not None else "provider"


class ServerError(PolzaError):
    """408, 5xx: provider timeout, bad gateway, unavailable."""


class RequestError(PolzaError):
    """Other 4xx: bad request, not found, payload too large."""


class NotFoundError(RequestError):
    """404: unknown model or endpoint (a model id without the provider/ prefix is a 404 too)."""


class MediaFailedError(PolzaError):
    """An image job that ended `failed` or `cancelled`, or finished without a picture.

    `id` is the job id, `code` Polza's error code (`BAD_GATEWAY`, `guard_blocked`, ...) and
    `provider` the upstream's name. `retryable` is True for transient codes (timeout, bad gateway,
    unavailable, rate limit): a new job may work. The library never resends a job itself, since
    Polza does not say whether a failed job is billed.
    """

    def __init__(
        self,
        id: str,
        code: str,
        message: str,
        metadata: dict[str, Any] | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(200, message, code=code, metadata=metadata)
        self.id = id
        self.retryable = retryable

    @property
    def provider(self) -> str | None:
        name = self.metadata.get("provider_name")
        return str(name) if name else None


class MediaModeratedError(MediaFailedError):
    """The job was blocked by moderation (`guard_blocked`) or has no route (`guard_no_route`).
    Never retry the same prompt."""


class MediaTimeoutError(PolzaError):
    """The job was still pending or processing when the wait ran out.

    The job keeps running (and may be billed): `wait_image(error.id)` picks it up again.
    """

    def __init__(self, id: str, status: str, waited: float) -> None:
        super().__init__(
            0, f"image job {id} still {status} after {waited:.0f} s", code="REQUEST_TIMEOUT"
        )
        self.id = id
        self.status = status
        self.waited = waited


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
        cost_rub: Decimal | None = None,
    ) -> None:
        super().__init__(200, message)
        self.finish_reason = finish_reason
        self.reasoning = reasoning
        self.usage = usage
        self.raw = raw or {}
        self.cost_rub = cost_rub  # what the failed call was billed


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
    error = obj(obj(body).get("error"))
    top_trace = obj(body).get("trace_id")
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
