"""Image generation over `POST /v1/media`: the I/O-free pieces shared by Polza and AsyncPolza.

A job is started with `async: true` and polled with `GET /v1/media/{id}` until it is `completed`,
`failed` or `cancelled` (Polza advises 3-5 s between polls for images). Results live on Polza for
7 days, so `generate_image` downloads the file by default.
"""

from __future__ import annotations

import base64
import importlib
from decimal import Decimal
from pathlib import Path
from typing import Any

import openai

from ._json import arr, obj
from ._models import ImageResult
from .errors import MediaFailedError, MediaModeratedError

MAX_REFERENCES = 8
MODERATION_CODES = frozenset({"guard_blocked", "guard_no_route"})
RETRYABLE_CODES = frozenset(
    {"REQUEST_TIMEOUT", "TOO_MANY_REQUESTS", "BAD_GATEWAY", "SERVICE_UNAVAILABLE", "INTERNAL_ERROR"}
)
PENDING = frozenset({"pending", "processing"})

# The SDK picks its own httpx flavor (httpx, or a vendored httpx2); download errors come from it.
_http: Any = importlib.import_module(openai.DefaultHttpxClient.__mro__[1].__module__.split(".")[0])
TRANSPORT_ERROR: type[Exception] = _http.HTTPError


def reference(image: bytes | str | Path) -> dict[str, str]:
    """A reference image for the request: bytes or a file path are sent inline, a URL as is."""
    if isinstance(image, str) and image.startswith(("http://", "https://")):
        return {"type": "url", "url": image}
    data = Path(image).read_bytes() if isinstance(image, str | Path) else image
    return {"type": "base64", "data": base64.b64encode(data).decode()}


def request_body(
    model: str,
    prompt: str,
    aspect_ratio: str | None,
    images: list[bytes | str | Path],
    seed: int | None,
    image_resolution: str | None,
    output_format: str | None,
    extra: dict[str, Any],
) -> dict[str, Any]:
    if len(images) > MAX_REFERENCES:
        raise ValueError(f"at most {MAX_REFERENCES} reference images, got {len(images)}")
    fields: dict[str, Any] = {
        "aspect_ratio": aspect_ratio,
        "seed": seed,
        "image_resolution": image_resolution,
        "output_format": output_format,
        "images": [reference(i) for i in images] or None,
    }
    given = {k: v for k, v in fields.items() if v is not None}
    return {"model": model, "async": True, "input": {"prompt": prompt, **given, **extra}}


def sniff_mime(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG"):
        return "image/png"
    if data.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def with_file(result: ImageResult, data: bytes) -> ImageResult:
    return result.model_copy(update={"data": data, "mime": sniff_mime(data)})


def status_of(job: Any) -> str:
    return str(obj(job).get("status") or "pending")


def settle(job: Any, model: str | None) -> ImageResult | None:
    """The result of a finished job, None while it is pending or processing.

    Raises MediaFailedError (MediaModeratedError for a moderation block) for a failed or cancelled
    job, and for a completed one without a picture."""
    body = obj(job)
    job_id = str(body.get("id") or "")
    status = status_of(body)
    if status in PENDING:
        return None
    if status != "completed":
        error = obj(body.get("error"))
        code = str(error.get("code") or status)
        failed = MediaModeratedError if code in MODERATION_CODES else MediaFailedError
        raise failed(
            job_id,
            code,
            str(error.get("message") or f"image job {status}"),
            metadata=obj(error.get("metadata")),
            retryable=code in RETRYABLE_CODES,
        )
    data = body.get("data")
    items = arr(data) if isinstance(data, list) else [data]
    item = obj(items[0] if items else None)
    inline = item.get("b64_json") or item.get("base64")
    url = item.get("url")
    if not inline and not url:
        raise MediaFailedError(job_id, "no_image", "the job completed with no image in its data")
    cost = obj(body.get("usage")).get("cost_rub")
    result = ImageResult(
        id=job_id,
        model=model or str(body.get("model") or ""),
        url=None if inline else str(url),
        cost_rub=Decimal(str(cost)) if cost is not None else None,
        raw=body,
    )
    return with_file(result, base64.b64decode(inline)) if inline else result
