from __future__ import annotations

import asyncio
import os
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any, Self, TypeVar

import openai

from ._models import Balance, ChatResult, KeyInfo, ModelInfo, ModelPage, StreamEvent
from .chat import parse_chat, parse_stream_chunk
from .errors import PolzaConnectionError, PolzaError, StreamError, error_from_response
from .retry import RetryPolicy

DEFAULT_BASE_URL = "https://polza.ai/api/v1"
API_KEY_ENV = "POLZA_AI_API_KEY"

T = TypeVar("T")


# --- shared, I/O-free pieces of Polza and AsyncPolza ---------------------------------------------

def _api_key(api_key: str | None) -> str:
    key = api_key or os.environ.get(API_KEY_ENV)
    if not key:
        raise ValueError(f"Polza API key missing: pass api_key or set {API_KEY_ENV}")
    return key


def _translate(exc: openai.APIStatusError | openai.APIConnectionError) -> PolzaError:
    """The PolzaError for an openai SDK request error."""
    if isinstance(exc, openai.APIConnectionError):
        return PolzaConnectionError(str(exc))
    try:
        body = exc.response.json()
    except ValueError:
        body = None
    return error_from_response(exc.status_code, body, dict(exc.response.headers))


def _stream_error(exc: openai.APIError) -> StreamError:
    body = exc.body if isinstance(exc.body, dict) else {}
    return StreamError(200, str(body.get("message") or exc.message), code=body.get("code"),
                       trace_id=body.get("trace_id"), metadata=body.get("metadata"))


def _chat_params(
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int | None,
    temperature: float | None,
    json: bool,
    extra: dict[str, Any],
    stream: bool = False,
) -> dict[str, Any]:
    params: dict[str, Any] = {"model": model, "messages": messages}
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    if temperature is not None:
        params["temperature"] = temperature
    if json:
        extra.setdefault("response_format", {"type": "json_object"})
    if extra:
        params["extra_body"] = extra
    if stream:
        params["stream"] = True
        params["stream_options"] = {"include_usage": True}
    return params


def _get_options(params: dict[str, Any] | None, language: str | None) -> Any:  # openai RequestOptions
    headers = {"accept-language": language} if language else {}
    return {"params": {k: v for k, v in (params or {}).items() if v is not None}, "headers": headers}


def _catalog_query(search: str | None, type: str | None, providers: list[str] | None, page: int,
                   limit: int | None, filters: dict[str, Any]) -> dict[str, Any]:
    return {"search": search, "type": type, "providers": ",".join(providers) if providers else None,
            "page": page, "limit": limit, **filters}


def _parse_page(data: Any, page: int, limit: int | None) -> ModelPage:
    meta = data.get("meta") or {}
    return ModelPage(
        items=[ModelInfo.from_api(m) for m in data.get("data") or []],
        page=meta.get("page", page), limit=meta.get("limit", limit),
        total=meta.get("total"), total_pages=meta.get("totalPages"),
    )


def _balance_url(base_url: str) -> str:
    return base_url.rstrip("/").removesuffix("/v1") + "/v2/balance"


def _parse_balance(data: Any) -> Balance:
    return Balance(amount=data["amount"], available=data["available"], reserved=data["reservedAmount"],
                   spent=data["spentAmount"], updated_at=data.get("updatedAt"))


class Polza:
    """Polza.ai facade: an OpenAI SDK client plus Polza-specific cost, errors and retries."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        retry: RetryPolicy | None = None,
        timeout: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
        http_client: Any | None = None,
    ) -> None:
        self.base_url = base_url
        self.retry = retry or RetryPolicy()
        self._sleep = sleep
        self._rng = random.Random()
        # Use the SDK's own client flavor (httpx or its vendored httpx2, depending on the openai
        # version) so errors are translated correctly; the Polza-only endpoints reuse it.
        self.http: Any = http_client or openai.DefaultHttpxClient(timeout=timeout)
        # SDK retries are off: Polza has its own retry rules (see RetryPolicy).
        self.openai = openai.OpenAI(base_url=base_url, api_key=_api_key(api_key), max_retries=0,
                                    http_client=self.http)

    def close(self) -> None:
        """Close the HTTP connections (also the http_client passed in)."""
        self.openai.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _with_retries(self, call: Callable[[], T]) -> T:
        attempt = 0
        while True:
            try:
                return call()
            except (openai.APIStatusError, openai.APIConnectionError) as exc:
                error = _translate(exc)
            delay = self.retry.delay_for(error, attempt, self._rng)
            if delay is None:
                raise error
            self._sleep(delay)
            attempt += 1

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        reasoning_fallback: bool = False,
        json: bool = False,
        **extra: Any,
    ) -> ChatResult:
        params = _chat_params(model, messages, max_tokens, temperature, json, extra)

        def call() -> dict[str, Any]:
            raw = self.openai.chat.completions.with_raw_response.create(**params)
            data: dict[str, Any] = raw.http_response.json()
            return data

        return parse_chat(self._with_retries(call), reasoning_fallback)

    def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        json: bool = False,
        **extra: Any,
    ) -> Iterator[StreamEvent]:
        """Stream a chat completion. The last event carries `usage` and `cost_rub`.

        Errors before the first chunk are retried like `chat()`; an error event inside the stream
        raises StreamError (text already yielded stays yielded).
        """
        params = _chat_params(model, messages, max_tokens, temperature, json, extra, stream=True)
        stream = self._with_retries(lambda: self.openai.chat.completions.create(**params))
        try:
            for chunk in stream:
                yield parse_stream_chunk(chunk.to_dict())
        except openai.APIError as exc:
            raise _stream_error(exc) from exc
        finally:
            stream.close()  # also when the caller stops reading early

    # --- Polza-only endpoints ------------------------------------------------

    def _get(self, path: str, params: dict[str, Any] | None = None, language: str | None = None) -> Any:
        """GET a Polza endpoint through the SDK client, so errors and retries match chat()."""
        options = _get_options(params, language)
        return self._with_retries(lambda: self.openai.get(path, cast_to=object, options=options))

    def models(
        self,
        search: str | None = None,
        type: str | None = None,
        providers: list[str] | None = None,
        page: int = 1,
        limit: int | None = None,
        language: str = "ru",
        **filters: Any,
    ) -> ModelPage:
        """One page of /models/catalog. Extra filters (e.g. sortBy, contextLengthMin) pass through as is."""
        data = self._get("/models/catalog", _catalog_query(search, type, providers, page, limit, filters), language)
        return _parse_page(data, page, limit)

    def model(self, model_id: str, language: str = "ru") -> ModelInfo:
        """Details of one model; raises NotFoundError for an unknown id."""
        data = self._get(f"/models/{model_id}", language=language)
        return ModelInfo.from_api(data.get("data", data))

    def balance(self) -> Balance:
        """Organization balance in rubles (GET /api/v2/balance)."""
        return _parse_balance(self._get(_balance_url(self.base_url)))

    def key_info(self) -> KeyInfo:
        """Limits and usage of the current API key (GET /v1/key)."""
        data = self._get("/key")
        return KeyInfo.model_validate(data.get("data", data))


class AsyncPolza:
    """Async mirror of Polza on openai.AsyncOpenAI; same methods, awaited."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        retry: RetryPolicy | None = None,
        timeout: float = 60.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        http_client: Any | None = None,
    ) -> None:
        self.base_url = base_url
        self.retry = retry or RetryPolicy()
        self._sleep = sleep
        self._rng = random.Random()
        self.http: Any = http_client or openai.DefaultAsyncHttpxClient(timeout=timeout)
        self.openai = openai.AsyncOpenAI(base_url=base_url, api_key=_api_key(api_key), max_retries=0,
                                         http_client=self.http)

    async def close(self) -> None:
        await self.openai.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def _with_retries(self, call: Callable[[], Awaitable[T]]) -> T:
        attempt = 0
        while True:
            try:
                return await call()
            except (openai.APIStatusError, openai.APIConnectionError) as exc:
                error = _translate(exc)
            delay = self.retry.delay_for(error, attempt, self._rng)
            if delay is None:
                raise error
            await self._sleep(delay)
            attempt += 1

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        reasoning_fallback: bool = False,
        json: bool = False,
        **extra: Any,
    ) -> ChatResult:
        params = _chat_params(model, messages, max_tokens, temperature, json, extra)

        async def call() -> dict[str, Any]:
            raw = await self.openai.chat.completions.with_raw_response.create(**params)
            data: dict[str, Any] = raw.http_response.json()
            return data

        return parse_chat(await self._with_retries(call), reasoning_fallback)

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        json: bool = False,
        **extra: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Async version of Polza.chat_stream: `async for event in polza.chat_stream(...)`."""
        params = _chat_params(model, messages, max_tokens, temperature, json, extra, stream=True)

        async def call() -> Any:
            return await self.openai.chat.completions.create(**params)

        stream = await self._with_retries(call)
        try:
            async for chunk in stream:
                yield parse_stream_chunk(chunk.to_dict())
        except openai.APIError as exc:
            raise _stream_error(exc) from exc
        finally:
            await stream.close()

    async def _get(self, path: str, params: dict[str, Any] | None = None, language: str | None = None) -> Any:
        options = _get_options(params, language)

        async def call() -> Any:
            return await self.openai.get(path, cast_to=object, options=options)

        return await self._with_retries(call)

    async def models(
        self,
        search: str | None = None,
        type: str | None = None,
        providers: list[str] | None = None,
        page: int = 1,
        limit: int | None = None,
        language: str = "ru",
        **filters: Any,
    ) -> ModelPage:
        query = _catalog_query(search, type, providers, page, limit, filters)
        return _parse_page(await self._get("/models/catalog", query, language), page, limit)

    async def model(self, model_id: str, language: str = "ru") -> ModelInfo:
        data = await self._get(f"/models/{model_id}", language=language)
        return ModelInfo.from_api(data.get("data", data))

    async def balance(self) -> Balance:
        return _parse_balance(await self._get(_balance_url(self.base_url)))

    async def key_info(self) -> KeyInfo:
        data = await self._get("/key")
        return KeyInfo.model_validate(data.get("data", data))
