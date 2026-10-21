from __future__ import annotations

import os
import random
import time
from collections.abc import Callable
from typing import Any, TypeVar

import openai

from ._models import ChatResult
from .chat import parse_chat
from .errors import PolzaConnectionError, PolzaError, error_from_response
from .retry import RetryPolicy

DEFAULT_BASE_URL = "https://polza.ai/api/v1"
API_KEY_ENV = "POLZA_AI_API_KEY"

T = TypeVar("T")


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
        key = api_key or os.environ.get(API_KEY_ENV)
        if not key:
            raise ValueError(f"Polza API key missing: pass api_key or set {API_KEY_ENV}")
        self.base_url = base_url
        self.retry = retry or RetryPolicy()
        self._sleep = sleep
        self._rng = random.Random()
        # Use the SDK's own client flavor (httpx or its vendored httpx2, depending on the openai
        # version) so errors are translated correctly; the Polza-only endpoints reuse it.
        self.http: Any = http_client or openai.DefaultHttpxClient(timeout=timeout)
        # SDK retries are off: Polza has its own retry rules (see RetryPolicy).
        self.openai = openai.OpenAI(base_url=base_url, api_key=key, max_retries=0, http_client=self.http)

    def _with_retries(self, call: Callable[[], T]) -> T:
        attempt = 0
        while True:
            try:
                return call()
            except openai.APIStatusError as exc:
                try:
                    body = exc.response.json()
                except ValueError:
                    body = None
                error: PolzaError = error_from_response(exc.status_code, body, dict(exc.response.headers))
            except openai.APIConnectionError as exc:
                error = PolzaConnectionError(str(exc))
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
        params: dict[str, Any] = {"model": model, "messages": messages}
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if temperature is not None:
            params["temperature"] = temperature
        if json:
            extra.setdefault("response_format", {"type": "json_object"})
        if extra:
            params["extra_body"] = extra

        def call() -> dict[str, Any]:
            raw = self.openai.chat.completions.with_raw_response.create(**params)
            data: dict[str, Any] = raw.http_response.json()
            return data

        return parse_chat(self._with_retries(call), reasoning_fallback)
