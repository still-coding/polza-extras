"""Tiny in-process fake of the Polza HTTP API.

The OpenAI SDK picks its own httpx flavor (httpx, or a vendored httpx2 in newer releases), so we
build the mock transport from whichever module the SDK's default client comes from.
"""
from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import openai

_http = importlib.import_module(openai.DefaultHttpxClient.__mro__[1].__module__.split(".")[0])
ConnectError = _http.ConnectError


def resp(status: int = 200, json: Any = None, headers: dict[str, str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(status=status, json=json, headers=headers or {})


class Route:
    def __init__(self) -> None:
        self.calls: list[Any] = []
        self.side_effect: Any = None
        self._default = resp(200, {})

    def respond(self, status: int = 200, json: Any = None, headers: dict[str, str] | None = None) -> Route:
        self._default = resp(status, json, headers)
        self.side_effect = None
        return self

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def last_request(self) -> Any:
        return self.calls[-1]

    def next(self) -> Any:
        if isinstance(self.side_effect, list) and self.side_effect:
            return self.side_effect.pop(0)
        if isinstance(self.side_effect, Exception):
            return self.side_effect
        return self._default


class FakeServer:
    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], Route] = {}

    def route(self, method: str, url: str) -> Route:
        return self.routes.setdefault((method, url), Route())

    def post(self, url: str) -> Route:
        return self.route("POST", url)

    def get(self, url: str) -> Route:
        return self.route("GET", url)

    def _handle(self, request: Any) -> Any:
        route = self.routes.get((request.method, str(request.url).split("?")[0]))
        if route is None:
            return _http.Response(404, json={"error": {"code": "NOT_FOUND", "message": "no fake route"}})
        route.calls.append(request)
        item = route.next()
        if isinstance(item, Exception):
            raise item
        return _http.Response(item.status, json=item.json, headers=item.headers)

    def http_client(self) -> Any:
        return openai.DefaultHttpxClient(transport=_http.MockTransport(self._handle))
