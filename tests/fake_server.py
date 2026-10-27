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


def resp(
    status: int = 200,
    json: Any = None,
    headers: dict[str, str] | None = None,
    content: bytes | None = None,
) -> SimpleNamespace:
    """A response: `json` for an API answer, `content` for a raw file body."""
    return SimpleNamespace(
        status=status, json=json, headers=headers or {}, text=None, content=content
    )


def sse(*events: Any, done: bool = True) -> SimpleNamespace:
    """A 200 text/event-stream response with one `data:` line per event (dicts become JSON)."""
    import json as _json

    lines = [
        f"data: {e if isinstance(e, str) else _json.dumps(e, ensure_ascii=False)}\n\n"
        for e in events
    ]
    if done:
        lines.append("data: [DONE]\n\n")
    return SimpleNamespace(
        status=200,
        json=None,
        headers={"content-type": "text/event-stream"},
        text="".join(lines),
        content=None,
    )


class TrackedBody(_http.SyncByteStream, _http.AsyncByteStream):
    """A response body that remembers whether the client closed it (released the connection)."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.closed = False

    def __iter__(self) -> Any:
        yield self.data

    async def __aiter__(self) -> Any:
        yield self.data

    def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.closed = True


class Route:
    def __init__(self) -> None:
        self.calls: list[Any] = []
        self.bodies: list[TrackedBody] = []  # streamed bodies, to check they get closed
        self.side_effect: Any = None
        self._default = resp(200, {})

    def respond(
        self,
        status: int = 200,
        json: Any = None,
        headers: dict[str, str] | None = None,
        content: bytes | None = None,
    ) -> Route:
        self._default = resp(status, json, headers, content)
        self.side_effect = None
        return self

    def stream(self, *events: Any, done: bool = True) -> Route:
        self._default = sse(*events, done=done)
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
            return _http.Response(
                404, json={"error": {"code": "NOT_FOUND", "message": "no fake route"}}
            )
        route.calls.append(request)
        item = route.next()
        if isinstance(item, Exception):
            raise item
        if item.text is not None:
            body = TrackedBody(item.text.encode())
            route.bodies.append(body)
            return _http.Response(item.status, stream=body, headers=item.headers)
        if item.content is not None:
            return _http.Response(item.status, content=item.content, headers=item.headers)
        return _http.Response(item.status, json=item.json, headers=item.headers)

    def http_client(self) -> Any:
        return openai.DefaultHttpxClient(transport=_http.MockTransport(self._handle))

    def async_http_client(self) -> Any:
        return openai.DefaultAsyncHttpxClient(transport=_http.MockTransport(self._handle))
