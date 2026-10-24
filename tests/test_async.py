"""AsyncPolza mirrors Polza; these check each method once plus retries and stream errors."""

from decimal import Decimal

import pytest

from polza_extras import AsyncPolza, AuthError, EmptyResponseError, NotFoundError, StreamError
from tests.conftest import BASE, MESSAGES, completion, error_body
from tests.fake_server import ConnectError, resp, sse
from tests.test_account import HAIKU
from tests.test_chat import tool_call_reply
from tests.test_stream import USAGE, chunk, tool_chunk


async def test_chat_returns_text_and_cost(apolza, chat_route):
    chat_route.respond(200, json=completion("Привет!"))
    result = await apolza.chat("m", MESSAGES, max_tokens=10, json=True)
    assert result.text == "Привет!" and result.cost_rub == Decimal("0.04131306")


async def test_chat_retries_502_then_succeeds(apolza, chat_route, sleeps):
    chat_route.side_effect = [resp(502, error_body("BAD_GATEWAY")), resp(200, completion("ок"))]
    assert (await apolza.chat("m", MESSAGES)).text == "ок"
    assert len(sleeps) == 1


async def test_chat_raises_typed_errors(apolza, chat_route):
    chat_route.respond(401, json=error_body("UNAUTHORIZED", "bad key"))
    with pytest.raises(AuthError):
        await apolza.chat("m", MESSAGES)


async def test_connection_failure_is_retried(apolza, chat_route, sleeps):
    chat_route.side_effect = [ConnectError("down"), resp(200, completion("ок"))]
    assert (await apolza.chat("m", MESSAGES)).text == "ок"


async def test_empty_content_raises(apolza, chat_route):
    chat_route.respond(200, json=completion("", finish="length"))
    with pytest.raises(EmptyResponseError):
        await apolza.chat("m", MESSAGES)


async def test_stream_yields_deltas_and_final_cost(apolza, chat_route):
    chat_route.stream(chunk("При"), chunk("вет"), USAGE)
    events = [e async for e in apolza.chat_stream("m", MESSAGES)]
    assert "".join(e.delta for e in events) == "Привет" and events[-1].cost_rub == Decimal("0.012")


async def test_stream_error_event_raises(apolza, chat_route):
    chat_route.stream(
        chunk("Нач"), {"error": {"code": "PROVIDER_ERROR", "message": "died"}}, done=False
    )
    with pytest.raises(StreamError):
        _ = [e async for e in apolza.chat_stream("m", MESSAGES)]


async def test_stream_retries_before_the_first_chunk(apolza, chat_route, sleeps):
    chat_route.side_effect = [
        resp(503, error_body("UNAVAILABLE"), {"Retry-After": "1"}),
        sse(chunk("ок"), USAGE),
    ]
    assert "".join([e.delta async for e in apolza.chat_stream("m", MESSAGES)]) == "ок"
    assert sleeps == [1.0]


async def test_catalog_balance_and_key(apolza, server):
    server.get(f"{BASE}/models/catalog").respond(200, json={"data": [HAIKU], "meta": {"total": 1}})
    server.get(f"{BASE}/models/{HAIKU['id']}").respond(200, json=HAIKU)
    server.get("https://polza.ai/api/v2/balance").respond(
        200,
        json={
            "amount": "10",
            "available": "9",
            "reservedAmount": "1",
            "spentAmount": "5",
            "updatedAt": None,
        },
    )
    server.get(f"{BASE}/key").respond(200, json={"label": "k", "usage": 1.5})
    assert [m.id for m in (await apolza.models(search="claude")).items] == [HAIKU["id"]]
    assert (await apolza.model(HAIKU["id"])).pricing.prompt_per_million == Decimal("11.70364000")
    assert (await apolza.balance()).available == Decimal(9)
    assert (await apolza.key_info()).usage == Decimal("1.5")


async def test_unknown_model_is_not_found(apolza, server):
    server.get(f"{BASE}/models/x/y").respond(404, json=error_body("NOT_FOUND", "nope"))
    with pytest.raises(NotFoundError):
        await apolza.model("x/y")


def test_missing_key_is_rejected(monkeypatch):
    monkeypatch.delenv("POLZA_AI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="POLZA_AI_API_KEY"):
        AsyncPolza()


async def test_async_stream_abandoned_early_releases_its_connection(apolza, chat_route):
    chat_route.stream(chunk("раз"), chunk("два"), USAGE)
    events = apolza.chat_stream("m", MESSAGES)
    await anext(events)
    await events.aclose()
    assert chat_route.bodies[-1].closed


async def test_async_client_is_a_context_manager_that_closes_its_connections(server):
    async with AsyncPolza(api_key="k", http_client=server.async_http_client()) as polza:
        assert not polza.http.is_closed
    assert polza.http.is_closed


async def test_chat_returns_tool_calls_without_text(apolza, chat_route):
    chat_route.respond(200, json=tool_call_reply())
    result = await apolza.chat("m", MESSAGES, tools=[{"type": "function"}])
    assert result.text == "" and result.tool_calls[0].parsed_arguments() == {"city": "Москва"}


async def test_stream_passes_tool_call_deltas_through(apolza, chat_route):
    chat_route.stream(tool_chunk(0, '{"a"', "call_1", "f"), tool_chunk(0, ": 1}"), USAGE)
    events = [e async for e in apolza.chat_stream("m", MESSAGES)]
    assert events[0].tool_calls[0].name == "f"
    assert "".join(d.arguments for e in events for d in e.tool_calls) == '{"a": 1}'
