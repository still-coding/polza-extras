import json
from decimal import Decimal

import pytest

from polza_extras import AuthError, StreamError
from tests.conftest import MESSAGES, error_body
from tests.fake_server import ConnectError, resp, sse


def chunk(content=None, reasoning=None, finish=None):
    delta = {"role": "assistant"}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning"] = reasoning
    return {
        "id": "gen_1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "anthropic/claude-haiku-5.5",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


USAGE = {
    "id": "gen_1",
    "object": "chat.completion.chunk",
    "created": 1,
    "model": "anthropic/claude-haiku-5.5",
    "choices": [],
    "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13, "cost_rub": 0.012},
}


def test_stream_yields_text_deltas_then_final_usage_and_cost(polza, chat_route):
    chat_route.stream(chunk("При"), chunk("вет"), chunk(finish="stop"), USAGE)
    events = list(polza.chat_stream("m", MESSAGES))
    assert "".join(e.delta for e in events) == "Привет"
    assert events[-2].finish_reason == "stop"
    last = events[-1]
    assert last.usage.completion_tokens == 3 and last.cost_rub == Decimal("0.012")


def test_stream_requests_usage_and_sends_extras(polza, chat_route):
    chat_route.stream(chunk("ок"), USAGE)
    list(polza.chat_stream("m", MESSAGES, max_tokens=50, reasoning={"type": "disabled"}))
    body = json.loads(chat_route.last_request.content)
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert body["max_tokens"] == 50 and body["reasoning"] == {"type": "disabled"}


def test_stream_exposes_reasoning_deltas_separately(polza, chat_route):
    chat_route.stream(chunk(reasoning="думаю"), chunk("ответ"), USAGE)
    events = list(polza.chat_stream("m", MESSAGES))
    assert events[0].reasoning == "думаю" and events[0].delta == ""
    assert events[1].delta == "ответ"


def test_error_inside_the_stream_raises_stream_error_after_partial_text(polza, chat_route):
    chat_route.stream(
        chunk("Нач"),
        {
            "error": {"code": "PROVIDER_ERROR", "message": "upstream died", "trace_id": "t-9"},
            "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
        },
        done=False,
    )
    received = []
    with pytest.raises(StreamError) as info:
        for event in polza.chat_stream("m", MESSAGES):
            received.append(event.delta)
    assert received == ["Нач"]
    assert info.value.code == "PROVIDER_ERROR" and info.value.trace_id == "t-9"


def test_status_error_before_streaming_is_typed_and_retried_like_chat(polza, chat_route, sleeps):
    chat_route.side_effect = [resp(502, error_body("BAD_GATEWAY")), sse(chunk("ок"), USAGE)]
    assert "".join(e.delta for e in polza.chat_stream("m", MESSAGES)) == "ок"
    assert len(sleeps) == 1


def test_auth_error_before_streaming(polza, chat_route):
    chat_route.respond(401, json=error_body("UNAUTHORIZED", "bad key"))
    with pytest.raises(AuthError):
        list(polza.chat_stream("m", MESSAGES))


def test_connection_error_before_streaming_is_retried(polza, chat_route, sleeps):
    chat_route.side_effect = [ConnectError("down"), sse(chunk("ок"), USAGE)]
    assert "".join(e.delta for e in polza.chat_stream("m", MESSAGES)) == "ок"


def test_a_stream_abandoned_early_releases_its_connection(polza, chat_route):
    chat_route.stream(chunk("раз"), chunk("два"), USAGE)
    events = polza.chat_stream("m", MESSAGES)
    next(events)
    events.close()  # the caller stops reading, e.g. `break` out of the loop
    assert chat_route.bodies[-1].closed


def test_client_is_a_context_manager_that_closes_its_connections(server):
    from polza_extras import Polza

    with Polza(api_key="k", http_client=server.http_client()) as polza:
        assert not polza.http.is_closed
    assert polza.http.is_closed


def test_every_name_in_all_is_importable():
    import polza_extras

    assert [n for n in polza_extras.__all__ if not hasattr(polza_extras, n)] == []


def tool_chunk(index, arguments, call_id=None, name=None):
    call = {"index": index, "function": {"arguments": arguments}}
    if call_id is not None:
        call |= {"id": call_id, "type": "function"}
        call["function"]["name"] = name
    data = chunk()
    data["choices"][0]["delta"]["tool_calls"] = [call]
    return data


def test_stream_passes_tool_call_deltas_through(polza, chat_route):
    chat_route.stream(
        tool_chunk(0, "", "call_1", "get_weather"),
        tool_chunk(0, '{"city": '),
        tool_chunk(0, '"Москва"}'),
        chunk(finish="tool_calls"),
        USAGE,
    )
    events = list(polza.chat_stream("m", MESSAGES, tools=[{"type": "function"}]))
    first = events[0].tool_calls[0]
    assert (first.index, first.id, first.name, first.arguments) == (0, "call_1", "get_weather", "")
    later = events[1].tool_calls[0]
    assert (later.index, later.id, later.name) == (0, None, None)
    assert "".join(e.tool_calls[0].arguments for e in events[:3]) == '{"city": "Москва"}'
    assert events[3].tool_calls == [] and events[3].finish_reason == "tool_calls"


def test_stream_final_usage_has_cache_and_reasoning_details(polza, chat_route):
    from tests.test_chat import DETAILED_USAGE

    chat_route.stream(chunk("ок"), USAGE | {"usage": DETAILED_USAGE})
    usage = list(polza.chat_stream("m", MESSAGES))[-1].usage
    assert (usage.cached_tokens, usage.cache_write_tokens, usage.reasoning_tokens) == (
        1500,
        400,
        120,
    )


def test_stream_sends_json_schema_response_format(polza, chat_route):
    from tests.test_chat import PERSON

    chat_route.stream(chunk('{"name": "Аня"}'), USAGE)
    list(polza.chat_stream("m", MESSAGES, json_schema=PERSON))
    sent = json.loads(chat_route.last_request.content)["response_format"]
    assert sent["type"] == "json_schema" and sent["json_schema"]["schema"] == PERSON
