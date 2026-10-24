import json
from decimal import Decimal

import pytest

from polza_extras import AuthError, EmptyResponseError, InsufficientFundsError, RateLimitError
from tests.conftest import MESSAGES, completion, error_body
from tests.fake_server import ConnectError, resp


def test_returns_text_usage_and_ruble_cost(polza, chat_route):
    chat_route.respond(200, json=completion("Здравствуй"))
    result = polza.chat(model="openai/gpt-4o", messages=MESSAGES)
    assert result.text == "Здравствуй"
    assert result.finish_reason == "stop"
    assert (
        result.usage.prompt_tokens,
        result.usage.completion_tokens,
        result.usage.total_tokens,
    ) == (10, 5, 15)
    assert result.cost_rub == Decimal("0.04131306")
    assert result.model == "openai/gpt-4o" and result.provider == "openai-direct"


def test_sends_auth_model_and_messages_to_the_polza_base_url(polza, chat_route):
    chat_route.respond(200, json=completion())
    polza.chat(
        model="anthropic/claude-sonnet-4.6", messages=MESSAGES, max_tokens=50, temperature=0.3
    )
    request = chat_route.last_request
    sent = json.loads(request.content)
    assert request.headers["authorization"] == "Bearer test-key"
    assert sent["model"] == "anthropic/claude-sonnet-4.6" and sent["messages"] == MESSAGES
    assert sent["max_tokens"] == 50 and sent["temperature"] == 0.3


def test_cost_falls_back_to_cost_alias_and_is_none_when_absent(polza, chat_route):
    chat_route.respond(
        200,
        json=completion(
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "cost": 1.5}
        ),
    )
    assert polza.chat(model="m", messages=MESSAGES).cost_rub == Decimal("1.5")
    chat_route.respond(
        200, json=completion(usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
    )
    assert polza.chat(model="m", messages=MESSAGES).cost_rub is None


def test_status_201_is_success(polza, chat_route):
    chat_route.respond(201, json=completion("ok"))
    assert polza.chat(model="m", messages=MESSAGES).text == "ok"


def test_empty_choices_raises_instead_of_returning_none(polza, chat_route):
    chat_route.respond(200, json=completion(choices=[]))
    with pytest.raises(EmptyResponseError):
        polza.chat(model="m", messages=MESSAGES)


def test_empty_content_with_length_finish_raises_and_carries_reasoning(polza, chat_route):
    chat_route.respond(200, json=completion("", reasoning="думаю…", finish="length"))
    with pytest.raises(EmptyResponseError) as info:
        polza.chat(model="m", messages=MESSAGES, max_tokens=10)
    assert info.value.finish_reason == "length" and info.value.reasoning == "думаю…"


def test_reasoning_is_exposed_and_can_be_used_as_text_on_request(polza, chat_route):
    chat_route.respond(200, json=completion("", reasoning="вот ответ"))
    with pytest.raises(EmptyResponseError):
        polza.chat(model="m", messages=MESSAGES)
    result = polza.chat(model="m", messages=MESSAGES, reasoning_fallback=True)
    assert result.text == "вот ответ" and result.reasoning == "вот ответ"
    chat_route.respond(200, json=completion("ответ", reasoning="мысли"))
    both = polza.chat(model="m", messages=MESSAGES)
    assert both.text == "ответ" and both.reasoning == "мысли"


def test_auth_error_is_typed_with_trace_id_and_not_retried(polza, chat_route, sleeps):
    chat_route.respond(401, json=error_body("UNAUTHORIZED", "bad key", "tr-9"))
    with pytest.raises(AuthError) as info:
        polza.chat(model="m", messages=MESSAGES)
    assert info.value.trace_id == "tr-9" and chat_route.call_count == 1 and sleeps == []


def test_502_is_retried_then_succeeds(polza, chat_route, sleeps):
    chat_route.side_effect = [
        resp(502, json=error_body("BAD_GATEWAY")),
        resp(200, json=completion("ок")),
    ]
    assert polza.chat(model="m", messages=MESSAGES).text == "ок"
    assert chat_route.call_count == 2 and len(sleeps) == 1


def test_429_own_limit_sleeps_retry_after(polza, chat_route, sleeps):
    chat_route.side_effect = [
        resp(429, json=error_body("TOO_MANY_REQUESTS"), headers={"Retry-After": "7"}),
        resp(200, json=completion("ок")),
    ]
    polza.chat(model="m", messages=MESSAGES)
    assert sleeps == [7.0]


def test_429_with_huge_retry_after_fails_fast(polza, chat_route, sleeps):
    chat_route.respond(429, json=error_body("TOO_MANY_REQUESTS"), headers={"Retry-After": "40000"})
    with pytest.raises(RateLimitError) as info:
        polza.chat(model="m", messages=MESSAGES)
    assert info.value.retry_after == 40000 and sleeps == []


def test_gives_up_after_max_retries_and_raises_last_error(polza, chat_route, sleeps):
    chat_route.respond(503, json=error_body("SERVICE_UNAVAILABLE"))
    with pytest.raises(Exception) as info:
        polza.chat(model="m", messages=MESSAGES)
    assert info.value.status == 503
    assert chat_route.call_count == 4 and len(sleeps) == 3  # 1 try + 3 retries


def test_402_insufficient_funds_raises_immediately(polza, chat_route):
    chat_route.respond(402, json=error_body("INSUFFICIENT_BALANCE", "no money"))
    with pytest.raises(InsufficientFundsError):
        polza.chat(model="m", messages=MESSAGES)
    assert chat_route.call_count == 1


def test_connection_failure_is_retried_then_raised_as_polza_error(polza, chat_route, sleeps):
    from polza_extras import PolzaConnectionError

    chat_route.side_effect = ConnectError("boom")
    with pytest.raises(PolzaConnectionError):
        polza.chat(model="m", messages=MESSAGES)
    assert len(sleeps) == 3


def test_json_flag_requests_json_object_format(polza, chat_route):
    chat_route.respond(200, json=completion('{"a": 1}'))
    polza.chat(model="m", messages=MESSAGES, json=True)
    assert json.loads(chat_route.last_request.content)["response_format"] == {"type": "json_object"}


def test_explicit_response_format_schema_is_passed_through(polza, chat_route):
    chat_route.respond(200, json=completion('{"name": "x"}'))
    fmt = {
        "type": "json_schema",
        "json_schema": {
            "name": "p",
            "strict": True,
            "schema": {"type": "object", "properties": {}},
        },
    }
    polza.chat(model="m", messages=MESSAGES, response_format=fmt)
    assert json.loads(chat_route.last_request.content)["response_format"] == fmt


def test_no_response_format_by_default(polza, chat_route):
    chat_route.respond(200, json=completion())
    polza.chat(model="m", messages=MESSAGES)
    assert "response_format" not in json.loads(chat_route.last_request.content)


def test_empty_content_from_hidden_thinking_keeps_usage_and_raw_response(polza, chat_route):
    data = completion(
        "",
        finish="length",
        usage={
            "prompt_tokens": 900,
            "completion_tokens": 1000,
            "total_tokens": 1900,
            "cost_rub": 0.3,
        },
    )
    data["choices"][0]["message"]["reasoning_details"] = [
        {"type": "reasoning.text", "signature": "sig"}
    ]
    chat_route.respond(200, json=data)
    with pytest.raises(EmptyResponseError) as info:
        polza.chat(model="m", messages=MESSAGES, max_tokens=1000)
    err = info.value
    assert err.usage is not None and err.usage.completion_tokens == 1000
    assert err.raw["choices"][0]["message"]["reasoning_details"]
    assert "thinking" in str(err)
    assert err.cost_rub == Decimal("0.3")  # the failed call is still billed


def tool_call_reply():
    data = completion(None, finish="tool_calls")
    data["choices"][0]["message"]["tool_calls"] = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Москва"}'},
        }
    ]
    return data


def test_tool_call_reply_without_text_returns_tool_calls(polza, chat_route):
    chat_route.respond(200, json=tool_call_reply())
    result = polza.chat(model="m", messages=MESSAGES, tools=[{"type": "function"}])
    assert result.text == "" and result.finish_reason == "tool_calls"
    [call] = result.tool_calls
    assert (call.id, call.name, call.arguments) == ("call_1", "get_weather", '{"city": "Москва"}')
    assert call.parsed_arguments() == {"city": "Москва"}


def test_plain_text_reply_has_no_tool_calls(polza, chat_route):
    chat_route.respond(200, json=completion("ok"))
    assert polza.chat(model="m", messages=MESSAGES).tool_calls == []


def test_tool_call_arguments_must_decode_to_an_object():
    from polza_extras import ToolCall

    assert ToolCall(id="c", name="f").parsed_arguments() == {}
    with pytest.raises(ValueError, match="not a JSON object"):
        ToolCall(id="c", name="f", arguments="[1]").parsed_arguments()


DETAILED_USAGE = {
    "prompt_tokens": 2000,
    "completion_tokens": 300,
    "total_tokens": 2300,
    "prompt_tokens_details": {"cached_tokens": 1500, "cache_write_tokens": 400},
    "completion_tokens_details": {"reasoning_tokens": 120},
    "cost_rub": 0.5,
}


def test_usage_reports_cached_cache_write_and_reasoning_tokens(polza, chat_route):
    chat_route.respond(200, json=completion(usage=DETAILED_USAGE))
    usage = polza.chat(model="m", messages=MESSAGES).usage
    assert (usage.cached_tokens, usage.cache_write_tokens, usage.reasoning_tokens) == (
        1500,
        400,
        120,
    )


def test_usage_details_default_to_zero_when_absent_or_null(polza, chat_route):
    usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    chat_route.respond(200, json=completion(usage=usage))
    plain = polza.chat(model="m", messages=MESSAGES).usage
    chat_route.respond(
        200,
        json=completion(
            usage=usage | {"prompt_tokens_details": None, "completion_tokens_details": None}
        ),
    )
    nulls = polza.chat(model="m", messages=MESSAGES).usage
    for u in (plain, nulls):
        assert (u.cached_tokens, u.cache_write_tokens, u.reasoning_tokens) == (0, 0, 0)
