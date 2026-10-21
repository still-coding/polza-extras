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
    assert (result.usage.prompt_tokens, result.usage.completion_tokens, result.usage.total_tokens) == (10, 5, 15)
    assert result.cost_rub == Decimal("0.04131306")
    assert result.model == "openai/gpt-4o" and result.provider == "openai-direct"


def test_sends_auth_model_and_messages_to_the_polza_base_url(polza, chat_route):
    chat_route.respond(200, json=completion())
    polza.chat(model="anthropic/claude-sonnet-4.6", messages=MESSAGES, max_tokens=50, temperature=0.3)
    request = chat_route.last_request
    sent = json.loads(request.content)
    assert request.headers["authorization"] == "Bearer test-key"
    assert sent["model"] == "anthropic/claude-sonnet-4.6" and sent["messages"] == MESSAGES
    assert sent["max_tokens"] == 50 and sent["temperature"] == 0.3


def test_cost_falls_back_to_cost_alias_and_is_none_when_absent(polza, chat_route):
    chat_route.respond(200, json=completion(usage={"prompt_tokens": 1, "completion_tokens": 1,
                                                   "total_tokens": 2, "cost": 1.5}))
    assert polza.chat(model="m", messages=MESSAGES).cost_rub == Decimal("1.5")
    chat_route.respond(200, json=completion(usage={"prompt_tokens": 1, "completion_tokens": 1,
                                                   "total_tokens": 2}))
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
    fmt = {"type": "json_schema", "json_schema": {"name": "p", "strict": True,
                                                  "schema": {"type": "object", "properties": {}}}}
    polza.chat(model="m", messages=MESSAGES, response_format=fmt)
    assert json.loads(chat_route.last_request.content)["response_format"] == fmt


def test_no_response_format_by_default(polza, chat_route):
    chat_route.respond(200, json=completion())
    polza.chat(model="m", messages=MESSAGES)
    assert "response_format" not in json.loads(chat_route.last_request.content)
