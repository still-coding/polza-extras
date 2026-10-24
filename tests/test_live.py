"""Smoke tests against the real Polza API. Costs < 0.1 ₽.

Run: `POLZA_AI_API_KEY=... uv run pytest -m live --no-cov`
"""

import os
from decimal import Decimal

import pytest

from polza_extras import AuthError, NotFoundError, Polza

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("POLZA_AI_API_KEY"), reason="POLZA_AI_API_KEY not set"),
]
CHEAP = os.environ.get("POLZA_LIVE_MODEL", "anthropic/claude-haiku-5.5")


@pytest.fixture(scope="module")
def polza():
    return Polza()


def test_chat_returns_text_and_ruble_cost(polza):
    result = polza.chat(
        CHEAP,
        [{"role": "user", "content": "Ответь одним словом: привет"}],
        max_tokens=50,
        reasoning={"type": "disabled"},
    )
    assert result.text.strip()
    assert result.cost_rub is not None and result.cost_rub > 0


def test_json_mode_returns_parseable_json(polza):
    import json

    result = polza.chat(
        CHEAP,
        [
            {"role": "system", "content": 'Верни JSON {"ok": true}'},
            {"role": "user", "content": "давай"},
        ],
        max_tokens=50,
        json=True,
        reasoning={"type": "disabled"},
    )
    text = result.text.strip().removeprefix("```json").removesuffix("```")
    assert json.loads(text)["ok"] is True


def test_catalog_lists_and_details_a_model_with_pricing(polza):
    page = polza.models(search="claude", type="chat", limit=5)
    assert page.items and page.total
    model = polza.model(CHEAP)
    assert model.pricing is not None and model.pricing.prompt_per_million > 0


def test_unknown_model_is_not_found(polza):
    with pytest.raises(NotFoundError):
        polza.model("anthropic/definitely-not-a-model")


def test_balance_and_key_info(polza):
    assert polza.balance().available >= Decimal(0)
    assert polza.key_info().usage is not None


def test_bad_key_raises_auth_error_with_trace_id():
    with pytest.raises(AuthError) as info:
        Polza(api_key="bad-key").balance()
    assert info.value.trace_id


def test_stream_returns_text_and_final_ruble_cost(polza):
    events = list(
        polza.chat_stream(
            CHEAP,
            [{"role": "user", "content": "Посчитай от одного до пяти словами"}],
            max_tokens=60,
            reasoning={"type": "disabled"},
        )
    )
    assert "".join(e.delta for e in events).strip()
    assert events[-1].cost_rub is not None and events[-1].cost_rub > 0


async def test_async_chat_and_stream():
    from polza_extras import AsyncPolza

    apolza = AsyncPolza()
    result = await apolza.chat(
        CHEAP,
        [{"role": "user", "content": "Скажи «ок»"}],
        max_tokens=20,
        reasoning={"type": "disabled"},
    )
    assert result.text.strip() and result.cost_rub
    events = [
        e
        async for e in apolza.chat_stream(
            CHEAP,
            [{"role": "user", "content": "Скажи «ок»"}],
            max_tokens=20,
            reasoning={"type": "disabled"},
        )
    ]
    assert events[-1].cost_rub


def test_tool_call_reply_returns_parsed_tool_calls(polza):
    weather = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Weather in a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
    result = polza.chat(
        CHEAP,
        [{"role": "user", "content": "Какая погода в Москве?"}],
        max_tokens=100,
        tools=[weather],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
        reasoning={"type": "disabled"},
    )
    [call] = result.tool_calls
    assert call.name == "get_weather" and "city" in call.parsed_arguments()
