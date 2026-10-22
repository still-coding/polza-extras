from decimal import Decimal

import pytest

from polza_extras import AuthError, NotFoundError
from tests.conftest import BASE, error_body

HAIKU = {
    "id": "anthropic/claude-haiku-5.5", "name": "Anthropic: Claude Haiku 5.5", "type": "chat",
    "context_length": 1000000, "max_completion_tokens": 128000,
    "reasoning": {"mandatory": False, "supported_efforts": ["low", "none"], "default_effort": "medium"},
    "top_provider": {
        "pricing": {"prompt_per_million": "11.70364000", "completion_per_million": "58.51820000",
                    "input_cache_read_per_million": "1.17036400", "currency": "RUB"},
        "supported_parameters": ["reasoning", "max_tokens", "response_format"],
    },
}
SONNET = {**HAIKU, "id": "anthropic/claude-sonnet-5.5",
          "reasoning": {"mandatory": True, "supported_efforts": ["low", "high"], "default_effort": "high"}}


def test_models_lists_catalog_page_with_filters_and_language(polza, server):
    route = server.get(f"{BASE}/models/catalog").respond(200, json={
        "data": [HAIKU, SONNET], "meta": {"page": 1, "limit": 2, "total": 14, "totalPages": 7}})
    page = polza.models(search="claude", type="chat", providers=["anthropic"], limit=2)
    assert [m.id for m in page.items] == ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"]
    assert (page.page, page.total, page.total_pages) == (1, 14, 7)
    request = route.last_request
    assert request.headers["accept-language"] == "ru"
    assert request.url.params["search"] == "claude" and request.url.params["type"] == "chat"
    assert request.url.params["providers"] == "anthropic" and request.url.params["limit"] == "2"


def test_model_info_exposes_decimal_pricing_and_reasoning(polza, server):
    server.get(f"{BASE}/models/anthropic/claude-sonnet-5.5").respond(200, json=SONNET)
    model = polza.model("anthropic/claude-sonnet-5.5")
    assert model.pricing.prompt_per_million == Decimal("11.70364000")
    assert model.pricing.input_cache_write_per_million is None
    assert model.reasoning.mandatory is True and model.reasoning.default_effort == "high"
    assert "response_format" in model.supported_parameters


def test_unknown_model_raises_a_typed_not_found_error(polza, server):
    server.get(f"{BASE}/models/anthropic/nope").respond(404, json=error_body("NOT_FOUND", "Модель не найдена"))
    with pytest.raises(NotFoundError) as info:
        polza.model("anthropic/nope")
    assert info.value.status == 404


def test_pricing_estimates_ruble_cost_of_a_call():
    from polza_extras import Pricing
    pricing = Pricing(prompt_per_million=Decimal(10), completion_per_million=Decimal(50),
                      input_cache_read_per_million=Decimal(1))
    assert pricing.estimate(prompt_tokens=2000, completion_tokens=100) == Decimal("0.025")
    assert pricing.estimate(prompt_tokens=2000, completion_tokens=100, cached_tokens=1000) == Decimal("0.016")


def test_balance_parses_decimal_strings_from_v2_endpoint(polza, server):
    server.get("https://polza.ai/api/v2/balance").respond(200, json={
        "amount": "1000.50", "available": "990.25", "reservedAmount": "10.25", "spentAmount": "295.08",
        "updatedAt": "2026-10-22T10:00:00.000Z"})
    balance = polza.balance()
    assert balance.available == Decimal("990.25") and balance.reserved == Decimal("10.25")
    assert balance.spent == Decimal("295.08") and balance.updated_at.year == 2026


def test_key_info_parses_limits_and_usage(polza, server):
    server.get(f"{BASE}/key").respond(200, json={
        "label": "flirt", "creator_user_id": "u", "limit": 500, "limit_remaining": 204.9, "limit_reset": "monthly",
        "usage": 295.08917456, "usage_daily": 44.1, "usage_weekly": 66.2, "usage_monthly": 66.2})
    key = polza.key_info()
    assert key.label == "flirt" and key.limit == Decimal(500) and key.limit_reset == "monthly"
    assert key.usage == Decimal("295.08917456")


def test_key_info_without_limit(polza, server):
    server.get(f"{BASE}/key").respond(200, json={"label": "x", "limit": None, "limit_remaining": None,
                                                 "limit_reset": None, "usage": 1})
    assert polza.key_info().limit is None


def test_account_calls_map_errors_like_chat(polza, server):
    server.get("https://polza.ai/api/v2/balance").respond(401, json=error_body("UNAUTHORIZED", "bad key"))
    with pytest.raises(AuthError):
        polza.balance()
