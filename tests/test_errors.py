import pytest

from polza_extras.errors import (
    AuthError,
    ForbiddenError,
    InsufficientFundsError,
    NotFoundError,
    PolzaError,
    RateLimitError,
    RequestError,
    ServerError,
    error_from_response,
)

BODY = {
    "error": {
        "code": "TOO_MANY_REQUESTS",
        "message": "Слишком много запросов. Повторите позже",
        "trace_id": "abc-123",
        "metadata": {"provider_name": "alibaba", "raw": "slow down", "reason": "providerRateLimit"},
    },
    "trace_id": "abc-123",
}


def test_parses_polza_error_body_fields():
    err = error_from_response(429, BODY, {})
    assert isinstance(err, RateLimitError)
    assert (err.status, err.code, err.trace_id) == (429, "TOO_MANY_REQUESTS", "abc-123")
    assert err.message.startswith("Слишком много")
    assert err.metadata == {"provider_name": "alibaba", "raw": "slow down", "reason": "providerRateLimit"}


def test_rate_limit_with_retry_after_is_our_own_limit():
    err = error_from_response(429, BODY, {"Retry-After": "42"})
    assert err.retry_after == 42.0 and err.scope == "own"


def test_rate_limit_without_retry_after_is_a_provider_limit():
    err = error_from_response(429, BODY, {})
    assert err.retry_after is None and err.scope == "provider"


@pytest.mark.parametrize(
    "status, expected",
    [
        (401, AuthError),
        (402, InsufficientFundsError),
        (403, ForbiddenError),
        (408, ServerError),
        (500, ServerError),
        (502, ServerError),
        (503, ServerError),
        (400, RequestError),
        (404, NotFoundError),
        (413, RequestError),
    ],
)
def test_status_maps_to_error_class(status, expected):
    assert type(error_from_response(status, {"error": {"code": "X", "message": "m"}}, {})) is expected


def test_all_errors_share_a_base_class():
    assert issubclass(AuthError, PolzaError) and issubclass(RateLimitError, PolzaError)


def test_missing_or_non_json_body_still_produces_an_error():
    err = error_from_response(502, None, {})
    assert isinstance(err, ServerError)
    assert err.code is None and err.trace_id is None and err.message
    err2 = error_from_response(400, {"unexpected": True}, {})
    assert isinstance(err2, RequestError) and err2.message


def test_top_level_trace_id_is_used_when_error_has_none():
    err = error_from_response(400, {"error": {"code": "BAD_REQUEST", "message": "x"}, "trace_id": "top"}, {})
    assert err.trace_id == "top"


def test_str_includes_status_code_and_trace_id():
    text = str(error_from_response(401, {"error": {"code": "UNAUTHORIZED", "message": "bad key"},
                                        "trace_id": "t1"}, {}))
    assert "401" in text and "UNAUTHORIZED" in text and "bad key" in text and "t1" in text


def test_str_includes_provider_raw_reason_hidden_behind_a_generic_message():
    body = {"error": {"code": "BAD_REQUEST", "message": "Некорректный запрос.",
                      "metadata": {"raw": "Reasoning is mandatory for this endpoint and cannot be disabled."}},
            "trace_id": "t2"}
    text = str(error_from_response(400, body, {}))
    assert "Некорректный запрос." in text and "Reasoning is mandatory" in text
