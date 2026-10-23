import pytest

from polza_extras import AsyncPolza, Polza
from tests.fake_server import FakeServer

BASE = "https://polza.ai/api/v1"
MESSAGES = [{"role": "user", "content": "Привет"}]


def completion(content="Привет!", reasoning=None, finish="stop", usage=None, choices=None):
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning"] = reasoning
    return {
        "id": "gen_1",
        "object": "chat.completion",
        "created": 1,
        "model": "openai/gpt-4o",
        "provider": "openai-direct",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}]
        if choices is None
        else choices,
        "usage": usage
        if usage is not None
        else {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "cost_rub": 0.04131306,
        },
    }


def error_body(code="BAD_REQUEST", message="m", trace="t-1"):
    return {"error": {"code": code, "message": message, "trace_id": trace}, "trace_id": trace}


@pytest.fixture
def server():
    return FakeServer()


@pytest.fixture
def sleeps():
    return []


@pytest.fixture
def polza(server, sleeps):
    return Polza(api_key="test-key", sleep=sleeps.append, http_client=server.http_client())


@pytest.fixture
def chat_route(server):
    return server.post(f"{BASE}/chat/completions")


@pytest.fixture
def apolza(server, sleeps):
    async def sleep(seconds):
        sleeps.append(seconds)

    return AsyncPolza(api_key="test-key", sleep=sleep, http_client=server.async_http_client())
