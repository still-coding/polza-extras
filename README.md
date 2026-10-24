# polza-extras

[![PyPI](https://img.shields.io/pypi/v/polza-extras)](https://pypi.org/project/polza-extras/)
[![Python](https://img.shields.io/pypi/pyversions/polza-extras)](https://pypi.org/project/polza-extras/)
[![CI](https://github.com/still-coding/polza-extras/actions/workflows/ci.yml/badge.svg)](https://github.com/still-coding/polza-extras/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/pypi/l/polza-extras)](https://github.com/still-coding/polza-extras/blob/main/LICENSE)

Small helpers for [Polza.ai](https://polza.ai) on top of the official `openai` SDK.

Polza is OpenAI-compatible, so for chat you can use the `openai` SDK directly. This library covers what the SDK doesn't know about:

- **Ruble cost** of each call (`usage.cost_rub`) as `Decimal`.
- **Typed errors** parsed from Polza's error body, with `trace_id` and the provider's real reason (`metadata.raw`), which is often hidden behind a generic «Некорректный запрос».
- **Polza's retry rules** (which statuses to retry, `Retry-After`, own vs provider rate limits).
- **Reasoning models**: a clear `EmptyResponseError` when thinking used up `max_tokens`, instead of an empty string.
- **Tool calls, usage details, structured output**: parsed `tool_calls`, cached/reasoning token counts, `json_schema=` with a dict or a pydantic model.
- **Catalog, balance and key info**: `models()`, `model()`, `balance()`, `key_info()`, with ruble prices as `Decimal` and a cost estimate.

## Install

```bash
uv add polza-extras     # or: pip install polza-extras
```

Python 3.11+. Dependencies: `openai`, `pydantic`.

## Quickstart

```python
from polza_extras import Polza

polza = Polza()  # reads POLZA_AI_API_KEY; base URL https://polza.ai/api/v1

result = polza.chat(
    "anthropic/claude-haiku-5.5",          # always provider/model; a bare name is a 404
    [{"role": "user", "content": "Привет!"}],
    max_tokens=200,
    reasoning={"type": "disabled"},        # any extra Polza field goes into the request body
)
print(result.text, result.cost_rub)       # 'Привет! …' Decimal('0.0123')

data = polza.chat(model, messages, json=True)   # response_format={"type": "json_object"}

polza.openai                               # the configured openai.OpenAI client, for anything else
polza.close()                              # or: with Polza() as polza: ...
```

### Tool calls

```python
result = polza.chat(model, messages, tools=[weather_tool])   # tools pass through as is
for call in result.tool_calls:                # [] for a plain text reply
    call.id, call.name, call.arguments        # arguments: the JSON string from the model
    args = call.parsed_arguments()            # dict; ValueError if not a JSON object
```

A reply with tool calls usually has no text: `result.text == ""` and `finish_reason == "tool_calls"`. `EmptyResponseError` is raised only when there is neither text nor tool calls.

In a stream, `event.tool_calls` holds the raw `ToolCallDelta` pieces: the first piece of a call has `index`, `id` and `name`; join the `arguments` fragments of one `index` to get the full JSON.

### Usage details

```python
u = result.usage
u.prompt_tokens, u.completion_tokens, u.total_tokens
u.cached_tokens        # prompt tokens read from the prompt cache
u.cache_write_tokens   # prompt tokens written to the prompt cache
u.reasoning_tokens     # completion tokens spent on thinking
```

They are 0 when Polza does not report them. The last stream event carries the same `usage`.

### Structured output

```python
from pydantic import BaseModel

class City(BaseModel):
    name: str
    population_millions: float

result = polza.chat(model, messages, json_schema=City, schema_name="city")
city = City.model_validate_json(result.text)
```

`json_schema` takes a pydantic model class or a JSON Schema dict and sends `response_format={"type": "json_schema", "json_schema": {"name": schema_name, "schema": ..., "strict": True}}`. Not every provider supports it: check `json_schema_strict` in the model's provider capabilities. Strict mode on some providers also wants `"additionalProperties": false` and every property in `required`; pass a dict when you need full control. Use either `json=True` (any JSON object) or `json_schema`, not both.

### Streaming

```python
for event in polza.chat_stream("anthropic/claude-haiku-5.5", messages, max_tokens=200):
    print(event.delta, end="", flush=True)   # event.reasoning holds thinking deltas, if any
print(event.cost_rub)                        # the last event carries usage and cost
```

Errors before the first chunk are retried like `chat()`. An error event inside the stream raises `StreamError`; text already yielded stays yielded, and the request is not retried. Stopping early (`break`) closes the connection.

### Async

`AsyncPolza` has the same methods, awaited (`async for` for `chat_stream`):

```python
from polza_extras import AsyncPolza

polza = AsyncPolza()
result = await polza.chat("anthropic/claude-haiku-5.5", messages)
async for event in polza.chat_stream("anthropic/claude-haiku-5.5", messages):
    ...
```

### Catalog, balance, key

```python
page = polza.models(search="claude", type="chat", limit=10)   # one catalog page
for m in page.items:
    print(m.id, m.pricing.prompt_per_million, m.reasoning.mandatory)

sonnet = polza.model("anthropic/claude-sonnet-5.5")            # NotFoundError for unknown ids
sonnet.pricing.estimate(prompt_tokens=1500, completion_tokens=80)  # Decimal rubles

polza.balance()    # Balance(amount, available, reserved, spent, updated_at)
polza.key_info()   # KeyInfo(label, limit, limit_remaining, limit_reset, usage, usage_daily, ...)
```

## Errors

All errors subclass `PolzaError` (`status`, `code`, `message`, `trace_id`, `metadata`, `details`, `retry_after`).

| Status | Class | Retried by default |
|---|---|---|
| 401 | `AuthError` | no |
| 402 | `InsufficientFundsError` | no (opt-in `retry_402` for parallel-load reserve conflicts) |
| 403 | `ForbiddenError` | no |
| 404 | `NotFoundError` | no |
| 400, 413, other 4xx | `RequestError` | no |
| 429 with `Retry-After` | `RateLimitError(scope="own")` | yes, waits `Retry-After` (fails fast if above `max_wait`) |
| 429 without `Retry-After` | `RateLimitError(scope="provider")` | yes, backoff 1, 2, 4, 8 s |
| 408, 502 | `ServerError` | yes, backoff |
| 503 | `ServerError` | yes, honors `Retry-After` |
| connection failure | `PolzaConnectionError` | yes, backoff |
| 200 with no text and no tool calls | `EmptyResponseError` | no; carries `finish_reason`, `reasoning`, `usage`, `cost_rub` (still billed), `raw` |
| error event inside a stream | `StreamError` | no |

Polza doesn't charge for failed requests, so retrying is safe. Tune with `Polza(retry=RetryPolicy(max_retries=..., base_delay=..., max_wait=...))`.

### Reasoning models

Thinking tokens count against `max_tokens`. If the model spends them all on thinking, `content` comes back empty and `chat()` raises `EmptyResponseError` with `finish_reason="length"`. Fixes:

- turn thinking off: `reasoning={"type": "disabled"}` (where allowed),
- lower it: `reasoning={"effort": "low"}`,
- raise `max_tokens`.

Some models (e.g. `anthropic/claude-sonnet-5.5`) make thinking mandatory; `model(id).reasoning.mandatory` tells you.

## Development

```bash
uv sync
uv run pytest                 # offline, against an in-process fake server; fails below 95% coverage
POLZA_AI_API_KEY=... uv run pytest -m live --no-cov   # real API smoke tests, costs ~0.1 ₽
uv run ruff check src tests && uv run ruff format --check src tests && uv run pyright
ln -sf ../../scripts/pre-commit .git/hooks/pre-commit   # all of the above before each commit
```

## License

MIT
