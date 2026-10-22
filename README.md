# polza-extras

Small helpers for [Polza.ai](https://polza.ai) on top of the official `openai` SDK.

Polza is OpenAI-compatible, so for chat you can use the `openai` SDK directly. This library covers what the SDK doesn't know about:

- **Ruble cost** of each call (`usage.cost_rub`) as `Decimal`.
- **Typed errors** parsed from Polza's error body, with `trace_id` and the provider's real reason (`metadata.raw`), which is often hidden behind a generic «Некорректный запрос».
- **Polza's retry rules** (which statuses to retry, `Retry-After`, own vs provider rate limits).
- **Reasoning models**: a clear `EmptyResponseError` when thinking used up `max_tokens`, instead of an empty string.
- **Catalog, balance and key info**: `models()`, `model()`, `balance()`, `key_info()`, with ruble prices as `Decimal` and a cost estimate.

## Install

Not on PyPI yet:

```bash
uv add "polza-extras @ git+https://github.com/<owner>/polza-extras"
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
| 200 with empty content | `EmptyResponseError` | no; carries `finish_reason`, `reasoning`, `usage`, `raw` |

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
POLZA_AI_API_KEY=... uv run pytest -m live --no-cov   # real API smoke tests, costs < 0.1 ₽
uv run ruff check src tests && uv run mypy src
```

## License

MIT
