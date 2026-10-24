# Changelog

## 0.2.1 — 2026-10-24

Discoverability only, no code changes.

- PyPI metadata: more keywords and classifiers, `Documentation` and `Source` links.
- README badges.
- Release workflow also creates a GitHub Release from this changelog.

## 0.2.0 — 2026-10-24

First release on PyPI: `uv add polza-extras`.

- Tool calls: `ChatResult.tool_calls` (`ToolCall` with `id`, `name`, `arguments` and `parsed_arguments()`). A reply with tool calls and no text no longer raises `EmptyResponseError`.
- Streaming tool calls: `StreamEvent.tool_calls` passes `ToolCallDelta` pieces (`index`, `id`, `name`, `arguments` fragment) through as they arrive.
- Usage details: `Usage.cached_tokens`, `Usage.cache_write_tokens`, `Usage.reasoning_tokens` (0 when Polza does not send them), in `chat()` and in the last stream event.
- Structured output: `chat(..., json_schema=..., schema_name="response")` and the same in `chat_stream()`; takes a JSON Schema dict or a pydantic model class and sends a strict `json_schema` response format. Combining it with `json=True` or `response_format` raises `ValueError`.
- CI runs on Python 3.11–3.14, plus 3.15 pre-releases as experimental.
- Release workflow: tags `v*` are built with `uv build` and published to PyPI with Trusted Publishing.

## 0.1.0 — 2026-10-24

First release. Not on PyPI yet; install from GitHub.

- `Polza` / `AsyncPolza` on top of the `openai` SDK, with `close()` and context managers.
- `chat()`: ruble cost (`cost_rub` as `Decimal`), reasoning text, JSON mode, Polza retry rules.
- `chat_stream()`: text and reasoning deltas, final usage and cost, `StreamError` for in-stream errors; abandoned streams are closed.
- Typed errors from Polza's error body (`AuthError`, `InsufficientFundsError`, `RateLimitError`, `NotFoundError`, …) with `trace_id` and the provider's real reason.
- `EmptyResponseError` when thinking used up `max_tokens`; it keeps usage, the raw response and the billed cost.
- Catalog: `models()`, `model()`, ruble pricing as `Decimal`, `Pricing.estimate`.
- Account: `balance()`, `key_info()`.
