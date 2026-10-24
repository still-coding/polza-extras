# Changelog

## 0.1.0 — 2026-10-24

First release. Not on PyPI yet; install from GitHub.

- `Polza` / `AsyncPolza` on top of the `openai` SDK, with `close()` and context managers.
- `chat()`: ruble cost (`cost_rub` as `Decimal`), reasoning text, JSON mode, Polza retry rules.
- `chat_stream()`: text and reasoning deltas, final usage and cost, `StreamError` for in-stream errors; abandoned streams are closed.
- Typed errors from Polza's error body (`AuthError`, `InsufficientFundsError`, `RateLimitError`, `NotFoundError`, …) with `trace_id` and the provider's real reason.
- `EmptyResponseError` when thinking used up `max_tokens`; it keeps usage, the raw response and the billed cost.
- Catalog: `models()`, `model()`, ruble pricing as `Decimal`, `Pricing.estimate`.
- Account: `balance()`, `key_info()`.
