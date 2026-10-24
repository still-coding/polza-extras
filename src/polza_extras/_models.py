from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal
from typing import Any, cast

from pydantic import BaseModel, Field

from ._json import obj


class Usage(BaseModel):
    """Token counts. `cached_tokens` and `cache_write_tokens` are part of `prompt_tokens`;
    `reasoning_tokens` are part of `completion_tokens`."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0


class ToolCall(BaseModel):
    """A function call the model asked for; `arguments` is the JSON string as the model wrote it."""

    id: str
    name: str
    arguments: str = ""

    def parsed_arguments(self) -> dict[str, Any]:
        """`arguments` decoded; raises ValueError when they are not a JSON object."""
        value: Any = json.loads(self.arguments or "{}")
        if not isinstance(value, dict):
            raise ValueError(f"tool call arguments are not a JSON object: {self.arguments!r}")
        return cast("dict[str, Any]", value)


class ChatResult(BaseModel):
    text: str
    tool_calls: list[ToolCall] = Field(default_factory=list[ToolCall])
    reasoning: str | None = None
    finish_reason: str | None = None
    usage: Usage
    cost_rub: Decimal | None = None
    model: str | None = None
    provider: str | None = None
    id: str | None = None
    raw: dict[str, Any]


class ToolCallDelta(BaseModel):
    """A streamed piece of a tool call. The first piece of call `index` has `id` and `name`;
    the `arguments` fragments of one index concatenate to the full JSON string."""

    index: int
    id: str | None = None
    name: str | None = None
    arguments: str = ""


class StreamEvent(BaseModel):
    """One streamed chunk: a text delta, or the final chunk with usage and ruble cost."""

    delta: str = ""
    tool_calls: list[ToolCallDelta] = Field(default_factory=list[ToolCallDelta])
    reasoning: str | None = None
    finish_reason: str | None = None
    usage: Usage | None = None
    cost_rub: Decimal | None = None


_MILLION = Decimal(1_000_000)


class Pricing(BaseModel):
    """Model prices in rubles per million tokens (None when the catalog does not list a price)."""

    prompt_per_million: Decimal | None = None
    completion_per_million: Decimal | None = None
    internal_reasoning_per_million: Decimal | None = None
    input_cache_read_per_million: Decimal | None = None
    input_cache_write_per_million: Decimal | None = None
    currency: str = "RUB"

    def estimate(
        self, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0
    ) -> Decimal:
        """Approximate cost of one call; `cached_tokens` are billed at the cache-read price."""
        prompt = self.prompt_per_million or Decimal(0)
        cache_read = (
            self.input_cache_read_per_million
            if self.input_cache_read_per_million is not None
            else prompt
        )
        completion = self.completion_per_million or Decimal(0)
        total = (
            prompt * (prompt_tokens - cached_tokens)
            + cache_read * cached_tokens
            + completion * completion_tokens
        )
        return total / _MILLION


class ReasoningInfo(BaseModel):
    mandatory: bool = False
    default_enabled: bool | None = None
    supported_efforts: list[str] = Field(default_factory=list)
    default_effort: str | None = None


class ModelInfo(BaseModel):
    """A catalog entry; pricing and parameters come from the model's top provider."""

    id: str
    name: str = ""
    type: str | None = None
    context_length: int | None = None
    max_completion_tokens: int | None = None
    reasoning: ReasoningInfo | None = None
    pricing: Pricing | None = None
    supported_parameters: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict, repr=False)

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> ModelInfo:
        top = obj(data.get("top_provider"))
        return cls(
            id=data["id"],
            name=data.get("name") or "",
            type=data.get("type"),
            context_length=data.get("context_length"),
            max_completion_tokens=data.get("max_completion_tokens"),
            reasoning=data.get("reasoning") or top.get("reasoning"),
            pricing=top.get("pricing") or data.get("pricing"),
            supported_parameters=top.get("supported_parameters") or [],
            raw=data,
        )


class ModelPage(BaseModel):
    items: list[ModelInfo]
    page: int = 1
    limit: int | None = None
    total: int | None = None
    total_pages: int | None = None


class Balance(BaseModel):
    amount: Decimal
    available: Decimal
    reserved: Decimal
    spent: Decimal
    updated_at: datetime | None = None


class KeyInfo(BaseModel):
    label: str | None = None
    limit: Decimal | None = None
    limit_remaining: Decimal | None = None
    limit_reset: str | None = None
    usage: Decimal | None = None
    usage_daily: Decimal | None = None
    usage_weekly: Decimal | None = None
    usage_monthly: Decimal | None = None
