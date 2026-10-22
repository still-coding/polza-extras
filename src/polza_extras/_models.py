from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResult(BaseModel):
    text: str
    reasoning: str | None = None
    finish_reason: str | None = None
    usage: Usage
    cost_rub: Decimal | None = None
    model: str | None = None
    provider: str | None = None
    id: str | None = None
    raw: dict[str, Any]


_MILLION = Decimal(1_000_000)


class Pricing(BaseModel):
    """Model prices in rubles per million tokens (None when the catalog does not list a price)."""

    prompt_per_million: Decimal | None = None
    completion_per_million: Decimal | None = None
    internal_reasoning_per_million: Decimal | None = None
    input_cache_read_per_million: Decimal | None = None
    input_cache_write_per_million: Decimal | None = None
    currency: str = "RUB"

    def estimate(self, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> Decimal:
        """Approximate cost of one call; `cached_tokens` of the prompt are billed at the cache-read price."""
        prompt = self.prompt_per_million or Decimal(0)
        cache_read = self.input_cache_read_per_million if self.input_cache_read_per_million is not None else prompt
        completion = self.completion_per_million or Decimal(0)
        total = (prompt * (prompt_tokens - cached_tokens) + cache_read * cached_tokens
                 + completion * completion_tokens)
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
        top = data.get("top_provider") or {}
        return cls(
            id=data["id"], name=data.get("name") or "", type=data.get("type"),
            context_length=data.get("context_length"), max_completion_tokens=data.get("max_completion_tokens"),
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
