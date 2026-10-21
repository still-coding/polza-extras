from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic import BaseModel


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
