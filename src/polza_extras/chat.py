from __future__ import annotations

from decimal import Decimal
from typing import Any

from ._models import ChatResult, Usage
from .errors import EmptyResponseError


def _cost_rub(usage: dict[str, Any]) -> Decimal | None:
    for key in ("cost_rub", "cost"):
        value = usage.get(key)
        if isinstance(value, (int, float, str)):
            return Decimal(str(value))
    return None


def parse_chat(data: dict[str, Any], reasoning_fallback: bool = False) -> ChatResult:
    choices = data.get("choices") or []
    if not choices:
        raise EmptyResponseError("Polza returned a response without choices")
    choice = choices[0]
    message = choice.get("message") or {}
    finish_reason = choice.get("finish_reason")
    reasoning = message.get("reasoning") or message.get("reasoning_content") or None
    text = message.get("content") or ""
    usage = data.get("usage") or {}
    parsed_usage = Usage(
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        total_tokens=int(usage.get("total_tokens") or 0),
    )
    if not text:
        if reasoning and reasoning_fallback:
            text = reasoning
        else:
            hint = ""
            if finish_reason == "length" and (reasoning or message.get("reasoning_details")):
                hint = "; max_tokens was spent on thinking"
            raise EmptyResponseError(
                f"Polza returned empty content (finish_reason={finish_reason}{hint})",
                finish_reason=finish_reason,
                reasoning=reasoning,
                usage=parsed_usage,
                raw=data,
            )
    return ChatResult(
        text=text,
        reasoning=reasoning,
        finish_reason=finish_reason,
        usage=parsed_usage,
        cost_rub=_cost_rub(usage),
        model=data.get("model"),
        provider=data.get("provider"),
        id=data.get("id"),
        raw=data,
    )
