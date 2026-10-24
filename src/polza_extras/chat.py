from __future__ import annotations

from decimal import Decimal
from typing import Any

from ._json import arr, obj
from ._models import ChatResult, StreamEvent, ToolCall, ToolCallDelta, Usage
from .errors import EmptyResponseError


def _cost_rub(usage: dict[str, Any]) -> Decimal | None:
    for key in ("cost_rub", "cost"):
        value = usage.get(key)
        if isinstance(value, (int, float, str)):
            return Decimal(str(value))
    return None


def _usage(usage: dict[str, Any]) -> Usage:
    return Usage(
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        total_tokens=int(usage.get("total_tokens") or 0),
    )


def _tool_calls(message: dict[str, Any]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for item in arr(message.get("tool_calls")):
        call = obj(item)
        function = obj(call.get("function"))
        calls.append(
            ToolCall(
                id=call.get("id") or "",
                name=function.get("name") or "",
                arguments=function.get("arguments") or "",
            )
        )
    return calls


def parse_chat(data: dict[str, Any], reasoning_fallback: bool = False) -> ChatResult:
    choices = arr(data.get("choices"))
    if not choices:
        raise EmptyResponseError("Polza returned a response without choices")
    choice = obj(choices[0])
    message = obj(choice.get("message"))
    finish_reason: str | None = choice.get("finish_reason")
    reasoning: str | None = message.get("reasoning") or message.get("reasoning_content") or None
    text: str = message.get("content") or ""
    tool_calls = _tool_calls(message)
    usage = obj(data.get("usage"))
    parsed_usage = _usage(usage)
    if not text and not tool_calls:
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
                cost_rub=_cost_rub(usage),
            )
    return ChatResult(
        text=text,
        tool_calls=tool_calls,
        reasoning=reasoning,
        finish_reason=finish_reason,
        usage=parsed_usage,
        cost_rub=_cost_rub(usage),
        model=data.get("model"),
        provider=data.get("provider"),
        id=data.get("id"),
        raw=data,
    )


def _tool_call_deltas(delta: dict[str, Any]) -> list[ToolCallDelta]:
    deltas: list[ToolCallDelta] = []
    for position, item in enumerate(arr(delta.get("tool_calls"))):
        call = obj(item)
        function = obj(call.get("function"))
        deltas.append(
            ToolCallDelta(
                index=call.get("index", position),
                id=call.get("id"),
                name=function.get("name"),
                arguments=function.get("arguments") or "",
            )
        )
    return deltas


def parse_stream_chunk(data: dict[str, Any]) -> StreamEvent:
    choices = arr(data.get("choices"))
    choice = obj(choices[0]) if choices else {}
    delta = obj(choice.get("delta"))
    usage = obj(data.get("usage"))
    return StreamEvent(
        delta=delta.get("content") or "",
        tool_calls=_tool_call_deltas(delta),
        reasoning=delta.get("reasoning") or delta.get("reasoning_content") or None,
        finish_reason=choice.get("finish_reason"),
        usage=_usage(usage) if usage else None,
        cost_rub=_cost_rub(usage) if usage else None,
    )
