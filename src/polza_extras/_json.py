"""Typed access to parsed JSON: containers get a known shape; a missing or wrong one is empty."""

from __future__ import annotations

from typing import Any, cast


def obj(value: Any) -> dict[str, Any]:
    return cast("dict[str, Any]", value) if isinstance(value, dict) else {}


def arr(value: Any) -> list[Any]:
    return cast("list[Any]", value) if isinstance(value, list) else []
