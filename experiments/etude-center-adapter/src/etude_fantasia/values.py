"""Narrow untrusted Fantasia JSON at the translation boundary."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from etude_client.errors import EtudeError
from etude_client.models import SourceKey

type JSON = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None
type Object = dict[str, JSON]


def obj(value: JSON) -> Object:
    if not isinstance(value, dict):
        raise EtudeError("invalid_source_object", 422)
    return value


def seq(value: JSON) -> list[JSON]:
    if not isinstance(value, list):
        raise EtudeError("invalid_source_array", 422)
    return value


def string(value: JSON) -> str:
    if not isinstance(value, str):
        raise EtudeError("invalid_source_string", 422)
    return value


def number(value: JSON) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EtudeError("invalid_source_number", 422)
    return float(value)


def integer(value: JSON) -> int:
    result = number(value)
    if not result.is_integer():
        raise EtudeError("invalid_source_integer", 422)
    return int(result)


def money(value: JSON) -> Decimal:
    number(value)
    return Decimal(str(value))


def timestamp(value: JSON) -> datetime:
    return datetime.fromtimestamp(number(value), UTC)


def key(kind: str, source_id: str) -> SourceKey:
    return SourceKey(namespace="fantasia", kind=kind, source_id=source_id)
