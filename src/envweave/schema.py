from __future__ import annotations

from dataclasses import MISSING, is_dataclass, fields
from enum import Enum
import types
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints


def json_schema_for_dataclass(cls: type) -> dict[str, Any]:
    if not is_dataclass(cls):
        raise TypeError(f"Expected a dataclass type, got {cls!r}")

    type_hints = get_type_hints(cls, include_extras=True)
    props: dict[str, Any] = {}
    required: list[str] = []

    for f in fields(cls):
        name = f.name
        annotated_type = type_hints.get(name, Any)
        props[name] = _json_schema_for_type(annotated_type)
        if f.default is MISSING and f.default_factory is MISSING and not _is_optional(
            annotated_type
        ):
            required.append(name)

    schema: dict[str, Any] = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return schema


def _is_optional(tp: Any) -> bool:
    origin = get_origin(tp)
    if origin is None:
        return False
    if origin in (Union, types.UnionType):
        return any(arg is type(None) for arg in get_args(tp))
    return origin is types.NoneType  # pragma: no cover


def _json_schema_for_type(tp: Any) -> dict[str, Any]:
    origin = get_origin(tp)

    if tp is Any or tp is object:
        return {}

    if tp is str:
        return {"type": "string"}
    if tp is int:
        return {"type": "integer"}
    if tp is float:
        return {"type": "number"}
    if tp is bool:
        return {"type": "boolean"}

    if origin is list:
        (item_type,) = get_args(tp) or (Any,)
        return {"type": "array", "items": _json_schema_for_type(item_type)}

    if origin is dict:
        key_type, value_type = get_args(tp) or (str, Any)
        if key_type not in (str, Any, object):
            return {"type": "object"}
        return {"type": "object", "additionalProperties": _json_schema_for_type(value_type)}

    # Optional / Union
    if origin in (Union, types.UnionType):
        args = [a for a in get_args(tp)]
        if len(args) == 2 and type(None) in args:
            other = args[0] if args[1] is type(None) else args[1]
            s = _json_schema_for_type(other)
            return {"anyOf": [s, {"type": "null"}]}
        return {"anyOf": [_json_schema_for_type(a) for a in args]}

    # Literal
    if origin is Literal:
        return {"enum": list(get_args(tp))}

    if isinstance(tp, type) and issubclass(tp, Enum):
        values = [e.value for e in tp]  # type: ignore[misc]
        return {"enum": values}

    if isinstance(tp, type) and is_dataclass(tp):
        return json_schema_for_dataclass(tp)

    return {}
