from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Any

from envweave.types import (
    AutoResetReady,
    EpisodeEnd,
    LogEvent,
    NeedAction,
    Request,
    ToolCallRequest,
)


def to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        result: dict[str, Any] = {}
        for f in dataclasses.fields(obj):
            result[f.name] = to_jsonable(getattr(obj, f.name))
        # Include request type for Request subclasses.
        if isinstance(obj, Request):
            result["type"] = obj.type
        return result
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def request_from_dict(payload: Any) -> Request:
    if not isinstance(payload, dict):
        raise TypeError(f"request payload must be a dict, got {type(payload).__name__}")
    req_type = payload.get("type")
    if req_type == "need_action":
        return NeedAction(
            env_id=payload["env_id"],
            agent_id=payload.get("agent_id"),
            action_schema=payload.get("action_schema"),
            deadline_ms=payload.get("deadline_ms"),
        )
    if req_type == "episode_end":
        return EpisodeEnd(
            env_id=payload["env_id"],
            final_info=payload.get("final_info") or {},
            reason=payload.get("reason") or "other",
        )
    if req_type == "autoreset_ready":
        return AutoResetReady(
            env_id=payload["env_id"],
            initial_obs=payload.get("initial_obs"),
            initial_info=payload.get("initial_info") or {},
        )
    if req_type == "tool_call":
        return ToolCallRequest(
            env_id=payload["env_id"],
            agent_id=payload.get("agent_id"),
            tool_name=payload["tool_name"],
            tool_args_schema=payload.get("tool_args_schema"),
        )
    if req_type == "log_event":
        return LogEvent(
            env_id=payload["env_id"],
            name=payload["name"],
            payload=payload.get("payload") or {},
        )
    raise ValueError(f"unknown request type: {req_type!r}")


def requests_from_json(payload: Any) -> list[Request]:
    if payload is None:
        return []
    if not isinstance(payload, list):
        raise TypeError(f"requests must be a list, got {type(payload).__name__}")
    return [request_from_dict(x) for x in payload]

