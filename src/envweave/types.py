from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

ObsT = TypeVar("ObsT")
ActionT = TypeVar("ActionT")

DoneReason = Literal["terminated", "truncated", "other"]


@dataclass(frozen=True)
class Request:
    type: str = field(init=False)


@dataclass(frozen=True)
class NeedAction(Request):
    env_id: str
    agent_id: str | None
    action_schema: dict[str, Any] | None = None
    deadline_ms: int | None = None
    type: str = field(default="need_action", init=False)


@dataclass(frozen=True)
class EpisodeEnd(Request):
    env_id: str
    final_info: dict[str, Any] = field(default_factory=dict)
    reason: DoneReason = "other"
    type: str = field(default="episode_end", init=False)


@dataclass(frozen=True)
class AutoResetReady(Request):
    env_id: str
    initial_obs: Any
    initial_info: dict[str, Any] = field(default_factory=dict)
    type: str = field(default="autoreset_ready", init=False)


@dataclass(frozen=True)
class ToolCallRequest(Request):
    env_id: str
    agent_id: str | None
    tool_name: str
    tool_args_schema: dict[str, Any] | None = None
    type: str = field(default="tool_call", init=False)


@dataclass(frozen=True)
class LogEvent(Request):
    env_id: str
    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    type: str = field(default="log_event", init=False)


@dataclass
class ResetResult:
    obs: Any
    info: dict[str, Any] = field(default_factory=dict)
    requests: list[Request] = field(default_factory=list)

    def __iter__(self):
        # Gym-style unpacking: obs, info = reset()
        yield self.obs
        yield self.info


@dataclass
class StepResult:
    obs: Any
    reward: float | dict[str, float]
    done: bool | dict[str, bool]
    info: dict[str, Any] = field(default_factory=dict)
    requests: list[Request] = field(default_factory=list)

    def __iter__(self):
        # Gym-style unpacking: obs, reward, done, info = step()
        yield self.obs
        yield self.reward
        yield self.done
        yield self.info


@dataclass
class ResetBatchResult:
    obs: list[Any]
    info: list[dict[str, Any]]
    requests: list[list[Request]]

    def __iter__(self):
        # Gym-style unpacking: obs_batch, info_batch = reset()
        yield self.obs
        yield self.info


@dataclass
class StepBatchResult:
    obs: list[Any]
    reward: list[float | dict[str, float]]
    done: list[bool | dict[str, bool]]
    info: list[dict[str, Any]]
    requests: list[list[Request]]

    def __iter__(self):
        # Gym-style unpacking: obs, reward, done, info = step()
        yield self.obs
        yield self.reward
        yield self.done
        yield self.info
