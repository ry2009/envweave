from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Protocol

from envweave.errors import ProtocolError
from envweave.schema import json_schema_for_dataclass
from envweave.types import (
    AutoResetReady,
    EpisodeEnd,
    Request,
    ResetResult,
    StepResult,
)


class Env(Protocol):
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> ResetResult: ...

    def step(self, action: Any) -> StepResult: ...

    def close(self) -> None: ...


def _coerce_reset_result(ret: Any) -> ResetResult:
    if isinstance(ret, ResetResult):
        ret.requests = list(ret.requests or [])
        ret.info = dict(ret.info or {})
        return ret
    if isinstance(ret, tuple):
        if len(ret) == 2:
            obs, info = ret
            return ResetResult(obs=obs, info=dict(info or {}), requests=[])
        if len(ret) == 1:
            return ResetResult(obs=ret[0], info={}, requests=[])
        raise ProtocolError(f"unsupported reset() return tuple length: {len(ret)}")
    return ResetResult(obs=ret, info={}, requests=[])


def _episode_end_reason(info: dict[str, Any]) -> str:
    if info.get("terminated") is True:
        return "terminated"
    if info.get("truncated") is True:
        return "truncated"
    return "other"


def _coerce_step_result(ret: Any) -> StepResult:
    if isinstance(ret, StepResult):
        ret.requests = list(ret.requests or [])
        ret.info = dict(ret.info or {})
        return ret
    if not isinstance(ret, tuple):
        raise ProtocolError(
            "step() must return StepResult or a gym-like tuple "
            "(obs, reward, done, info) or (obs, reward, terminated, truncated, info)"
        )
    if len(ret) == 4:
        obs, reward, done, info = ret
        return StepResult(
            obs=obs,
            reward=reward,
            done=done,
            info=dict(info or {}),
            requests=[],
        )
    if len(ret) == 5:
        obs, reward, terminated, truncated, info = ret
        info_dict = dict(info or {})
        info_dict.setdefault("terminated", bool(terminated))
        info_dict.setdefault("truncated", bool(truncated))
        done = bool(terminated) or bool(truncated)
        return StepResult(
            obs=obs,
            reward=reward,
            done=done,
            info=info_dict,
            requests=[],
        )
    raise ProtocolError(f"unsupported step() return tuple length: {len(ret)}")


def _has_episode_end(requests: list[Request]) -> bool:
    return any(isinstance(r, EpisodeEnd) for r in requests)


def _done_scalar(done: bool | dict[str, bool]) -> bool:
    if isinstance(done, bool):
        return done
    if "__all__" in done:
        return bool(done["__all__"])
    return all(bool(v) for v in done.values())


@dataclass
class EnvHandle:
    env_id: str
    instance_id: str
    _impl: Any
    observation_type: type | None = None
    action_type: type | None = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> ResetResult:
        rr = _coerce_reset_result(self._impl.reset(seed=seed, options=options))
        rr.requests = list(rr.requests or [])
        return rr

    def step(self, action: Any) -> StepResult:
        sr = _coerce_step_result(self._impl.step(action))
        sr.requests = list(sr.requests or [])

        # Ensure we always expose episode boundaries as a first-class event.
        done_scalar = _done_scalar(sr.done)
        if done_scalar and not _has_episode_end(sr.requests):
            sr.requests.append(
                EpisodeEnd(
                    env_id=self.instance_id,
                    final_info=dict(sr.info or {}),
                    reason=_episode_end_reason(sr.info or {}),
                )
            )

        autoreset = bool(getattr(self._impl, "_envweave_autoreset", False))
        if autoreset and done_scalar:
            rr = self.reset()
            sr.requests.append(
                AutoResetReady(
                    env_id=self.instance_id,
                    initial_obs=rr.obs,
                    initial_info=rr.info,
                )
            )

        return sr

    def close(self) -> None:
        close = getattr(self._impl, "close", None)
        if callable(close):
            close()

    def observation_schema(self) -> dict[str, Any] | None:
        method = getattr(self._impl, "observation_schema", None)
        if callable(method):
            schema = method()
            if schema is not None:
                return dict(schema)
        if self.observation_type and dataclasses.is_dataclass(self.observation_type):
            return json_schema_for_dataclass(self.observation_type)
        return None

    def action_schema(self) -> dict[str, Any] | None:
        method = getattr(self._impl, "action_schema", None)
        if callable(method):
            schema = method()
            if schema is not None:
                return dict(schema)
        if self.action_type and dataclasses.is_dataclass(self.action_type):
            return json_schema_for_dataclass(self.action_type)
        return None
