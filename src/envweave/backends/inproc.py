from __future__ import annotations

import inspect
from typing import Any, Callable

from envweave.registry import EnvSpec


class _InProcEnvWrapper:
    def __init__(self, env: Any, *, autoreset: bool) -> None:
        self._env = env
        self._envweave_autoreset = autoreset

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        reset = getattr(self._env, "reset", None)
        if not callable(reset):
            raise TypeError("in-proc environment has no callable reset()")

        # Prefer gymnasium keyword-only signature; fall back to positional / no-arg.
        try:
            return reset(seed=seed, options=options)
        except TypeError:
            pass

        if seed is None and options is None:
            try:
                return reset()
            except TypeError:
                pass

        try:
            return reset(seed, options)
        except TypeError:
            return reset(seed)

    def step(self, action: Any):
        step = getattr(self._env, "step", None)
        if not callable(step):
            raise TypeError("in-proc environment has no callable step()")
        return step(action)

    def close(self) -> None:
        close = getattr(self._env, "close", None)
        if callable(close):
            close()

    def observation_schema(self):
        obs_schema = getattr(self._env, "observation_schema", None)
        if callable(obs_schema):
            return obs_schema()
        obs_type = getattr(self._env, "observation_type", None)
        if obs_type is not None and inspect.isclass(obs_type):
            from envweave.schema import json_schema_for_dataclass

            try:
                return json_schema_for_dataclass(obs_type)
            except Exception:
                return None
        return None

    def action_schema(self):
        action_schema = getattr(self._env, "action_schema", None)
        if callable(action_schema):
            return action_schema()
        action_type = getattr(self._env, "action_type", None)
        if action_type is not None and inspect.isclass(action_type):
            from envweave.schema import json_schema_for_dataclass

            try:
                return json_schema_for_dataclass(action_type)
            except Exception:
                return None
        return None


class InProcBackend:
    name = "inproc"

    def create(self, spec: EnvSpec, *, autoreset: bool, env_kwargs: dict[str, Any]) -> Any:
        if not callable(spec.factory):
            raise TypeError(f"EnvSpec.factory is not callable for env_id={spec.env_id!r}")
        env = spec.factory(**env_kwargs)
        return _InProcEnvWrapper(env, autoreset=autoreset)

