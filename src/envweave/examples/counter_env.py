from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from envweave.types import ResetResult, StepResult


@dataclass(frozen=True)
class CounterObs:
    value: int


@dataclass(frozen=True)
class CounterAction:
    delta: int


class CounterEnv:
    """
    Tiny example environment used for tests and demos.

    - Observation/action are dataclasses (req #6).
    - reset()/step() are gym-like, but return envweave results directly.
    """

    observation_type = CounterObs
    action_type = CounterAction

    def __init__(self, *, start: int = 0, done_at: int = 3) -> None:
        self._start = int(start)
        self._done_at = int(done_at)
        self._value = self._start

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del seed, options
        self._value = self._start
        return ResetResult(obs=CounterObs(value=self._value), info={})

    def step(self, action: CounterAction | dict[str, Any]):
        if isinstance(action, dict):
            action = CounterAction(delta=int(action.get("delta", 0)))
        self._value += int(action.delta)
        done = self._value >= self._done_at
        reward = float(self._value)
        return StepResult(
            obs=CounterObs(value=self._value),
            reward=reward,
            done=done,
            info={},
            requests=[],
        )

    def close(self) -> None:
        return None

