from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from envweave.types import ResetResult, StepResult


@dataclass(frozen=True)
class LineWorldObs:
    pos: int
    goal: int
    t: int


@dataclass(frozen=True)
class LineWorldAction:
    move: int  # -1 (left) or +1 (right)


class LineWorldEnv:
    """
    Simple 1D navigation task for RL demos.

    - Start at pos=0, goal is either -size or +size (random each episode).
    - Actions move left/right by 1 with clamping at borders.
    - Reward: +1 on reaching goal; small step penalty; timeout penalty.
    """

    observation_type = LineWorldObs
    action_type = LineWorldAction

    def __init__(self, *, size: int = 5, max_steps: int | None = None, seed: int | None = None):
        self.size = int(size)
        if self.size <= 0:
            raise ValueError("size must be positive")
        self.max_steps = int(max_steps if max_steps is not None else (self.size * 4))
        self._rng = random.Random(seed)
        self._pos = 0
        self._goal = self.size
        self._t = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        if seed is not None:
            self._rng.seed(int(seed))
        self._pos = 0
        self._t = 0
        self._goal = self.size if self._rng.random() < 0.5 else -self.size
        return ResetResult(obs=LineWorldObs(pos=self._pos, goal=self._goal, t=self._t), info={})

    def step(self, action: LineWorldAction | dict[str, Any] | int):
        move = self._parse_move(action)
        self._t += 1
        self._pos = max(-self.size, min(self.size, self._pos + move))

        terminated = self._pos == self._goal
        truncated = self._t >= self.max_steps and not terminated
        done = terminated or truncated

        reward = 1.0 if terminated else -0.01
        if truncated:
            reward -= 1.0

        info = {"terminated": terminated, "truncated": truncated}
        return StepResult(
            obs=LineWorldObs(pos=self._pos, goal=self._goal, t=self._t),
            reward=float(reward),
            done=bool(done),
            info=info,
            requests=[],
        )

    @staticmethod
    def _parse_move(action: LineWorldAction | dict[str, Any] | int) -> int:
        if isinstance(action, LineWorldAction):
            move = int(action.move)
        elif isinstance(action, dict):
            move = int(action.get("move", 0))
        elif isinstance(action, int):
            if action in (-1, 1):
                move = int(action)
            elif action in (0, 1):
                move = -1 if int(action) == 0 else 1
            else:
                raise ValueError("int action must be -1/+1 or 0/1")
        else:
            raise TypeError(f"unsupported action type: {type(action).__name__}")

        if move not in (-1, 1):
            raise ValueError("move must be -1 or +1")
        return move

    def close(self) -> None:
        return None


__all__ = ["LineWorldEnv", "LineWorldObs", "LineWorldAction"]

