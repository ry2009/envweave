from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from envweave.types import ResetResult, StepResult


@dataclass(frozen=True)
class CodeFixObs:
    prompt: str
    choices: tuple[str, ...]
    t: int


@dataclass(frozen=True)
class CodeFixAction:
    choice: int  # index into obs.choices


_CHOICES: tuple[str, ...] = ("+", "-", "*", "//")


class CodeFixEnv:
    """
    Tiny SWE-ish "bugfix" env (single-step contextual bandit).

    The agent chooses the operator that makes a toy function satisfy the spec.
    This is intentionally small so end-to-end RL demos converge quickly.
    """

    observation_type = CodeFixObs
    action_type = CodeFixAction

    def __init__(self, *, seed: int | None = None):
        self._rng = random.Random(seed)
        self._t = 0
        self._task_name = "add"
        self._correct_choice = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        if seed is not None:
            self._rng.seed(int(seed))
        self._t = 0

        task = self._rng.choice(
            [
                ("add", "Return a + b.", 0, (3, 2, 5)),
                ("sub", "Return a - b.", 1, (3, 2, 1)),
                ("mul", "Return a * b.", 2, (3, 2, 6)),
                ("idiv", "Return a // b (integer division).", 3, (7, 2, 3)),
            ]
        )
        self._task_name, spec, self._correct_choice, (a, b, out) = task

        prompt = (
            "Fix the operator so the function matches the spec.\n\n"
            f"SPEC: {spec}\n\n"
            "CODE:\n"
            "def f(a: int, b: int) -> int:\n"
            "    return a ? b\n\n"
            f"EXAMPLE: f({a}, {b}) == {out}\n\n"
            "Choose one operator from the choices.\n"
        )
        return ResetResult(obs=CodeFixObs(prompt=prompt, choices=_CHOICES, t=self._t), info={})

    def step(self, action: CodeFixAction | dict[str, Any] | int):
        choice = self._parse_choice(action)
        self._t += 1

        correct = int(choice) == int(self._correct_choice)
        reward = 1.0 if correct else -1.0
        info = {
            "terminated": True,
            "truncated": False,
            "task": self._task_name,
            "choice": int(choice),
            "correct_choice": int(self._correct_choice),
            "correct": bool(correct),
        }
        return StepResult(
            obs=CodeFixObs(prompt="", choices=_CHOICES, t=self._t),
            reward=float(reward),
            done=True,
            info=info,
            requests=[],
        )

    @staticmethod
    def _parse_choice(action: CodeFixAction | dict[str, Any] | int) -> int:
        if isinstance(action, CodeFixAction):
            choice = int(action.choice)
        elif isinstance(action, dict):
            choice = int(action.get("choice", -1))
        elif isinstance(action, int):
            choice = int(action)
        else:
            raise TypeError(f"unsupported action type: {type(action).__name__}")

        if choice < 0 or choice >= len(_CHOICES):
            raise ValueError(f"choice must be in [0, {len(_CHOICES) - 1}]")
        return choice

    def close(self) -> None:
        return None


__all__ = ["CodeFixEnv", "CodeFixObs", "CodeFixAction"]

