from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from pathlib import Path

from envweave.examples.gsm8k_env import (
    _GSM8K_TEST_URL,
    _GSM8K_TRAIN_URL,
    _default_cache_dir,
    _download,
    _extract_final_number,
    _load_jsonl,
)
from envweave.types import ResetResult, StepResult


@dataclass(frozen=True)
class GSM8KMCQObs:
    prompt: str
    choices: tuple[str, str, str, str]  # A,B,C,D
    t: int


@dataclass(frozen=True)
class GSM8KMCQAction:
    choice: int  # 0..3


class GSM8KMCQEnv:
    """
    GSM8K as a 4-way multiple-choice environment.

    - obs: (question + 4 numeric answer options)
    - action: choose option index 0..3
    - reward: +1 correct, -1 incorrect

    This is intentionally dense-reward and short-output to make RL demos converge quickly
    while still using real GSM8K data.
    """

    observation_type = GSM8KMCQObs
    action_type = GSM8KMCQAction

    def __init__(
        self,
        *,
        split: str = "train",
        seed: int | None = None,
        cache_dir: str | None = None,
        dataset_path: str | None = None,
        max_examples: int | None = None,
    ):
        split = str(split).strip().lower()
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        self._split = split
        self._rng = random.Random(seed)
        self._t = 0

        if dataset_path:
            path = Path(dataset_path).expanduser()
        else:
            cache = Path(cache_dir).expanduser() if cache_dir else _default_cache_dir()
            path = cache / f"{split}.jsonl"
            if not path.exists():
                _download(_GSM8K_TRAIN_URL if split == "train" else _GSM8K_TEST_URL, path)

        rows = _load_jsonl(path)
        if not rows:
            raise ValueError(f"no rows found in {path}")
        if max_examples is not None:
            rows = rows[: int(max_examples)]
        self._rows = rows

        self._prompt = ""
        self._choices: tuple[str, str, str, str] = ("", "", "", "")
        self._correct_choice = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        if seed is not None:
            self._rng.seed(int(seed))
        self._t = 0

        row = self._rng.choice(self._rows)
        question = str(row.get("question", "")).strip()
        gold = str(row.get("answer", "")).strip()
        gold_num = _extract_final_number(gold)
        if not question or not gold_num:
            return self.reset(seed=seed, options=options)

        correct = gold_num
        distractors: set[str] = set()
        while len(distractors) < 3:
            delta = self._rng.randint(1, 15)
            sign = -1 if self._rng.random() < 0.5 else 1
            try:
                candidate = str(int(correct) + sign * delta)
            except Exception:
                # If parsing fails, fallback to simple string noise.
                candidate = correct + str(sign * delta)
            if candidate != correct:
                distractors.add(candidate)

        choices = [correct, *sorted(distractors)]
        self._rng.shuffle(choices)
        self._correct_choice = int(choices.index(correct))
        a, b, c, d = choices[:4]
        self._choices = (str(a), str(b), str(c), str(d))

        self._prompt = (
            "Solve the math word problem and pick the correct final answer.\n"
            "Reply with a single letter: A, B, C, or D.\n\n"
            f"Question: {question}\n\n"
            f"A) {self._choices[0]}\n"
            f"B) {self._choices[1]}\n"
            f"C) {self._choices[2]}\n"
            f"D) {self._choices[3]}\n\n"
            "Answer:\n"
        )
        return ResetResult(obs=GSM8KMCQObs(prompt=self._prompt, choices=self._choices, t=self._t), info={})

    def step(self, action: GSM8KMCQAction | dict[str, Any] | int):
        choice = self._parse_choice(action)
        self._t += 1
        correct = int(choice) == int(self._correct_choice)
        reward = 1.0 if correct else -1.0
        info = {
            "terminated": True,
            "truncated": False,
            "correct": bool(correct),
            "correct_choice": int(self._correct_choice),
            "choice": int(choice),
            "split": self._split,
        }
        return StepResult(
            obs=GSM8KMCQObs(prompt="", choices=self._choices, t=self._t),
            reward=float(reward),
            done=True,
            info=info,
            requests=[],
        )

    @staticmethod
    def _parse_choice(action: GSM8KMCQAction | dict[str, Any] | int) -> int:
        if isinstance(action, GSM8KMCQAction):
            choice = int(action.choice)
        elif isinstance(action, dict):
            choice = int(action.get("choice", -1))
        elif isinstance(action, int):
            choice = int(action)
        else:
            raise TypeError(f"unsupported action type: {type(action).__name__}")

        if choice < 0 or choice > 3:
            raise ValueError("choice must be in [0, 3]")
        return choice

    def close(self) -> None:
        return None


__all__ = ["GSM8KMCQEnv", "GSM8KMCQObs", "GSM8KMCQAction"]
