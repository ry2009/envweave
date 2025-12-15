from __future__ import annotations

import json
import random
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from envweave.types import ResetResult, StepResult

_GSM8K_TRAIN_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
_GSM8K_TEST_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"

_FINAL_ANSWER_RE = re.compile(r"####\s*([-+]?\d+(?:\.\d+)?)")


@dataclass(frozen=True)
class GSM8KObs:
    prompt: str
    t: int


@dataclass(frozen=True)
class GSM8KAction:
    answer: str


def _default_cache_dir() -> Path:
    return (Path.home() / ".cache" / "envweave" / "gsm8k").expanduser()


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
        f.write(r.read())
    tmp.replace(path)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _extract_final_number(text: str) -> str | None:
    m = _FINAL_ANSWER_RE.search(text)
    if not m:
        return None
    return m.group(1).strip()


class GSM8KEnv:
    """
    Dataset-backed, single-step environment over GSM8K.

    - reset() returns a prompt containing the question.
    - step(answer_text) returns reward based on final answer correctness and formatting.

    This is designed to match envweave's text-first contract while being usable for
    real LLM RL loops (e.g. via Tinker).
    """

    observation_type = GSM8KObs
    action_type = GSM8KAction

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

        cache = Path(cache_dir).expanduser() if cache_dir else _default_cache_dir()
        if dataset_path:
            path = Path(dataset_path).expanduser()
        else:
            path = cache / f"{split}.jsonl"
            if not path.exists():
                _download(_GSM8K_TRAIN_URL if split == "train" else _GSM8K_TEST_URL, path)

        rows = _load_jsonl(path)
        if not rows:
            raise ValueError(f"no rows found in {path}")
        if max_examples is not None:
            rows = rows[: int(max_examples)]
        self._rows = rows

        self._question = ""
        self._gold = ""
        self._prompt = ""

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        if seed is not None:
            self._rng.seed(int(seed))
        self._t = 0
        row = self._rng.choice(self._rows)
        self._question = str(row.get("question", "")).strip()
        self._gold = str(row.get("answer", "")).strip()
        if not self._question or not self._gold:
            # Extremely defensive; resample.
            return self.reset(seed=seed, options=options)

        self._prompt = (
            "Solve the following math word problem. Show your work.\n"
            "Put the final numeric answer on its own line formatted exactly as:\n"
            "#### <number>\n\n"
            f"Question: {self._question}\n\nAnswer:\n"
        )
        return ResetResult(obs=GSM8KObs(prompt=self._prompt, t=self._t), info={})

    def step(self, action: GSM8KAction | dict[str, Any] | str):
        answer_text = self._parse_answer(action)
        self._t += 1

        gold_num = _extract_final_number(self._gold)
        pred_num = _extract_final_number(answer_text)
        correct = (pred_num is not None) and (gold_num is not None) and (pred_num == gold_num)
        format_ok = pred_num is not None

        # Same shaping as the Tinker cookbook example:
        #   +1 for correct, and a small penalty if format isn't respected.
        reward = (1.0 if correct else 0.0) + 0.1 * ((1.0 if format_ok else 0.0) - 1.0)

        info = {
            "terminated": True,
            "truncated": False,
            "correct": bool(correct),
            "format_ok": bool(format_ok),
            "gold_final": gold_num,
            "pred_final": pred_num,
            "split": self._split,
        }
        return StepResult(
            obs=GSM8KObs(prompt="", t=self._t),
            reward=float(reward),
            done=True,
            info=info,
            requests=[],
        )

    @staticmethod
    def _parse_answer(action: GSM8KAction | dict[str, Any] | str) -> str:
        if isinstance(action, GSM8KAction):
            return str(action.answer)
        if isinstance(action, dict):
            return str(action.get("answer", ""))
        if isinstance(action, str):
            return action
        raise TypeError(f"unsupported action type: {type(action).__name__}")

    def close(self) -> None:
        return None


__all__ = ["GSM8KEnv", "GSM8KObs", "GSM8KAction"]
