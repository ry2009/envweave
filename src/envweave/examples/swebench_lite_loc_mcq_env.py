from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from envweave.types import ResetResult, StepResult


@dataclass(frozen=True)
class SWEbenchLiteLocMCQObs:
    prompt: str
    choices: tuple[str, str, str, str]  # A,B,C,D (file paths)
    instance_id: str
    repo: str
    t: int


@dataclass(frozen=True)
class SWEbenchLiteLocMCQAction:
    choice: int  # 0..3


_DIFF_GIT_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)\s*$")
_PLUSPLUS_RE = re.compile(r"^\+\+\+\s+b/(.+?)\s*$")


def _extract_changed_files(patch: str) -> list[str]:
    files: list[str] = []
    seen: set[str] = set()
    for line in (patch or "").splitlines():
        m = _DIFF_GIT_RE.match(line)
        if m:
            path = m.group(2).strip()
            if path and path not in seen and path != "/dev/null":
                files.append(path)
                seen.add(path)
            continue
        m2 = _PLUSPLUS_RE.match(line)
        if m2:
            path = m2.group(1).strip()
            if path and path not in seen and path != "/dev/null":
                files.append(path)
                seen.add(path)
    return files


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _load_swebench_lite(split: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Loading SWE-bench Lite from Hugging Face requires the optional dependency. "
            "Install with: pip install -e '.[swebench]'"
        ) from e

    split = str(split).strip()
    # The project has moved orgs over time; try the modern name first, then legacy.
    for name in ("SWE-bench/SWE-bench_Lite", "princeton-nlp/SWE-bench_Lite"):
        try:
            ds = load_dataset(name, split=split)  # type: ignore[arg-type]
            return [dict(r) for r in ds]  # type: ignore[return-value]
        except Exception:
            continue
    raise RuntimeError(f"failed to load SWE-bench Lite split={split!r} from Hugging Face")


class SWEbenchLiteLocMCQEnv:
    """
    SWE-bench Lite environment for *file localization* as a 4-way multiple-choice task.

    - obs: problem statement + 4 candidate file paths
    - action: choose option index 0..3
    - reward: +1 correct, -1 incorrect

    This is intentionally "dense reward + short output" so an end-to-end RL demo can
    converge quickly while still using real SWE-bench Lite data.
    """

    observation_type = SWEbenchLiteLocMCQObs
    action_type = SWEbenchLiteLocMCQAction

    def __init__(
        self,
        *,
        split: str = "test",
        seed: int | None = None,
        dataset_path: str | None = None,
        max_examples: int | None = None,
        instance_ids: Iterable[str] | None = None,
    ):
        self._split = str(split).strip()
        self._rng = random.Random(seed)
        self._t = 0

        if dataset_path:
            rows = _load_jsonl(Path(dataset_path).expanduser())
        else:
            rows = _load_swebench_lite(self._split)

        if instance_ids is not None:
            wanted = {str(x) for x in instance_ids}
            rows = [r for r in rows if str(r.get("instance_id", "")) in wanted]
        if max_examples is not None:
            rows = rows[: int(max_examples)]
        if not rows:
            raise ValueError("no SWE-bench Lite rows loaded")

        # Keep only rows with a parseable single-file patch (Lite selection criteria).
        filtered: list[dict[str, Any]] = []
        for r in rows:
            patch = str(r.get("patch", "") or "")
            files = _extract_changed_files(patch)
            if files:
                r2 = dict(r)
                r2["_changed_files"] = files
                filtered.append(r2)
        if not filtered:
            raise ValueError("no usable rows with parseable patches (changed files)")
        self._rows = filtered

        all_files: list[str] = []
        seen: set[str] = set()
        for r in self._rows:
            for f in r.get("_changed_files", []):
                if f not in seen:
                    all_files.append(str(f))
                    seen.add(str(f))
        if len(all_files) < 4:
            raise ValueError("not enough unique changed files to build MCQ choices")
        self._all_files = all_files

        self._instance_id = ""
        self._repo = ""
        self._choices: tuple[str, str, str, str] = ("", "", "", "")
        self._correct_choice = 0
        self._prompt = ""

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        if seed is not None:
            self._rng.seed(int(seed))
        self._t = 0

        row = self._rng.choice(self._rows)
        self._instance_id = str(row.get("instance_id", "") or "")
        self._repo = str(row.get("repo", "") or "")
        problem = str(row.get("problem_statement", "") or "").strip()
        changed_files = list(row.get("_changed_files") or [])
        if not self._instance_id or not self._repo or not problem or not changed_files:
            return self.reset(seed=seed, options=options)

        correct_file = str(changed_files[0])
        distractors: list[str] = []
        while len(distractors) < 3:
            cand = str(self._rng.choice(self._all_files))
            if cand != correct_file and cand not in distractors:
                distractors.append(cand)

        choices = [correct_file, *distractors]
        self._rng.shuffle(choices)
        self._correct_choice = int(choices.index(correct_file))
        self._choices = (choices[0], choices[1], choices[2], choices[3])

        self._prompt = (
            "You are given a SWE-bench Lite GitHub issue description.\n"
            "Pick the file path that should be edited to fix the bug.\n"
            "Reply with a single letter: A, B, C, or D.\n\n"
            f"Repo: {self._repo}\n"
            f"Instance: {self._instance_id}\n\n"
            "Issue:\n"
            f"{problem}\n\n"
            f"A) {self._choices[0]}\n"
            f"B) {self._choices[1]}\n"
            f"C) {self._choices[2]}\n"
            f"D) {self._choices[3]}\n\n"
            "Answer:\n"
        )
        return ResetResult(
            obs=SWEbenchLiteLocMCQObs(
                prompt=self._prompt,
                choices=self._choices,
                instance_id=self._instance_id,
                repo=self._repo,
                t=self._t,
            ),
            info={"instance_id": self._instance_id, "repo": self._repo, "split": self._split},
        )

    def step(self, action: SWEbenchLiteLocMCQAction | dict[str, Any] | int):
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
            "instance_id": self._instance_id,
            "repo": self._repo,
            "split": self._split,
            "file_choices": list(self._choices),
            "correct_file": self._choices[int(self._correct_choice)],
        }
        return StepResult(
            obs=SWEbenchLiteLocMCQObs(
                prompt="",
                choices=self._choices,
                instance_id=self._instance_id,
                repo=self._repo,
                t=self._t,
            ),
            reward=float(reward),
            done=True,
            info=info,
            requests=[],
        )

    @staticmethod
    def _parse_choice(action: SWEbenchLiteLocMCQAction | dict[str, Any] | int) -> int:
        if isinstance(action, SWEbenchLiteLocMCQAction):
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


__all__ = ["SWEbenchLiteLocMCQEnv", "SWEbenchLiteLocMCQObs", "SWEbenchLiteLocMCQAction"]

