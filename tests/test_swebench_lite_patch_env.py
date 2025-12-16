from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import envweave as ew
from envweave.examples.swebench_lite_patch_env import SWEbenchEvalResult


@dataclass
class _FakeRunner:
    def evaluate(self, *, instance_id: str, test_patch: str, model_patch: str, tests: list[str], timeout_s: float):
        del instance_id, test_patch, timeout_s
        ok = "GOOD" in (model_patch or "")
        return SWEbenchEvalResult(
            ok=bool(ok),
            passed=len(tests) if ok else 0,
            failed=0 if ok else len(tests),
            total=len(tests),
            image="fake",
            duration_s=0.01,
            stdout="ok" if ok else "fail",
            stderr="",
            error=None if ok else "bad patch",
        )


def test_swebench_lite_patch_env_uses_runner(tmp_path: Path):
    data = tmp_path / "swebench_lite.jsonl"
    row = {
        "instance_id": "repo__x-1",
        "repo": "org/repo",
        "base_commit": "deadbeef",
        "problem_statement": "Fix the bug.",
        "patch": "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-print('x')\n+print('y')\n",
        "test_patch": "diff --git a/test_a.py b/test_a.py\n--- a/test_a.py\n+++ b/test_a.py\n@@ -1 +1 @@\n-assert False\n+assert True\n",
        "FAIL_TO_PASS": json.dumps(["t0", "t1"]),
        "PASS_TO_PASS": json.dumps(["p0"]),
    }
    data.write_text(json.dumps(row) + "\n", encoding="utf-8")

    env = ew.make(
        "pkg://envweave.examples.swebench_lite_patch_env:SWEbenchLitePatchEnv",
        backend="inproc",
        dataset_path=str(data),
        max_examples=1,
        seed=0,
        max_fail_tests=1,
        timeout_s=1.0,
        runner=_FakeRunner(),
    )

    obs, info = env.reset(seed=123)
    assert "Patch:" in obs.prompt
    assert info["repo"] == "org/repo"

    sr_bad = env.step({"patch": "BAD"})
    assert sr_bad.done is True
    assert sr_bad.reward == -1.0
    assert sr_bad.info["correct"] is False

    env.reset(seed=123)
    sr_good = env.step({"patch": "GOOD"})
    assert sr_good.reward == 1.0
    assert sr_good.info["correct"] is True

    env.close()

