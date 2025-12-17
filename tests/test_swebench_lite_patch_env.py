from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import envweave as ew
from envweave.examples.swebench_lite_patch_env import SWEbenchEvalResult


@dataclass
class _FakeRunner:
    seen_model_patch: str = ""
    force_ok: bool = False
    force_model_patch_applied: bool = False

    def evaluate(self, *, instance_id: str, test_patch: str, model_patch: str, tests: list[str], timeout_s: float):
        del instance_id, test_patch, timeout_s
        self.seen_model_patch = str(model_patch or "")
        ok = bool(self.force_ok) or ("GOOD" in (model_patch or ""))
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
            model_patch_applied=bool(self.force_model_patch_applied),
            tests_ran=True,
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
    assert sr_bad.reward == 0.0
    assert sr_bad.info["correct"] is False

    env.reset(seed=123)
    sr_good = env.step(
        {
            "patch": (
                "diff --git a/a.py b/a.py\n"
                "--- a/a.py\n"
                "+++ b/a.py\n"
                "@@ -1 +1 @@\n"
                "-BAD\n"
                "+GOOD\n"
            )
        }
    )
    assert sr_good.reward == 1.0
    assert sr_good.info["correct"] is True

    env.close()


def test_swebench_lite_patch_env_rewrites_paths_and_shapes_reward(tmp_path: Path):
    data = tmp_path / "swebench_lite.jsonl"
    row = {
        "instance_id": "repo__x-2",
        "repo": "pytest-dev/pytest",
        "base_commit": "deadbeef",
        "problem_statement": "Fix the bug.",
        "patch": (
            "diff --git a/src/_pytest/foo.py b/src/_pytest/foo.py\n"
            "--- a/src/_pytest/foo.py\n"
            "+++ b/src/_pytest/foo.py\n"
            "@@ -1 +1 @@\n"
            "-x=1\n"
            "+x=2\n"
        ),
        "test_patch": "",
        "FAIL_TO_PASS": json.dumps(["t0"]),
        "PASS_TO_PASS": json.dumps([]),
    }
    data.write_text(json.dumps(row) + "\n", encoding="utf-8")

    runner = _FakeRunner(force_ok=False, force_model_patch_applied=False)
    env = ew.make(
        "pkg://envweave.examples.swebench_lite_patch_env:SWEbenchLitePatchEnv",
        backend="inproc",
        dataset_path=str(data),
        max_examples=1,
        seed=0,
        max_fail_tests=1,
        timeout_s=1.0,
        runner=runner,
    )

    env.reset(seed=123)
    sr = env.step(
        {
            "patch": (
                "diff --git a/_pytest/foo.py b/_pytest/foo.py\n"
                "--- a/_pytest/foo.py\n"
                "+++ b/_pytest/foo.py\n"
                "@@ -1 +1 @@\n"
                "-x=1\n"
                "+x=2\n"
            )
        }
    )
    assert "diff --git a/src/_pytest/foo.py b/src/_pytest/foo.py" in runner.seen_model_patch
    assert sr.info["patch_format_ok"] is True
    assert sr.info["patch_applied"] is False
    assert sr.info["patch_similarity"] == 1.0
    assert abs(float(sr.reward) - 0.1) < 1e-9

    runner.force_model_patch_applied = True
    env.reset(seed=123)
    sr2 = env.step(
        {
            "patch": (
                "diff --git a/src/_pytest/foo.py b/src/_pytest/foo.py\n"
                "--- a/src/_pytest/foo.py\n"
                "+++ b/src/_pytest/foo.py\n"
                "@@ -1 +1 @@\n"
                "-x=1\n"
                "+x=2\n"
            )
        }
    )
    assert sr2.info["patch_applied"] is True
    assert sr2.info["patch_similarity"] == 1.0
    assert abs(float(sr2.reward) - 0.3) < 1e-9
    env.close()
