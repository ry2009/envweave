from __future__ import annotations

import json
from pathlib import Path

import envweave as ew


def _row(instance_id: str, repo: str, file_path: str) -> dict:
    patch = (
        f"diff --git a/{file_path} b/{file_path}\n"
        "index 0000000..1111111 100644\n"
        f"--- a/{file_path}\n"
        f"+++ b/{file_path}\n"
        "@@ -1 +1 @@\n"
        '-print(\"hi\")\n'
        '+print(\"hello\")\n'
    )
    return {
        "instance_id": instance_id,
        "repo": repo,
        "problem_statement": f"Bug report for {repo}: please fix {file_path}.",
        "patch": patch,
    }


def test_swebench_lite_loc_mcq_env(tmp_path: Path):
    data = tmp_path / "swebench_lite.jsonl"
    rows = [
        _row("i0", "org/repo", "a.py"),
        _row("i1", "org/repo", "b.py"),
        _row("i2", "org/repo", "c.py"),
        _row("i3", "org/repo", "d.py"),
    ]
    data.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    env = ew.make(
        "pkg://envweave.examples.swebench_lite_loc_mcq_env:SWEbenchLiteLocMCQEnv",
        backend="inproc",
        dataset_path=str(data),
        max_examples=4,
        seed=0,
    )
    obs, info = env.reset(seed=123)
    assert "Repo:" in obs.prompt
    assert "A)" in obs.prompt
    assert len(obs.choices) == 4
    assert info["repo"] == "org/repo"

    sr = env.step({"choice": 0})
    assert sr.done is True
    assert sr.reward in (-1.0, 1.0)
    assert "correct_file" in sr.info
    assert sr.info["correct_file"] in sr.info["file_choices"]

    env.close()

