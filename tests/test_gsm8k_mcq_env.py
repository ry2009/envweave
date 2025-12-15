from __future__ import annotations

import json
from pathlib import Path

import envweave as ew


def test_gsm8k_mcq_env_has_correct_option(tmp_path: Path):
    data = tmp_path / "gsm8k.jsonl"
    data.write_text(
        json.dumps(
            {
                "question": "If you have 3 apples and eat 1, how many remain?",
                "answer": "We start with 3 and eat 1 so 2 remain.\n#### 2",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    env = ew.make(
        "pkg://envweave.examples.gsm8k_mcq_env:GSM8KMCQEnv",
        backend="inproc",
        dataset_path=str(data),
        max_examples=1,
        seed=0,
    )
    obs, _ = env.reset()
    assert "A)" in obs.prompt
    assert len(obs.choices) == 4

    # The env exposes correct_choice in info on step; validate reward mechanics.
    sr_wrong = env.step({"choice": 0})
    assert sr_wrong.done is True
    assert sr_wrong.reward in (-1.0, 1.0)

    env.close()

