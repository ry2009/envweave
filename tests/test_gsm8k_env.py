from __future__ import annotations

import json
from pathlib import Path

import envweave as ew


def test_gsm8k_env_reward_correct_and_format(tmp_path: Path):
    data = tmp_path / "gsm8k.jsonl"
    data.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "question": "If you have 3 apples and eat 1, how many remain?",
                        "answer": "We start with 3 and eat 1 so 2 remain.\n#### 2",
                    }
                )
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    env = ew.make(
        "pkg://envweave.examples.gsm8k_env:GSM8KEnv",
        backend="inproc",
        dataset_path=str(data),
        max_examples=1,
        seed=0,
    )
    obs, _ = env.reset()
    assert "Question:" in obs.prompt

    sr = env.step({"answer": "some work\n#### 2"})
    assert sr.done is True
    assert sr.info["correct"] is True
    assert sr.reward == 1.0

    sr2 = env.step({"answer": "2"})
    assert sr2.reward == -0.1
    env.close()

