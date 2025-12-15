from __future__ import annotations

import envweave as ew


def test_codefix_env_basic():
    env = ew.make("pkg://envweave.examples.codefix_env:CodeFixEnv", backend="inproc")
    obs, info = env.reset(seed=123)
    assert info == {}
    assert "SPEC:" in obs.prompt
    assert len(obs.choices) == 4

    sr = env.step({"choice": 0})
    assert sr.done is True
    assert sr.reward in (-1.0, 1.0)
    assert "correct_choice" in sr.info
    env.close()

