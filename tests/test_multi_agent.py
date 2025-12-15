from __future__ import annotations

import envweave as ew
from envweave.types import ResetResult, StepResult


class _TwoAgentOneStepEnv:
    def __init__(self):
        self._t = 0

    def reset(self, *, seed=None, options=None):
        del seed, options
        self._t = 0
        return ResetResult(obs={"alice": "ready", "bob": "ready"}, info={})

    def step(self, action):
        assert isinstance(action, dict)
        self._t += 1
        done = True
        reward = {agent: float(payload.get("delta", 0)) for agent, payload in action.items()}
        obs = {agent: f"t={self._t}" for agent in action.keys()}
        done_dict = {"__all__": done, **{agent: done for agent in action.keys()}}
        return StepResult(obs=obs, reward=reward, done=done_dict, info={}, requests=[])

    def close(self):
        return None


def test_multi_agent_dict_action_and_outputs():
    ew.register("local://two-agent-v0", factory=_TwoAgentOneStepEnv, overwrite=True)
    env = ew.make("local://two-agent-v0", backend="inproc")

    obs, info = env.reset()
    assert obs == {"alice": "ready", "bob": "ready"}
    assert info == {}

    obs2, reward2, done2, info2 = env.step({"alice": {"delta": 1}, "bob": {"delta": 2}})
    assert reward2 == {"alice": 1.0, "bob": 2.0}
    assert done2["__all__"] is True
    assert info2 == {}

    # Episode boundary is surfaced via requests, even for multi-agent dones.
    sr = env.step({"alice": {"delta": 0}, "bob": {"delta": 0}})
    assert any(r.type == "episode_end" for r in sr.requests)

    env.close()

