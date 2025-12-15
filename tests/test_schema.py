from __future__ import annotations

import envweave as ew


def test_dataclass_schema_export():
    ew.register("local://counter-schema-v0", factory=lambda: ew.examples.CounterEnv(), overwrite=True)
    env = ew.make("local://counter-schema-v0")

    obs_schema = env.observation_schema()
    act_schema = env.action_schema()

    assert obs_schema and obs_schema["type"] == "object"
    assert "value" in obs_schema["properties"]

    assert act_schema and act_schema["type"] == "object"
    assert "delta" in act_schema["properties"]

    env.close()

