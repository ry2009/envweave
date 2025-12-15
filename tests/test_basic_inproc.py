from __future__ import annotations

import envweave as ew


def test_register_and_make_inproc():
    ew.register("local://counter-v0", factory=lambda: ew.examples.CounterEnv(done_at=2), overwrite=True)

    env = ew.make("local://counter-v0", backend="inproc")
    rr = env.reset()
    assert rr.obs.value == 0

    sr = env.step({"delta": 1})
    assert sr.obs.value == 1
    assert sr.reward == 1.0
    assert sr.done is False

    env.close()


def test_pkg_resolver_inproc():
    env = ew.make("pkg://envweave.examples.counter_env:CounterEnv", backend="inproc")
    rr = env.reset()
    assert rr.obs.value == 0
    sr = env.step({"delta": 3})
    assert sr.done is True
    env.close()

