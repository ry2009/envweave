from __future__ import annotations

import envweave as ew


def test_vector_env_step_async_and_wait():
    ew.register(
        "local://counter-vec-v0",
        factory=lambda: ew.examples.CounterEnv(done_at=2),
        overwrite=True,
    )
    venv = ew.make_vector("local://counter-vec-v0", num_envs=4, autoreset=True)
    rr = venv.reset()
    assert rr.obs == [ew.examples.CounterObs(value=0)] * 4

    venv.step_async([{"delta": 1}] * 4)
    sr = venv.step_wait()
    assert [o.value for o in sr.obs] == [1, 1, 1, 1]

    venv.close()

