from __future__ import annotations

from dataclasses import dataclass

import envweave as ew
from envweave.types import ResetResult


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


@dataclass(frozen=True)
class _SeedObs:
    seed: int


class _SeedEnv:
    def reset(self, *, seed=None, options=None):
        del options
        return ResetResult(obs=_SeedObs(seed=0 if seed is None else int(seed)), info={})

    def step(self, action):
        raise NotImplementedError

    def close(self) -> None:
        return None


def test_vector_env_reset_offsets_seed_per_env():
    ew.register(
        "local://seed-vec-v0",
        factory=lambda: _SeedEnv(),
        overwrite=True,
    )
    venv = ew.make_vector("local://seed-vec-v0", num_envs=4, autoreset=False)
    rr = venv.reset(seed=100)
    assert [o.seed for o in rr.obs] == [100, 101, 102, 103]
    venv.close()
