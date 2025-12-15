from __future__ import annotations

import envweave as ew
from envweave.types import AutoResetReady, EpisodeEnd


def test_autoreset_emits_events_and_is_noncontroversial():
    ew.register(
        "local://counter-autoreset-v0",
        factory=lambda: ew.examples.CounterEnv(done_at=1),
        overwrite=True,
    )
    env = ew.make("local://counter-autoreset-v0", autoreset=True)
    env.reset()

    sr = env.step({"delta": 1})
    assert sr.done is True
    assert any(isinstance(r, EpisodeEnd) for r in sr.requests)
    assert any(isinstance(r, AutoResetReady) for r in sr.requests)

    # The StepResult keeps the final observation; the initial observation for the new episode
    # is delivered via AutoResetReady.
    final_obs = sr.obs
    assert final_obs.value == 1
    ar = next(r for r in sr.requests if isinstance(r, AutoResetReady))
    assert ar.initial_obs.value == 0

    # Next step happens in a fresh episode.
    sr2 = env.step({"delta": 1})
    assert sr2.obs.value == 1

    env.close()

