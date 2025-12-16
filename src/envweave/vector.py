from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Sequence

from envweave.env import EnvHandle
from envweave.types import ResetBatchResult, StepBatchResult


class VectorEnv:
    """
    Thread-based vectorization with a stable, gym-like API.

    - `step_async(actions)` schedules per-env steps concurrently.
    - `step_wait()` collects results.
    - `step(actions)` is a convenience wrapper.
    """

    def __init__(self, envs: Sequence[EnvHandle], *, max_workers: int | None = None) -> None:
        self.envs = list(envs)
        if not self.envs:
            raise ValueError("VectorEnv requires at least one environment")
        self._executor = ThreadPoolExecutor(max_workers=max_workers or len(self.envs))
        self._pending: list[Future] | None = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> ResetBatchResult:
        futures = [
            self._executor.submit(
                env.reset,
                seed=(None if seed is None else int(seed) + i),
                options=options,
            )
            for i, env in enumerate(self.envs)
        ]
        results = [f.result() for f in futures]
        return ResetBatchResult(
            obs=[r.obs for r in results],
            info=[r.info for r in results],
            requests=[r.requests for r in results],
        )

    def step_async(self, actions: Sequence[Any]) -> None:
        if self._pending is not None:
            raise RuntimeError("step_async already called; call step_wait() first")
        if len(actions) != len(self.envs):
            raise ValueError(f"expected {len(self.envs)} actions, got {len(actions)}")
        self._pending = [
            self._executor.submit(self.envs[i].step, actions[i]) for i in range(len(self.envs))
        ]

    def step_wait(self) -> StepBatchResult:
        if self._pending is None:
            raise RuntimeError("step_wait called without step_async")
        pending = self._pending
        self._pending = None
        results = [f.result() for f in pending]
        return StepBatchResult(
            obs=[r.obs for r in results],
            reward=[r.reward for r in results],
            done=[r.done for r in results],
            info=[r.info for r in results],
            requests=[r.requests for r in results],
        )

    def step(self, actions: Sequence[Any]) -> StepBatchResult:
        self.step_async(actions)
        return self.step_wait()

    def close(self) -> None:
        try:
            for env in self.envs:
                env.close()
        finally:
            self._executor.shutdown(wait=True, cancel_futures=False)
