from __future__ import annotations

import argparse
import json
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import envweave as ew
from envweave.types import AutoResetReady


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - float(np.max(x))
    ex = np.exp(x)
    return ex / float(np.sum(ex))


def _state_index(obs: ew.examples.LineWorldObs, *, size: int) -> int:
    pos_i = int(obs.pos) + size
    goal_i = 0 if int(obs.goal) < 0 else 1
    return goal_i * (2 * size + 1) + pos_i


@dataclass
class _StepLog:
    state: int
    action: int
    probs: np.ndarray
    reward: float


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values[:]
    out: list[float] = []
    acc = 0.0
    q: deque[float] = deque()
    for v in values:
        q.append(v)
        acc += v
        if len(q) > window:
            acc -= q.popleft()
        out.append(acc / len(q))
    return out


def _save_plots(
    run_dir: Path,
    *,
    episodes: list[int],
    returns: list[float],
    success: list[float],
    baseline: list[float],
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), dpi=160)
    ax_r = axes[0][0]
    ax_s = axes[0][1]
    ax_b = axes[1][0]
    ax_rs = axes[1][1]

    ax_r.plot(episodes, returns, linewidth=1.0, alpha=0.35, label="return")
    ax_r.plot(
        episodes,
        _rolling_mean(returns, window=50),
        linewidth=2.0,
        label="return/mean@50",
    )
    ax_r.set_title("return")
    ax_r.set_xlabel("episode")
    ax_r.set_ylabel("return")
    ax_r.grid(True, alpha=0.25)
    ax_r.legend(loc="lower right", fontsize=8)

    ax_s.plot(episodes, success, linewidth=1.0, alpha=0.35, label="success")
    ax_s.plot(
        episodes,
        _rolling_mean(success, window=50),
        linewidth=2.0,
        label="success/mean@50",
    )
    ax_s.set_title("success")
    ax_s.set_xlabel("episode")
    ax_s.set_ylabel("terminated (1/0)")
    ax_s.set_ylim(-0.05, 1.05)
    ax_s.grid(True, alpha=0.25)
    ax_s.legend(loc="lower right", fontsize=8)

    ax_b.plot(episodes, baseline, linewidth=2.0, label="baseline")
    ax_b.set_title("baseline")
    ax_b.set_xlabel("episode")
    ax_b.set_ylabel("baseline")
    ax_b.grid(True, alpha=0.25)
    ax_b.legend(loc="lower right", fontsize=8)

    ax_rs.scatter(success, returns, s=6, alpha=0.25)
    ax_rs.set_title("return vs success")
    ax_rs.set_xlabel("success")
    ax_rs.set_ylabel("return")
    ax_rs.grid(True, alpha=0.25)

    fig.suptitle("envweave LineWorld REINFORCE", fontsize=12)
    fig.tight_layout()

    out_path = run_dir / "metrics.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Minimal REINFORCE demo on envweave LineWorldEnv.")
    p.add_argument("--size", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=0, help="0 => default (size*4)")
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="0 => run until convergence (bounded by --max-episodes)",
    )
    p.add_argument("--max-episodes", type=int, default=5000)
    p.add_argument("--window", type=int, default=200, help="window for convergence checks")
    p.add_argument("--target-avg-return", type=float, default=0.6)
    p.add_argument("--target-success-rate", type=float, default=0.9)
    p.add_argument("--min-episodes", type=int, default=400)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--backend", type=str, default="inproc", choices=["inproc", "docker_http"])
    p.add_argument("--docker-image", type=str, default="envweave-lineworld")
    p.add_argument("--run-dir", type=str, default="")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args(argv)

    rng = np.random.default_rng(int(args.seed))
    size = int(args.size)
    max_steps = int(args.max_steps) if int(args.max_steps) > 0 else None
    window = max(1, int(args.window))

    run_dir = (
        Path(args.run_dir).expanduser()
        if str(args.run_dir).strip()
        else (Path("runs") / "lineworld_pg" / _utc_run_id())
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    env_id = "local://lineworld-v0"
    ew.register(
        env_id,
        factory=lambda: ew.examples.LineWorldEnv(size=size, max_steps=max_steps, seed=int(args.seed)),
        docker_image=str(args.docker_image),
        observation_type=ew.examples.LineWorldObs,
        action_type=ew.examples.LineWorldAction,
        overwrite=True,
    )

    venv = ew.make_vector(
        env_id,
        num_envs=int(args.num_envs),
        backend=str(args.backend),
        autoreset=True,
        max_workers=int(args.num_envs),
    )
    rr = venv.reset(seed=int(args.seed))
    current_obs: list[ew.examples.LineWorldObs] = list(rr.obs)

    num_states = (2 * size + 1) * 2
    logits = np.zeros((num_states, 2), dtype=np.float32)

    baseline = 0.0
    returns_window: deque[float] = deque(maxlen=window)
    success_window: deque[float] = deque(maxlen=window)
    episodes_done = 0

    per_env_traj: list[list[_StepLog]] = [[] for _ in range(int(args.num_envs))]

    t0 = time.perf_counter()
    with metrics_path.open("w", encoding="utf-8") as f_metrics:
        while True:
            if int(args.episodes) > 0 and episodes_done >= int(args.episodes):
                break
            if int(args.episodes) <= 0 and episodes_done >= int(args.max_episodes):
                break

            actions: list[dict[str, Any]] = []
            step_logs: list[_StepLog] = []
            for obs in current_obs:
                s = _state_index(obs, size=size)
                probs = _softmax(logits[s])
                a = int(rng.choice(2, p=probs))
                move = -1 if a == 0 else 1
                actions.append({"move": move})
                step_logs.append(_StepLog(state=s, action=a, probs=probs, reward=0.0))

            batch = venv.step(actions)

            for i in range(int(args.num_envs)):
                reward = float(batch.reward[i])  # type: ignore[arg-type]
                done = bool(batch.done[i])  # type: ignore[arg-type]
                step_logs[i].reward = reward
                per_env_traj[i].append(step_logs[i])

                if done:
                    traj = per_env_traj[i]
                    per_env_traj[i] = []

                    G = 0.0
                    returns: list[float] = []
                    for st in reversed(traj):
                        G = st.reward + float(args.gamma) * G
                        returns.append(G)
                    returns.reverse()

                    ep_return = float(returns[0]) if returns else 0.0
                    terminated = bool((batch.info[i] or {}).get("terminated", False))

                    returns_window.append(ep_return)
                    success_window.append(1.0 if terminated else 0.0)
                    baseline = 0.95 * baseline + 0.05 * ep_return

                    for st, Gt in zip(traj, returns):
                        adv = float(Gt - baseline)
                        onehot = (
                            np.array([1.0, 0.0], dtype=np.float32)
                            if st.action == 0
                            else np.array([0.0, 1.0], dtype=np.float32)
                        )
                        grad = onehot - st.probs.astype(np.float32)
                        logits[st.state] += float(args.lr) * adv * grad

                    episodes_done += 1

                    avg_return = sum(returns_window) / max(len(returns_window), 1)
                    success_rate = sum(success_window) / max(len(success_window), 1)
                    elapsed = time.perf_counter() - t0

                    f_metrics.write(
                        json.dumps(
                            {
                                "episode": episodes_done,
                                "return": ep_return,
                                "terminated": terminated,
                                "baseline": baseline,
                                "avg_return": avg_return,
                                "success_rate": success_rate,
                                "elapsed_s": elapsed,
                            }
                        )
                        + "\n"
                    )
                    f_metrics.flush()

                    next_obs = batch.obs[i]
                    ar = next(
                        (r for r in batch.requests[i] if isinstance(r, AutoResetReady)), None  # type: ignore[arg-type]
                    )
                    if ar is not None:
                        next_obs = ar.initial_obs
                    current_obs[i] = next_obs

                    if int(args.log_every) > 0 and episodes_done % int(args.log_every) == 0:
                        print(
                            f"episodes={episodes_done} avg_return={avg_return:.3f} "
                            f"success_rate={success_rate:.3f} baseline={baseline:.3f}"
                        )

                    if (
                        int(args.episodes) <= 0
                        and episodes_done >= int(args.min_episodes)
                        and len(returns_window) >= window
                        and avg_return >= float(args.target_avg_return)
                        and success_rate >= float(args.target_success_rate)
                    ):
                        break
                else:
                    current_obs[i] = batch.obs[i]

            if int(args.episodes) <= 0 and episodes_done >= int(args.min_episodes) and len(returns_window) >= window:
                avg_return = sum(returns_window) / max(len(returns_window), 1)
                success_rate = sum(success_window) / max(len(success_window), 1)
                if avg_return >= float(args.target_avg_return) and success_rate >= float(args.target_success_rate):
                    break

    elapsed = time.perf_counter() - t0
    venv.close()

    episodes: list[int] = []
    returns: list[float] = []
    success: list[float] = []
    baseline_series: list[float] = []
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            episodes.append(int(row["episode"]))
            returns.append(float(row["return"]))
            success.append(1.0 if bool(row.get("terminated")) else 0.0)
            baseline_series.append(float(row["baseline"]))

    png_path = run_dir / "metrics.png"
    if episodes and not bool(args.no_plot):
        png_path = _save_plots(
            run_dir,
            episodes=episodes,
            returns=returns,
            success=success,
            baseline=baseline_series,
        )

    avg = sum(returns_window) / max(len(returns_window), 1)
    success_rate = sum(success_window) / max(len(success_window), 1)
    eps_per_s = episodes_done / max(elapsed, 1e-9)

    summary = {
        "episodes": episodes_done,
        "avg_return": float(avg),
        "success_rate": float(success_rate),
        "eps_per_s": float(eps_per_s),
        "backend": str(args.backend),
        "run_dir": str(run_dir),
        "metrics_jsonl": str(metrics_path),
        "metrics_png": str(png_path),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"done: {json.dumps(summary)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
