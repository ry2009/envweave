from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import envweave as ew
from envweave.types import AutoResetReady


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
    ax_s.set_ylabel("correct (1/0)")
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

    fig.suptitle("envweave CodeFix TinyTransformer REINFORCE", fontsize=12)
    fig.tight_layout()

    out_path = run_dir / "metrics.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _encode_ascii_batch(texts: list[str], *, max_len: int) -> tuple["torch.Tensor", "torch.Tensor"]:
    import torch

    if max_len <= 0:
        raise ValueError("max_len must be positive")
    batch = torch.zeros((len(texts), max_len), dtype=torch.long)
    for i, t in enumerate(texts):
        t = t or ""
        encoded = [min(ord(ch), 127) + 1 for ch in t[:max_len]]
        if encoded:
            batch[i, : len(encoded)] = torch.tensor(encoded, dtype=torch.long)
    pad_mask = batch == 0
    return batch, pad_mask


class TinyTransformerPolicy:
    def __init__(
        self,
        *,
        num_actions: int,
        max_len: int = 256,
        vocab_size: int = 129,  # 0=pad, 1..128 = ascii 0..127 + 1
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 128,
        dropout: float = 0.0,
    ) -> None:
        import torch.nn as nn

        self.max_len = int(max_len)
        self.vocab_size = int(vocab_size)
        self.num_actions = int(num_actions)

        self._model = nn.Module()
        self._model.emb = nn.Embedding(self.vocab_size, d_model)
        self._model.pos = nn.Embedding(self.max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            batch_first=True,
            activation="gelu",
            dropout=float(dropout),
        )
        self._model.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self._model.head = nn.Linear(d_model, self.num_actions)

    def to(self, device: str):
        self._model.to(device)
        return self

    def parameters(self):
        return self._model.parameters()

    def forward(self, tokens, pad_mask):
        import torch

        bsz, seq_len = tokens.shape
        if seq_len != self.max_len:
            raise ValueError(f"expected seq_len={self.max_len}, got {seq_len}")

        pos_ids = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(bsz, seq_len)
        x = self._model.emb(tokens) + self._model.pos(pos_ids)
        x = self._model.encoder(x, src_key_padding_mask=pad_mask)

        keep = (~pad_mask).unsqueeze(-1)
        denom = keep.sum(dim=1).clamp(min=1)
        pooled = (x * keep).sum(dim=1) / denom
        return self._model.head(pooled)


@dataclass
class _RunSummary:
    episodes: int
    avg_return: float
    success_rate: float
    eps_per_s: float
    backend: str
    run_dir: str
    metrics_jsonl: str
    metrics_png: str


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Tiny REINFORCE demo on a SWE-ish text env using a tiny transformer policy."
    )
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--episodes", type=int, default=0, help="0 => run until convergence")
    p.add_argument("--max-episodes", type=int, default=20000)
    p.add_argument("--window", type=int, default=500)
    p.add_argument("--target-avg-return", type=float, default=0.7)
    p.add_argument("--target-success-rate", type=float, default=0.85)
    p.add_argument("--min-episodes", type=int, default=2000)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--backend", type=str, default="inproc", choices=["inproc", "docker_http"])
    p.add_argument("--docker-image", type=str, default="envweave-codefix")
    p.add_argument("--run-dir", type=str, default="")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args(argv)

    import torch

    torch.manual_seed(int(args.seed))

    run_dir = (
        Path(args.run_dir).expanduser()
        if str(args.run_dir).strip()
        else (Path("runs") / "codefix_pg" / _utc_run_id())
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    env_id = "local://codefix-v0"
    ew.register(
        env_id,
        factory=lambda: ew.examples.CodeFixEnv(seed=int(args.seed)),
        docker_image=str(args.docker_image),
        observation_type=ew.examples.CodeFixObs,
        action_type=ew.examples.CodeFixAction,
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
    current_obs: list[ew.examples.CodeFixObs] = list(rr.obs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = TinyTransformerPolicy(num_actions=4, max_len=int(args.max_len)).to(device)
    opt = torch.optim.AdamW(policy.parameters(), lr=float(args.lr), weight_decay=0.0)

    baseline = 0.0
    window = max(1, int(args.window))
    returns_window: deque[float] = deque(maxlen=window)
    success_window: deque[float] = deque(maxlen=window)
    episodes_done = 0

    t0 = time.perf_counter()
    with metrics_path.open("w", encoding="utf-8") as f_metrics:
        while True:
            if int(args.episodes) > 0 and episodes_done >= int(args.episodes):
                break
            if int(args.episodes) <= 0 and episodes_done >= int(args.max_episodes):
                break

            prompts = [o.prompt for o in current_obs]
            tokens, pad_mask = _encode_ascii_batch(prompts, max_len=int(args.max_len))
            tokens = tokens.to(device)
            pad_mask = pad_mask.to(device)

            logits = policy.forward(tokens, pad_mask)
            dist = torch.distributions.Categorical(logits=logits)
            actions_t = dist.sample()
            logp = dist.log_prob(actions_t)
            entropy = dist.entropy().mean()
            actions = [{"choice": int(a)} for a in actions_t.detach().cpu().tolist()]

            batch = venv.step(actions)

            rewards_list = [float(r) for r in batch.reward]  # type: ignore[arg-type]
            rewards_t = torch.tensor(rewards_list, dtype=torch.float32, device=device)
            mean_reward = float(rewards_t.mean().item())
            baseline = 0.95 * baseline + 0.05 * mean_reward

            adv = rewards_t - float(baseline)
            loss = -(logp * adv.detach()).mean() - float(args.entropy_coef) * entropy
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            for i in range(int(args.num_envs)):
                reward = float(rewards_list[i])
                correct = bool((batch.info[i] or {}).get("correct", False))
                returns_window.append(reward)
                success_window.append(1.0 if correct else 0.0)
                episodes_done += 1

                avg_return = sum(returns_window) / max(len(returns_window), 1)
                success_rate = sum(success_window) / max(len(success_window), 1)
                elapsed = time.perf_counter() - t0

                f_metrics.write(
                    json.dumps(
                        {
                            "episode": episodes_done,
                            "return": reward,
                            "terminated": correct,
                            "baseline": baseline,
                            "avg_return": avg_return,
                            "success_rate": success_rate,
                            "loss": float(loss.item()),
                            "elapsed_s": elapsed,
                            "device": device,
                        }
                    )
                    + "\n"
                )

                next_obs = batch.obs[i]
                ar = next(
                    (r for r in batch.requests[i] if isinstance(r, AutoResetReady)), None  # type: ignore[arg-type]
                )
                if ar is not None:
                    next_obs = ar.initial_obs
                current_obs[i] = next_obs

            f_metrics.flush()

            if int(args.log_every) > 0 and episodes_done % int(args.log_every) == 0:
                avg_return = sum(returns_window) / max(len(returns_window), 1)
                success_rate = sum(success_window) / max(len(success_window), 1)
                print(
                    f"episodes={episodes_done} avg_return={avg_return:.3f} "
                    f"success_rate={success_rate:.3f} baseline={baseline:.3f} "
                    f"loss={float(loss.item()):.4f} device={device}"
                )

            if (
                int(args.episodes) <= 0
                and episodes_done >= int(args.min_episodes)
                and len(returns_window) >= window
                and (sum(returns_window) / len(returns_window)) >= float(args.target_avg_return)
                and (sum(success_window) / len(success_window)) >= float(args.target_success_rate)
            ):
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

    summary = _RunSummary(
        episodes=int(episodes_done),
        avg_return=float(avg),
        success_rate=float(success_rate),
        eps_per_s=float(eps_per_s),
        backend=str(args.backend),
        run_dir=str(run_dir),
        metrics_jsonl=str(metrics_path),
        metrics_png=str(png_path),
    )
    (run_dir / "summary.json").write_text(json.dumps(summary.__dict__, indent=2) + "\n", encoding="utf-8")
    print(f"done: {json.dumps(summary.__dict__)}")

    if os.environ.get("ENVWEAVE_PRINT_LAST_PNG", "").strip():
        print(str(png_path))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
