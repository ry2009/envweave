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

import numpy as np

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
    rewards: list[float],
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

    ax_r.plot(episodes, rewards, linewidth=1.0, alpha=0.35, label="reward")
    ax_r.plot(
        episodes,
        _rolling_mean(rewards, window=50),
        linewidth=2.0,
        label="reward/mean@50",
    )
    ax_r.set_title("reward")
    ax_r.set_xlabel("episode")
    ax_r.set_ylabel("reward")
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

    ax_rs.scatter(success, rewards, s=6, alpha=0.25)
    ax_rs.set_title("reward vs success")
    ax_rs.set_xlabel("success")
    ax_rs.set_ylabel("reward")
    ax_rs.grid(True, alpha=0.25)

    fig.suptitle("envweave SWE-bench Lite Patch (Tinker) RL", fontsize=12)
    fig.tight_layout()

    out_path = run_dir / "metrics.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _require_env(name: str) -> None:
    if not os.environ.get(name, "").strip():
        raise RuntimeError(f"{name} is required in the environment")


def _to_tinker_tensor(array: np.ndarray):
    from tinker import TensorData

    return TensorData.from_numpy(array)


def _build_datum(
    *,
    full_tokens: list[int],
    sampling_logprobs: list[float],
    prompt_len: int,
    advantage: float,
):
    from tinker import Datum, ModelInput

    if len(full_tokens) != len(sampling_logprobs):
        raise ValueError("full_tokens and sampling_logprobs must have same length")
    if prompt_len < 0 or prompt_len > len(full_tokens):
        raise ValueError("invalid prompt_len")

    advantages = [0.0] * int(prompt_len) + [float(advantage)] * (len(full_tokens) - int(prompt_len))

    return Datum(
        model_input=ModelInput.from_ints(list(full_tokens)),
        loss_fn_inputs={
            "target_tokens": _to_tinker_tensor(np.array(full_tokens, dtype=np.int64)),
            "logprobs": _to_tinker_tensor(np.array(sampling_logprobs, dtype=np.float32)),
            "advantages": _to_tinker_tensor(np.array(advantages, dtype=np.float32)),
        },
    )


def _extract_patch(text: str) -> str:
    t = (text or "").strip()
    if "```" in t:
        parts = t.split("```")
        if len(parts) >= 3:
            t = parts[1].strip()
            # Strip an optional language header like ```diff
            if "\n" in t and t.splitlines()[0].strip().lower() in ("diff", "patch"):
                t = "\n".join(t.splitlines()[1:]).strip()
    return t.strip()


def _eval_patch(
    *,
    env_id: str,
    backend: str,
    docker_extra_args: list[str],
    num_envs: int,
    seed: int,
    sampling_client,
    tokenizer,
    max_tokens: int,
    temperature: float,
    top_p: float,
    episodes: int,
) -> tuple[float, float]:
    from tinker import ModelInput, SamplingParams

    vector_kwargs: dict[str, Any] = {}
    if str(backend) == "docker_http":
        vector_kwargs["docker_extra_args"] = list(docker_extra_args)

    venv = ew.make_vector(
        env_id,
        num_envs=int(num_envs),
        backend=str(backend),
        autoreset=True,
        max_workers=int(num_envs),
        **vector_kwargs,
    )
    rr = venv.reset(seed=int(seed))
    current_obs: list[ew.examples.SWEbenchLitePatchObs] = list(rr.obs)

    total_reward = 0.0
    total_correct = 0
    episodes_done = 0
    while episodes_done < int(episodes):
        prompt_tokens_list = [tokenizer.encode(o.prompt, add_special_tokens=True) for o in current_obs]

        sample_futures = []
        for i, pt in enumerate(prompt_tokens_list):
            params = SamplingParams(
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                seed=int(seed) + episodes_done + i,
            )
            sample_futures.append(
                sampling_client.sample(
                    ModelInput.from_ints(list(pt)),
                    num_samples=1,
                    sampling_params=params,
                    include_prompt_logprobs=False,
                )
            )
        sample_responses = [f.result() for f in sample_futures]

        actions: list[dict[str, Any]] = []
        for pt, resp in zip(prompt_tokens_list, sample_responses):
            del pt
            seq = resp.sequences[0]
            gen_tokens = list(seq.tokens)
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            patch = _extract_patch(gen_text)
            actions.append({"patch": patch})

        batch = venv.step(actions)
        batch_rewards = [float(r) for r in batch.reward]  # type: ignore[arg-type]
        batch_correct = [bool((batch.info[i] or {}).get("correct", False)) for i in range(int(num_envs))]

        for i in range(int(num_envs)):
            if episodes_done >= int(episodes):
                break
            total_reward += float(batch_rewards[i])
            total_correct += 1 if bool(batch_correct[i]) else 0
            episodes_done += 1

            next_obs = batch.obs[i]
            ar = next(
                (r for r in batch.requests[i] if isinstance(r, AutoResetReady)), None  # type: ignore[arg-type]
            )
            if ar is not None:
                next_obs = ar.initial_obs
            current_obs[i] = next_obs

    venv.close()
    avg_reward = total_reward / max(float(episodes_done), 1.0)
    success_rate = float(total_correct) / max(float(episodes_done), 1.0)
    return float(avg_reward), float(success_rate)


@dataclass
class _RunSummary:
    episodes: int
    avg_reward: float
    success_rate: float
    eps_per_s: float
    backend: str
    run_dir: str
    metrics_jsonl: str
    metrics_png: str
    base_model: str
    lora_rank: int
    train_split: str
    train_instance_ids: list[str] | None = None
    eval_split: str | None = None
    eval_episodes: int | None = None
    eval_base_avg_reward: float | None = None
    eval_base_success_rate: float | None = None
    eval_trained_avg_reward: float | None = None
    eval_trained_success_rate: float | None = None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="RL fine-tuning on SWE-bench Lite patch-generation via Tinker, driven by envweave."
    )
    p.add_argument("--base-model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", type=str, default="inproc", choices=["inproc", "docker_http"])
    p.add_argument("--docker-image", type=str, default="envweave-swebench-lite-patch")
    p.add_argument("--docker-sock", action="store_true", help="mount /var/run/docker.sock into env container")
    p.add_argument("--train-split", type=str, default="dev", choices=["dev", "test"])
    p.add_argument("--instance-ids", type=str, default="", help="comma-separated subset of SWE-bench instance_ids")
    p.add_argument("--max-examples", type=int, default=16, help="limit dataset examples for faster demos")
    p.add_argument("--max-fail-tests", type=int, default=8, help="limit FAIL_TO_PASS tests per episode")
    p.add_argument("--env-timeout-s", type=float, default=600.0)
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--eval-split", type=str, default="test", choices=["dev", "test", "none"])
    p.add_argument("--eval-episodes", type=int, default=16)
    p.add_argument("--eval-temperature", type=float, default=0.2)
    p.add_argument("--eval-top-p", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--clip-low-threshold", type=float, default=0.9)
    p.add_argument("--clip-high-threshold", type=float, default=1.1)
    p.add_argument("--sync-every", type=int, default=5)
    p.add_argument("--episodes", type=int, default=0, help="0 => run until convergence")
    p.add_argument("--max-episodes", type=int, default=2000)
    p.add_argument("--window", type=int, default=200)
    p.add_argument("--target-success-rate", type=float, default=0.5)
    p.add_argument("--min-episodes", type=int, default=200)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--run-dir", type=str, default="")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args(argv)

    _require_env("TINKER_API_KEY")

    from tinker import AdamParams, ModelInput, SamplingParams, ServiceClient
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model), trust_remote_code=True)

    run_id = _utc_run_id()
    run_dir = (
        Path(args.run_dir).expanduser()
        if str(args.run_dir).strip()
        else (Path("runs") / "swebench_lite_patch_tinker_rl" / run_id)
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    instance_ids = [s.strip() for s in str(args.instance_ids).split(",") if s.strip()]

    env_id = "local://swebench-lite-patch-v0"
    ew.register(
        env_id,
        factory=lambda: ew.examples.SWEbenchLitePatchEnv(
            split=str(args.train_split),
            seed=int(args.seed),
            max_examples=int(args.max_examples) if int(args.max_examples) > 0 else None,
            instance_ids=(instance_ids or None),
            max_fail_tests=int(args.max_fail_tests),
            timeout_s=float(args.env_timeout_s),
        ),
        docker_image=str(args.docker_image),
        observation_type=ew.examples.SWEbenchLitePatchObs,
        action_type=ew.examples.SWEbenchLitePatchAction,
        overwrite=True,
    )

    docker_extra_args: list[str] = []
    if str(args.backend) == "docker_http" and bool(args.docker_sock):
        docker_extra_args = ["-v", "/var/run/docker.sock:/var/run/docker.sock"]

    vector_kwargs: dict[str, Any] = {}
    if str(args.backend) == "docker_http":
        vector_kwargs["docker_extra_args"] = list(docker_extra_args)

    venv = ew.make_vector(
        env_id,
        num_envs=int(args.num_envs),
        backend=str(args.backend),
        autoreset=True,
        max_workers=int(args.num_envs),
        **vector_kwargs,
    )
    rr = venv.reset(seed=int(args.seed))
    current_obs: list[ew.examples.SWEbenchLitePatchObs] = list(rr.obs)

    service = ServiceClient()
    training_client = service.create_lora_training_client(
        base_model=str(args.base_model),
        rank=int(args.rank),
        seed=int(args.seed),
    )

    sampler_name_base = f"swebench_lite_patch_rl_{run_id}"
    init_sampler_path = training_client.save_weights_for_sampler(f"{sampler_name_base}_000000").result().path
    sampling_client = service.create_sampling_client(model_path=init_sampler_path)

    eval_env_id = "local://swebench-lite-patch-eval-v0"
    if str(args.eval_split) in ("dev", "test") and int(args.eval_episodes) > 0:
        ew.register(
            eval_env_id,
            factory=lambda: ew.examples.SWEbenchLitePatchEnv(
                split=str(args.eval_split),
                seed=int(args.seed) + 1337,
                max_examples=int(args.max_examples) if int(args.max_examples) > 0 else None,
                instance_ids=(instance_ids or None),
                max_fail_tests=int(args.max_fail_tests),
                timeout_s=float(args.env_timeout_s),
            ),
            docker_image=str(args.docker_image),
            observation_type=ew.examples.SWEbenchLitePatchObs,
            action_type=ew.examples.SWEbenchLitePatchAction,
            overwrite=True,
        )
        base_eval_avg_reward, base_eval_success = _eval_patch(
            env_id=eval_env_id,
            backend=str(args.backend),
            docker_extra_args=list(docker_extra_args),
            num_envs=min(int(args.num_envs), 2),
            seed=int(args.seed) + 4242,
            sampling_client=sampling_client,
            tokenizer=tokenizer,
            max_tokens=int(args.max_tokens),
            temperature=float(args.eval_temperature),
            top_p=float(args.eval_top_p),
            episodes=int(args.eval_episodes),
        )
        print(
            f"eval(base): split={args.eval_split} episodes={args.eval_episodes} "
            f"avg_reward={base_eval_avg_reward:.3f} success_rate={base_eval_success:.3f}"
        )
    else:
        base_eval_avg_reward = None
        base_eval_success = None

    baseline = 0.0
    window = max(1, int(args.window))
    rewards_window: deque[float] = deque(maxlen=window)
    success_window: deque[float] = deque(maxlen=window)
    episodes_done = 0
    updates_done = 0

    t0 = time.perf_counter()
    with metrics_path.open("w", encoding="utf-8") as f_metrics:
        while True:
            if int(args.episodes) > 0 and episodes_done >= int(args.episodes):
                break
            if int(args.episodes) <= 0 and episodes_done >= int(args.max_episodes):
                break

            prompt_tokens_list = [tokenizer.encode(o.prompt, add_special_tokens=True) for o in current_obs]

            sample_futures = []
            for i, pt in enumerate(prompt_tokens_list):
                params = SamplingParams(
                    max_tokens=int(args.max_tokens),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    seed=int(args.seed) + episodes_done + i,
                )
                sample_futures.append(
                    sampling_client.sample(
                        ModelInput.from_ints(list(pt)),
                        num_samples=1,
                        sampling_params=params,
                        include_prompt_logprobs=True,
                    )
                )
            sample_responses = [f.result() for f in sample_futures]

            actions: list[dict[str, Any]] = []
            full_tokens_list: list[list[int]] = []
            prompt_len_list: list[int] = []
            sampling_logprobs_list: list[list[float]] = []

            for pt, resp in zip(prompt_tokens_list, sample_responses):
                seq = resp.sequences[0]
                gen_tokens = list(seq.tokens)
                gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                patch = _extract_patch(gen_text)
                actions.append({"patch": patch})

                prompt_logprobs = [0.0 if v is None else float(v) for v in (resp.prompt_logprobs or [])]
                if len(prompt_logprobs) != len(pt):
                    prompt_logprobs = [0.0] * len(pt)
                gen_logprobs = [float(v) for v in (seq.logprobs or [])]
                if len(gen_logprobs) != len(gen_tokens):
                    gen_logprobs = [0.0] * len(gen_tokens)

                full_tokens = list(pt) + gen_tokens
                full_logprobs = prompt_logprobs + gen_logprobs

                full_tokens_list.append(full_tokens)
                prompt_len_list.append(len(pt))
                sampling_logprobs_list.append(full_logprobs)

            batch = venv.step(actions)
            batch_rewards = [float(r) for r in batch.reward]  # type: ignore[arg-type]
            batch_correct = [bool((batch.info[i] or {}).get("correct", False)) for i in range(int(args.num_envs))]
            batch_instance_ids = [
                str((batch.info[i] or {}).get("instance_id", "")) for i in range(int(args.num_envs))
            ]

            mean_reward = float(sum(batch_rewards) / max(len(batch_rewards), 1))
            baseline = 0.95 * baseline + 0.05 * mean_reward
            adv_arr = np.array(batch_rewards, dtype=np.float32) - float(baseline)
            adv_arr = adv_arr - float(np.mean(adv_arr))
            std = float(np.std(adv_arr))
            if std > 1e-6:
                adv_arr = adv_arr / std
            advantages = [float(x) for x in adv_arr.tolist()]

            data = [
                _build_datum(
                    full_tokens=full_tokens_list[i],
                    sampling_logprobs=sampling_logprobs_list[i],
                    prompt_len=prompt_len_list[i],
                    advantage=advantages[i],
                )
                for i in range(int(args.num_envs))
            ]

            training_client.forward_backward(
                data,
                loss_fn="ppo",
                loss_fn_config={
                    "clip_low_threshold": float(args.clip_low_threshold),
                    "clip_high_threshold": float(args.clip_high_threshold),
                },
            ).result()
            training_client.optim_step(
                AdamParams(
                    learning_rate=float(args.lr),
                    grad_clip_norm=float(args.grad_clip),
                )
            ).result()
            updates_done += 1

            if int(args.sync_every) > 0 and updates_done % int(args.sync_every) == 0:
                sampling_client = service.create_sampling_client(
                    model_path=training_client.save_weights_for_sampler(
                        f"{sampler_name_base}_{updates_done:06d}"
                    ).result().path
                )

            for i in range(int(args.num_envs)):
                episodes_done += 1
                reward = float(batch_rewards[i])
                correct = bool(batch_correct[i])
                rewards_window.append(reward)
                success_window.append(1.0 if correct else 0.0)

                avg_reward = sum(rewards_window) / max(len(rewards_window), 1)
                success_rate = sum(success_window) / max(len(success_window), 1)
                elapsed = time.perf_counter() - t0

                f_metrics.write(
                    json.dumps(
                        {
                            "episode": episodes_done,
                            "reward": reward,
                            "correct": correct,
                            "baseline": baseline,
                            "avg_reward": avg_reward,
                            "success_rate": success_rate,
                            "elapsed_s": elapsed,
                            "instance_id": batch_instance_ids[i],
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
                avg_reward = sum(rewards_window) / max(len(rewards_window), 1)
                success_rate = sum(success_window) / max(len(success_window), 1)
                print(
                    f"episodes={episodes_done} avg_reward={avg_reward:.3f} "
                    f"success_rate={success_rate:.3f} baseline={baseline:.3f} updates={updates_done}"
                )

            if (
                int(args.episodes) <= 0
                and episodes_done >= int(args.min_episodes)
                and len(success_window) >= window
                and (sum(success_window) / len(success_window)) >= float(args.target_success_rate)
            ):
                break

    elapsed = time.perf_counter() - t0
    venv.close()

    final_sampler_path = training_client.save_weights_for_sampler(
        f"{sampler_name_base}_{updates_done:06d}_final"
    ).result().path

    if str(args.eval_split) in ("dev", "test") and int(args.eval_episodes) > 0:
        trained_sampling_client = service.create_sampling_client(model_path=final_sampler_path)
        trained_eval_avg_reward, trained_eval_success = _eval_patch(
            env_id=eval_env_id,
            backend=str(args.backend),
            docker_extra_args=list(docker_extra_args),
            num_envs=min(int(args.num_envs), 2),
            seed=int(args.seed) + 4242,
            sampling_client=trained_sampling_client,
            tokenizer=tokenizer,
            max_tokens=int(args.max_tokens),
            temperature=float(args.eval_temperature),
            top_p=float(args.eval_top_p),
            episodes=int(args.eval_episodes),
        )
        print(
            f"eval(trained): split={args.eval_split} episodes={args.eval_episodes} "
            f"avg_reward={trained_eval_avg_reward:.3f} success_rate={trained_eval_success:.3f}"
        )
    else:
        trained_eval_avg_reward = None
        trained_eval_success = None

    episodes: list[int] = []
    rewards: list[float] = []
    success: list[float] = []
    baseline_series: list[float] = []
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))
            success.append(1.0 if bool(row.get("correct")) else 0.0)
            baseline_series.append(float(row["baseline"]))

    png_path = run_dir / "metrics.png"
    if episodes and not bool(args.no_plot):
        png_path = _save_plots(
            run_dir,
            episodes=episodes,
            rewards=rewards,
            success=success,
            baseline=baseline_series,
        )

    avg_reward = sum(rewards_window) / max(len(rewards_window), 1)
    success_rate = sum(success_window) / max(len(success_window), 1)
    eps_per_s = episodes_done / max(elapsed, 1e-9)

    summary = _RunSummary(
        episodes=int(episodes_done),
        avg_reward=float(avg_reward),
        success_rate=float(success_rate),
        eps_per_s=float(eps_per_s),
        backend=str(args.backend),
        run_dir=str(run_dir),
        metrics_jsonl=str(metrics_path),
        metrics_png=str(png_path),
        base_model=str(args.base_model),
        lora_rank=int(args.rank),
        train_split=str(args.train_split),
        train_instance_ids=(instance_ids or None),
        eval_split=(None if str(args.eval_split) == "none" else str(args.eval_split)),
        eval_episodes=(None if str(args.eval_split) == "none" else int(args.eval_episodes)),
        eval_base_avg_reward=(None if base_eval_avg_reward is None else float(base_eval_avg_reward)),
        eval_base_success_rate=(None if base_eval_success is None else float(base_eval_success)),
        eval_trained_avg_reward=(None if trained_eval_avg_reward is None else float(trained_eval_avg_reward)),
        eval_trained_success_rate=(None if trained_eval_success is None else float(trained_eval_success)),
    )
    (run_dir / "summary.json").write_text(json.dumps(summary.__dict__, indent=2) + "\n", encoding="utf-8")
    print(f"done: {json.dumps(summary.__dict__)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
