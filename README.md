# envweave

`envweave` is a small Python library that:

- Resolves string environment IDs into constructors (local registry, entry points, Prime Hub).
- Runs environments **in-process** or over **HTTP (optionally via Docker)** with the same `reset()/step()` API.
- Treats **vectorization** and **multi-agent** outputs as first-class.
- Returns a **request/event stream** (`StepResult.requests`) to handle episode boundaries and turn-taking cleanly.

## Quickstart

```bash
uv venv --python 3.11
uv pip install -e '.[dev]'
pytest
```

```python
import envweave as ew

# Register a local env factory
ew.register("local://counter-v0", factory=lambda: ew.examples.CounterEnv())

env = ew.make("local://counter-v0", autoreset=True)
rr = env.reset()
sr = env.step({"delta": 1})
```

## RL training demo (in-proc)

```bash
uv pip install -e '.[train]'
uv run -m envweave.examples.train_lineworld_pg --num-envs 16 --episodes 400
```

Docker/HTTP variant (same API, slower):

```bash
docker build -t envweave-lineworld -f docker/lineworld_env_server/Dockerfile .
uv run -m envweave.examples.train_lineworld_pg --backend docker_http --num-envs 4 --episodes 200
```

## SWE-ish RL demo (tiny "LLM" policy)

`CodeFixEnv` is a single-step "choose the correct operator to fix the code" bandit. The demo trains a **tiny transformer policy from scratch** (2-layer encoder; no pretraining).

In-proc:

```bash
uv pip install -e '.[train,torch]'
uv run -m envweave.examples.train_codefix_pg --backend inproc --num-envs 32
```

Docker/HTTP (same semantics, env runs in a container; training stays local):

```bash
docker build -t envweave-codefix -f docker/codefix_env_server/Dockerfile .
uv run -m envweave.examples.train_codefix_pg --backend docker_http --docker-image envweave-codefix --num-envs 16
```

## Real base model RL via Tinker (GSM8K)

This demo uses a **real open-weight base model** (LoRA) via **Tinker** and a dataset-backed env (`GSM8KEnv`). It produces `metrics.jsonl` + `metrics.png` in `runs/`.

```bash
uv pip install -e '.[train,tinker]'
export TINKER_API_KEY=...
uv run -m envweave.examples.train_gsm8k_tinker_rl --backend inproc --num-envs 8
```

## Real base model RL via Tinker (GSM8K-MCQ)

This variant turns GSM8K into a 4-way multiple-choice env (dense reward, short outputs) which is useful for quick end-to-end RL wiring tests.

```bash
uv pip install -e '.[train,tinker]'
export TINKER_API_KEY=...
uv run -m envweave.examples.train_gsm8k_mcq_tinker_rl --backend inproc --num-envs 8 --episodes 800
```

Docker/HTTP env backend (same `reset()/step()` semantics):

```bash
docker build -t envweave-gsm8k -f docker/gsm8k_env_server/Dockerfile .
uv run -m envweave.examples.train_gsm8k_tinker_rl --backend docker_http --docker-image envweave-gsm8k --num-envs 8
```

## Real base model RL via Tinker (SWE-bench Lite localization MCQ)

SWE-bench Lite tasks include a gold patch (typically editing a single file). This environment turns the task into
**file-localization MCQ**: given the issue, pick which file should be edited.

```bash
uv pip install -e '.[train,tinker,swebench]'
export TINKER_API_KEY=...
uv run -m envweave.examples.train_swebench_lite_loc_mcq_tinker_rl --backend inproc --num-envs 8 --episodes 0 --target-success-rate 0.7
```

Docker/HTTP env backend (same semantics):

```bash
docker build -t envweave-swebench-lite-loc-mcq -f docker/swebench_lite_loc_mcq_env_server/Dockerfile .
uv run -m envweave.examples.train_swebench_lite_loc_mcq_tinker_rl --backend docker_http --docker-image envweave-swebench-lite-loc-mcq --num-envs 8
```

## Real base model RL via Tinker (SWE-bench Lite patch + tests)

This is the full “generate unified diff patch → run repo tests in Docker” loop using official SWE-bench instance
images. Each episode:

1) env emits an issue prompt
2) model generates a unified diff
3) env applies the instance `test_patch`, applies the model patch, then runs a subset of `FAIL_TO_PASS` tests

```bash
uv pip install -e '.[train,tinker,swebench]'
export TINKER_API_KEY=...
uv run -m envweave.examples.train_swebench_lite_patch_tinker_rl --backend inproc --num-envs 1 --episodes 0 --target-success-rate 0.5
```

Docker/HTTP env backend (same semantics; env container needs docker.sock):

```bash
docker build -t envweave-swebench-lite-patch -f docker/swebench_lite_patch_env_server/Dockerfile .
uv run -m envweave.examples.train_swebench_lite_patch_tinker_rl --backend docker_http --docker-image envweave-swebench-lite-patch --docker-sock --num-envs 1
```
