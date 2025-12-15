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

## Benchmark (tinker-mxb)

If you have the sibling repo `../tinker-mxb` checked out, you can run a quick vectorized throughput benchmark:

```bash
uv pip install -e '.[bench]'
uv run --env-file .env -m envweave.bench.tinker_search_env --num-envs 8 --steps 1024
```
This benchmark always runs fully offline (tinker-mxb’s deterministic hash embeddings).
