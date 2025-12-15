from __future__ import annotations

import argparse
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import envweave as ew


@dataclass(frozen=True)
class _Doc:
    doc_id: str
    text: str


@dataclass(frozen=True)
class _Query:
    query_id: str
    text: str
    relevant_doc_ids: list[str]


def _default_tinker_path() -> Path:
    # Best-effort default for this repo layout.
    return (Path.cwd() / ".." / "tinker-mxb").resolve()


def _load_tinker_modules(tinker_path: Path):
    if not tinker_path.exists():
        raise FileNotFoundError(f"tinker-mxb path does not exist: {tinker_path}")

    sys.path.insert(0, str(tinker_path))
    try:
        from data_loader import Document, QueryExample  # type: ignore[import-not-found]
        from env.search_env import SearchEnv  # type: ignore[import-not-found]
        from pipeline_configs import default_pipeline_configs  # type: ignore[import-not-found]
        from search_backend.mixedbread_client import (  # type: ignore[import-not-found]
            InMemorySearchBackend,
            MixedbreadClient,
        )
    finally:
        # Keep tinker on sys.path for the lifetime of the process; the imported modules
        # refer to each other by absolute top-level names.
        pass

    return Document, QueryExample, SearchEnv, default_pipeline_configs, InMemorySearchBackend, MixedbreadClient


def _make_synth_data(num_docs: int, num_queries: int, *, rng: random.Random):
    docs = [_Doc(doc_id=f"d{i}", text=f"doc {i} about topic {i%10}") for i in range(num_docs)]

    queries: list[_Query] = []
    for i in range(num_queries):
        topic = i % 10
        rels = [d.doc_id for d in docs if f"topic {topic}" in d.text][:3]
        queries.append(_Query(query_id=f"q{i}", text=f"query about topic {topic}", relevant_doc_ids=rels))
    return docs, queries


def _env_factory_for_tinker(
    *,
    tinker_path: Path,
    num_docs: int,
    num_queries: int,
    k: int,
    shuffle: bool,
    seed: int,
) -> Any:
    Document, QueryExample, SearchEnv, default_pipeline_configs, InMemorySearchBackend, MixedbreadClient = (
        _load_tinker_modules(tinker_path)
    )

    rng = random.Random(seed)
    docs_s, queries_s = _make_synth_data(num_docs, num_queries, rng=rng)

    # Always run fully offline: MixedbreadClient falls back to deterministic hash embeddings
    # when api_key is not provided.
    client = MixedbreadClient(api_key=None)

    docs = [Document(doc_id=d.doc_id, text=d.text) for d in docs_s]
    queries = [
        QueryExample(query_id=q.query_id, text=q.text, relevant_doc_ids=q.relevant_doc_ids)
        for q in queries_s
    ]

    backend = InMemorySearchBackend(docs, client)
    configs = default_pipeline_configs()
    return SearchEnv(queries=queries, configs=configs, backend=backend, k=k, shuffle=shuffle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark envweave over tinker-mxb SearchEnv.")
    parser.add_argument("--tinker-path", type=str, default=os.environ.get("TINKER_MXB_PATH") or str(_default_tinker_path()))
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--num-docs", type=int, default=200)
    parser.add_argument("--num-queries", type=int, default=256)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    tinker_path = Path(args.tinker_path).expanduser().resolve()
    env_id = "local://tinker-search-v0"

    ew.register(
        env_id,
        factory=lambda: _env_factory_for_tinker(
            tinker_path=tinker_path,
            num_docs=int(args.num_docs),
            num_queries=int(args.num_queries),
            k=int(args.k),
            shuffle=bool(args.shuffle),
            seed=int(args.seed),
        ),
        overwrite=True,
    )

    venv = ew.make_vector(env_id, num_envs=int(args.num_envs), autoreset=True, max_workers=int(args.num_envs))
    venv.reset()

    rng = random.Random(int(args.seed))
    num_actions = 6  # default_pipeline_configs() returns 6 configs

    start = time.perf_counter()
    total_steps = int(args.steps)
    for _ in range(total_steps):
        actions = [rng.randrange(num_actions) for _ in range(int(args.num_envs))]
        venv.step(actions)
    elapsed = time.perf_counter() - start

    venv.close()

    sps = (total_steps * int(args.num_envs)) / max(elapsed, 1e-9)
    print(
        f"tinker-mxb SearchEnv (offline): envs={args.num_envs} total_steps={total_steps} "
        f"elapsed_s={elapsed:.3f} steps_per_s={sps:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
