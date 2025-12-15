from __future__ import annotations

from typing import Any

from envweave.ids import parse_env_id
from envweave.registry import EnvSpec


def _unsupported_factory(**_kwargs: Any):
    raise RuntimeError("HTTP URL env specs are not instantiable in-process; use backend='docker_http'.")


class HttpUrlResolver:
    def can_resolve(self, env_id: str) -> bool:
        parsed = parse_env_id(env_id)
        return parsed.scheme in ("http", "https")

    def resolve(self, env_id: str) -> EnvSpec:
        parsed = parse_env_id(env_id)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"not an http(s) env_id: {env_id}")
        return EnvSpec(
            env_id=env_id,
            factory=_unsupported_factory,
            metadata={"resolver": "http_url", "base_url": env_id},
        )

