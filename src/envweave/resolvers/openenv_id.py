from __future__ import annotations

from typing import Any

from envweave.ids import parse_env_id
from envweave.registry import EnvSpec


def _unsupported_factory(**_kwargs: Any):
    raise RuntimeError(
        "openenv:// env ids are not instantiable in-process; use backend='docker_http' "
        "and provide either base_url=... or docker_image=...."
    )


class OpenEnvIdResolver:
    def can_resolve(self, env_id: str) -> bool:
        parsed = parse_env_id(env_id)
        return parsed.scheme == "openenv"

    def resolve(self, env_id: str) -> EnvSpec:
        parsed = parse_env_id(env_id)
        if parsed.scheme != "openenv":
            raise ValueError(f"not an openenv env_id: {env_id}")
        return EnvSpec(
            env_id=env_id,
            factory=_unsupported_factory,
            metadata={"resolver": "openenv", "openenv": parsed.locator},
        )

