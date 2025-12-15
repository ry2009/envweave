from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from envweave.errors import EnvNotFoundError

EnvFactory = Callable[..., Any]


@dataclass(frozen=True)
class EnvSpec:
    env_id: str
    factory: EnvFactory
    metadata: dict[str, Any] = field(default_factory=dict)
    observation_type: type | None = None
    action_type: type | None = None


class Registry:
    def __init__(self) -> None:
        self._specs: dict[str, EnvSpec] = {}

    def register(
        self,
        env_id: str,
        factory: EnvFactory,
        *,
        metadata: dict[str, Any] | None = None,
        docker_image: str | None = None,
        base_url: str | None = None,
        observation_type: type | None = None,
        action_type: type | None = None,
        overwrite: bool = False,
    ) -> None:
        if (not overwrite) and env_id in self._specs:
            raise ValueError(f"env_id already registered: {env_id}")
        merged_metadata = dict(metadata or {})
        if docker_image is not None:
            merged_metadata.setdefault("docker_image", str(docker_image))
        if base_url is not None:
            merged_metadata.setdefault("base_url", str(base_url))
        self._specs[env_id] = EnvSpec(
            env_id=env_id,
            factory=factory,
            metadata=merged_metadata,
            observation_type=observation_type,
            action_type=action_type,
        )

    def get(self, env_id: str) -> EnvSpec:
        try:
            return self._specs[env_id]
        except KeyError as e:
            raise EnvNotFoundError(f"env_id not found in local registry: {env_id}") from e

    def has(self, env_id: str) -> bool:
        return env_id in self._specs

    def list(self) -> list[str]:
        return sorted(self._specs.keys())


DEFAULT_REGISTRY = Registry()
