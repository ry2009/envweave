from __future__ import annotations

from importlib import metadata
from typing import Any, Callable

from envweave.ids import parse_env_id
from envweave.registry import EnvSpec

ENTRYPOINT_GROUP = "envweave.envs"


class EntrypointResolver:
    def __init__(self, group: str = ENTRYPOINT_GROUP) -> None:
        self.group = group

    def can_resolve(self, env_id: str) -> bool:
        parsed = parse_env_id(env_id)
        if parsed.scheme not in (None, "ep", "entrypoint"):
            return False
        key = parsed.locator if parsed.scheme in ("ep", "entrypoint") else env_id
        try:
            eps = metadata.entry_points(group=self.group)
        except TypeError:
            # Python <3.10 compatibility; not expected here, but keep it robust.
            eps = metadata.entry_points().get(self.group, [])  # type: ignore[assignment]
        return any(ep.name == key for ep in eps)

    def resolve(self, env_id: str) -> EnvSpec:
        parsed = parse_env_id(env_id)
        key = parsed.locator if parsed.scheme in ("ep", "entrypoint") else env_id
        try:
            eps = metadata.entry_points(group=self.group)
        except TypeError:
            eps = metadata.entry_points().get(self.group, [])  # type: ignore[assignment]

        for ep in eps:
            if ep.name != key:
                continue
            loaded = ep.load()
            if isinstance(loaded, EnvSpec):
                return loaded
            if callable(loaded):
                factory: Callable[..., Any] = loaded
                return EnvSpec(
                    env_id=env_id,
                    factory=factory,
                    metadata={"resolver": "entrypoint", "entrypoint": f"{ep.module}:{ep.attr}"},
                )
            raise TypeError(
                f"entrypoint {self.group}:{ep.name} must return EnvSpec or a callable factory"
            )
        raise KeyError(f"no entrypoint found for env_id: {env_id}")

