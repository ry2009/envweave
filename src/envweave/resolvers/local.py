from __future__ import annotations

from envweave.ids import parse_env_id
from envweave.registry import DEFAULT_REGISTRY, EnvSpec, Registry


class LocalRegistryResolver:
    def __init__(self, registry: Registry | None = None) -> None:
        self._registry = registry or DEFAULT_REGISTRY

    def can_resolve(self, env_id: str) -> bool:
        parsed = parse_env_id(env_id)
        if parsed.scheme not in (None, "local"):
            return False
        return self._registry.has(parsed.locator if parsed.scheme == "local" else env_id) or (
            parsed.scheme == "local" and self._registry.has(env_id)
        )

    def resolve(self, env_id: str) -> EnvSpec:
        parsed = parse_env_id(env_id)
        if parsed.scheme == "local":
            # Accept either the full key (local://foo) or the locator (foo), depending on how it was registered.
            if self._registry.has(env_id):
                return self._registry.get(env_id)
            return self._registry.get(parsed.locator)
        return self._registry.get(env_id)

