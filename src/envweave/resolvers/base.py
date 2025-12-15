from __future__ import annotations

from typing import Protocol

from envweave.registry import EnvSpec


class Resolver(Protocol):
    def can_resolve(self, env_id: str) -> bool: ...

    def resolve(self, env_id: str) -> EnvSpec: ...

