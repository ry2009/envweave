from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedEnvId:
    original: str
    scheme: str | None
    locator: str


def parse_env_id(env_id: str) -> ParsedEnvId:
    env_id = str(env_id)
    if "://" not in env_id:
        return ParsedEnvId(original=env_id, scheme=None, locator=env_id)
    scheme, locator = env_id.split("://", 1)
    scheme = scheme.strip().lower() or None
    return ParsedEnvId(original=env_id, scheme=scheme, locator=locator)

