from __future__ import annotations

from typing import Iterable

from envweave.errors import ResolveError
from envweave.ids import parse_env_id
from envweave.registry import EnvSpec
from envweave.resolvers.base import Resolver
from envweave.resolvers.entrypoints import EntrypointResolver
from envweave.resolvers.http_url import HttpUrlResolver
from envweave.resolvers.local import LocalRegistryResolver
from envweave.resolvers.openenv_id import OpenEnvIdResolver
from envweave.resolvers.pkg import PkgResolver
from envweave.resolvers.prime import PrimeHubResolver


DEFAULT_RESOLVERS: list[Resolver] = [
    LocalRegistryResolver(),
    EntrypointResolver(),
    PrimeHubResolver(),
]


def resolve_env_spec(env_id: str, *, resolvers: Iterable[Resolver] | None = None) -> EnvSpec:
    parsed = parse_env_id(env_id)
    resolvers_list = list(resolvers or DEFAULT_RESOLVERS)

    # Explicit scheme: route to the matching resolver.
    if parsed.scheme is not None:
        if parsed.scheme == "local":
            return LocalRegistryResolver().resolve(env_id)
        if parsed.scheme in ("http", "https"):
            return HttpUrlResolver().resolve(env_id)
        if parsed.scheme == "openenv":
            return OpenEnvIdResolver().resolve(env_id)
        if parsed.scheme in ("ep", "entrypoint"):
            return EntrypointResolver().resolve(env_id)
        if parsed.scheme == "prime":
            return PrimeHubResolver().resolve(env_id)
        if parsed.scheme == "pkg":
            return PkgResolver().resolve(env_id)
        raise ResolveError(f"unknown env_id scheme: {parsed.scheme!r}")

    # Implicit: try chain.
    for r in resolvers_list:
        try:
            if r.can_resolve(env_id):
                return r.resolve(env_id)
        except Exception:
            # If a resolver claims it can resolve but fails, surface that error.
            raise
    raise ResolveError(f"could not resolve env_id: {env_id}")
