from __future__ import annotations

import importlib
import subprocess
from dataclasses import dataclass
from typing import Any, Callable

from envweave.ids import parse_env_id
from envweave.registry import EnvSpec


@dataclass(frozen=True)
class PrimeEnvRef:
    raw: str
    owner_and_name: str
    version: str | None
    module_name: str


def _parse_prime_locator(locator: str) -> PrimeEnvRef:
    raw = locator.strip()
    if not raw:
        raise ValueError("prime env_id locator is empty")

    if "@" in raw:
        owner_and_name, version = raw.split("@", 1)
        owner_and_name = owner_and_name.strip()
        version = version.strip() or None
    else:
        owner_and_name, version = raw, None

    # verifiers convention: module name = last segment, "-" -> "_"
    name = owner_and_name.split("/")[-1]
    module_name = name.replace("-", "_")
    return PrimeEnvRef(
        raw=raw,
        owner_and_name=owner_and_name,
        version=version,
        module_name=module_name,
    )


class PrimeHubResolver:
    def __init__(self, *, auto_install: bool = True) -> None:
        self.auto_install = auto_install

    def can_resolve(self, env_id: str) -> bool:
        parsed = parse_env_id(env_id)
        if parsed.scheme not in (None, "prime"):
            return False
        # Only claim "can resolve" for explicit prime://... or obvious owner/name patterns.
        if parsed.scheme == "prime":
            return True
        return "/" in parsed.locator

    def resolve(self, env_id: str) -> EnvSpec:
        parsed = parse_env_id(env_id)
        if parsed.scheme not in (None, "prime"):
            raise ValueError(f"not a prime env_id: {env_id}")

        ref = _parse_prime_locator(parsed.locator)

        module: Any | None = None
        try:
            module = importlib.import_module(ref.module_name)
        except ImportError:
            if not self.auto_install:
                raise
            install_target = ref.owner_and_name
            if ref.version:
                install_target = f"{install_target}@{ref.version}"
            subprocess.run(
                ["prime", "env", "install", install_target],
                check=True,
                capture_output=True,
                text=True,
            )
            module = importlib.import_module(ref.module_name)

        load = getattr(module, "load_environment", None)
        if load is None:
            load = getattr(module, "make", None)
        if load is None:
            raise AttributeError(
                f"prime environment module {ref.module_name!r} has neither "
                "'load_environment' nor 'make'"
            )
        if not callable(load):
            raise TypeError(
                f"prime environment module {ref.module_name!r} load callable is not callable"
            )
        factory: Callable[..., Any] = load
        return EnvSpec(
            env_id=env_id,
            factory=factory,
            metadata={
                "resolver": "prime",
                "prime": ref.owner_and_name,
                "version": ref.version,
                "module": ref.module_name,
            },
        )

