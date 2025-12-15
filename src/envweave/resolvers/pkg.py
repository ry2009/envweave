from __future__ import annotations

import importlib
from typing import Any, Callable

from envweave.ids import parse_env_id
from envweave.registry import EnvSpec


def _parse_module_attr(locator: str) -> tuple[str, str]:
    if ":" in locator:
        module, attr = locator.split(":", 1)
        module = module.strip()
        attr = attr.strip()
        if not module:
            raise ValueError("pkg locator missing module path")
        if not attr:
            raise ValueError("pkg locator missing attribute")
        return module, attr
    return locator.strip(), "load_environment"


class PkgResolver:
    def can_resolve(self, env_id: str) -> bool:
        parsed = parse_env_id(env_id)
        if parsed.scheme != "pkg":
            return False
        return True

    def resolve(self, env_id: str) -> EnvSpec:
        parsed = parse_env_id(env_id)
        if parsed.scheme != "pkg":
            raise ValueError(f"not a pkg env_id: {env_id}")

        module_name, attr = _parse_module_attr(parsed.locator)
        module = importlib.import_module(module_name)
        obj: Any = getattr(module, attr, None)
        if obj is None:
            raise AttributeError(f"{module_name!r} has no attribute {attr!r}")
        if not callable(obj):
            raise TypeError(f"{module_name}:{attr} is not callable")
        factory: Callable[..., Any] = obj
        return EnvSpec(
            env_id=env_id,
            factory=factory,
            metadata={"resolver": "pkg", "module": module_name, "attr": attr},
        )

