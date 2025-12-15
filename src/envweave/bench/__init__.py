from __future__ import annotations


def tinker_search_main(argv: list[str] | None = None) -> int:
    from envweave.bench.tinker_search_env import main

    return main(argv)

__all__ = ["tinker_search_main"]
