from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, TypedDict

from envweave.types import ResetResult, StepResult


class ChatMessage(TypedDict):
    role: str
    content: str


def _messages_to_text(messages: Iterable[ChatMessage]) -> str:
    # Minimal, explicit format: one message per line.
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def _text_to_messages(text: str) -> list[ChatMessage]:
    # Best-effort round-trip: treat the full text as assistant content.
    return [{"role": "assistant", "content": text}]


def _map_maybe_dict(value: Any, fn):
    if isinstance(value, dict):
        return {k: fn(v) for k, v in value.items()}
    return fn(value)


class ChatWrapper:
    """
    Wraps a text-native env so callers can use OpenAI-style chat messages.
    """

    def __init__(self, env: Any) -> None:
        self._env = env

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> ResetResult:
        rr: ResetResult = self._env.reset(seed=seed, options=options)
        obs = _map_maybe_dict(rr.obs, lambda t: _text_to_messages(str(t)))
        return replace(rr, obs=obs)

    def step(self, action: Any) -> StepResult:
        text_action = _map_maybe_dict(action, lambda msgs: _messages_to_text(msgs))
        sr: StepResult = self._env.step(text_action)
        obs = _map_maybe_dict(sr.obs, lambda t: _text_to_messages(str(t)))
        return replace(sr, obs=obs)

    def close(self) -> None:
        close = getattr(self._env, "close", None)
        if callable(close):
            close()

