from __future__ import annotations

from dataclasses import replace
from typing import Any, Protocol

from envweave.types import ResetResult, StepResult


class Tokenizer(Protocol):
    def encode(self, text: str, **kwargs) -> list[int]: ...

    def decode(self, ids: list[int], **kwargs) -> str: ...


def _encode(tokenizer: Any, text: str, *, encode_kwargs: dict[str, Any]) -> list[int]:
    if hasattr(tokenizer, "encode") and callable(tokenizer.encode):
        return list(tokenizer.encode(text, **encode_kwargs))
    # HuggingFace-style: tokenizer(text)["input_ids"]
    out = tokenizer(text, **encode_kwargs)
    if isinstance(out, dict) and "input_ids" in out:
        return list(out["input_ids"])
    raise TypeError("tokenizer must provide encode() or be callable returning input_ids")


def _decode(tokenizer: Any, ids: list[int], *, decode_kwargs: dict[str, Any]) -> str:
    if hasattr(tokenizer, "decode") and callable(tokenizer.decode):
        return str(tokenizer.decode(list(ids), **decode_kwargs))
    raise TypeError("tokenizer must provide decode()")


def _map_maybe_dict(value: Any, fn):
    if isinstance(value, dict):
        return {k: fn(v) for k, v in value.items()}
    return fn(value)


class TokenizerWrapper:
    """
    Wraps a text-native env so callers can send/receive token IDs instead of strings.
    """

    def __init__(
        self,
        env: Any,
        tokenizer: Any,
        *,
        encode_kwargs: dict[str, Any] | None = None,
        decode_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._env = env
        self._tok = tokenizer
        self._encode_kwargs = dict(encode_kwargs or {})
        self._decode_kwargs = dict(decode_kwargs or {"skip_special_tokens": True})

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> ResetResult:
        rr: ResetResult = self._env.reset(seed=seed, options=options)
        obs = _map_maybe_dict(rr.obs, lambda t: _encode(self._tok, str(t), encode_kwargs=self._encode_kwargs))
        return replace(rr, obs=obs)

    def step(self, action: Any) -> StepResult:
        decoded_action = _map_maybe_dict(action, lambda ids: _decode(self._tok, list(ids), decode_kwargs=self._decode_kwargs))
        sr: StepResult = self._env.step(decoded_action)
        obs = _map_maybe_dict(sr.obs, lambda t: _encode(self._tok, str(t), encode_kwargs=self._encode_kwargs))
        return replace(sr, obs=obs)

    def close(self) -> None:
        close = getattr(self._env, "close", None)
        if callable(close):
            close()

