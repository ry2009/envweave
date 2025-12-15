from __future__ import annotations

from dataclasses import dataclass

import envweave as ew
from envweave.types import ResetResult, StepResult
from envweave.wrappers import ChatWrapper, TokenizerWrapper


class _DummyTok:
    def encode(self, text: str, **kwargs):
        del kwargs
        return [ord(c) for c in text]

    def decode(self, ids: list[int], **kwargs):
        del kwargs
        return "".join(chr(i) for i in ids)


@dataclass
class _TextEnv:
    def reset(self, *, seed=None, options=None):
        del seed, options
        return ResetResult(obs="hello", info={})

    def step(self, action):
        return StepResult(obs=f"echo:{action}", reward=0.0, done=False, info={})

    def close(self):
        return None


def test_tokenizer_wrapper():
    env = TokenizerWrapper(_TextEnv(), _DummyTok())
    rr = env.reset()
    assert rr.obs == [ord(c) for c in "hello"]
    sr = env.step([ord(c) for c in "ok"])
    assert sr.obs == [ord(c) for c in "echo:ok"]


def test_chat_wrapper():
    env = ChatWrapper(_TextEnv())
    rr = env.reset()
    assert rr.obs == [{"role": "assistant", "content": "hello"}]
    sr = env.step([{"role": "user", "content": "hi"}])
    assert sr.obs == [{"role": "assistant", "content": "echo:user: hi"}]

