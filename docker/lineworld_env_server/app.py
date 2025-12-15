from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer


@dataclass
class State:
    size: int
    max_steps: int
    rng: random.Random
    pos: int = 0
    goal: int = 0
    t: int = 0


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return int(default)


SIZE = _int_env("SIZE", 5)
MAX_STEPS = _int_env("MAX_STEPS", SIZE * 4)
STATE = State(size=SIZE, max_steps=MAX_STEPS, rng=random.Random())


def _reset(seed: int | None = None) -> dict:
    if seed is not None:
        STATE.rng.seed(int(seed))
    STATE.pos = 0
    STATE.t = 0
    STATE.goal = STATE.size if STATE.rng.random() < 0.5 else -STATE.size
    return {"obs": {"pos": STATE.pos, "goal": STATE.goal, "t": STATE.t}, "info": {}, "requests": []}


def _step(move: int) -> dict:
    if move not in (-1, 1):
        raise ValueError("move must be -1 or +1")

    STATE.t += 1
    STATE.pos = max(-STATE.size, min(STATE.size, STATE.pos + move))

    terminated = STATE.pos == STATE.goal
    truncated = STATE.t >= STATE.max_steps and not terminated
    done = terminated or truncated

    reward = 1.0 if terminated else -0.01
    if truncated:
        reward -= 1.0

    info = {"terminated": terminated, "truncated": truncated}
    return {
        "obs": {"pos": STATE.pos, "goal": STATE.goal, "t": STATE.t},
        "reward": float(reward),
        "done": bool(done),
        "info": info,
        "requests": [],
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return

    def _read_json(self) -> dict:
        length = int(self.headers.get("content-length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except Exception:
            parsed = {}
        return parsed if isinstance(parsed, dict) else {}

    def _write_json(self, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            return self._write_json({"ok": True})
        self.send_response(404)
        self.end_headers()

    def do_POST(self):  # noqa: N802
        if self.path == "/reset":
            payload = self._read_json()
            seed = payload.get("seed")
            return self._write_json(_reset(seed=seed))

        if self.path == "/step":
            payload = self._read_json()
            action = payload.get("action") or {}
            if isinstance(action, dict):
                move = int(action.get("move", 0))
            else:
                move = int(action)
            return self._write_json(_step(move))

        if self.path == "/close":
            self._read_json()
            return self._write_json({"ok": True})

        self.send_response(404)
        self.end_headers()


def main() -> None:
    _reset(seed=None)
    server = HTTPServer(("0.0.0.0", 8000), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()

