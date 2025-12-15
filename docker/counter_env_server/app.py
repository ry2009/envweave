from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class CounterState:
    def __init__(self) -> None:
        self.value = 0


STATE = CounterState()


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
            self._read_json()
            STATE.value = 0
            return self._write_json({"obs": {"value": STATE.value}, "info": {}, "requests": []})

        if self.path == "/step":
            payload = self._read_json()
            action = payload.get("action") or {}
            if isinstance(action, dict):
                delta = int(action.get("delta", 0))
            else:
                delta = int(action)

            STATE.value += delta
            done = STATE.value >= 3
            reward = float(STATE.value)
            return self._write_json(
                {"obs": {"value": STATE.value}, "reward": reward, "done": done, "info": {}, "requests": []}
            )

        if self.path == "/close":
            self._read_json()
            return self._write_json({"ok": True})

        self.send_response(404)
        self.end_headers()


def main() -> None:
    server = HTTPServer(("0.0.0.0", 8000), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()

