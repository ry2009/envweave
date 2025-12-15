from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import envweave as ew


class _Handler(BaseHTTPRequestHandler):
    env = ew.examples.CounterEnv(done_at=2)

    def log_message(self, format, *args):  # noqa: A002
        return

    def _read_json(self):
        length = int(self.headers.get("content-length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _write_json(self, obj):
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
            rr = self.env.reset(seed=payload.get("seed"), options=payload.get("options"))
            return self._write_json({"obs": {"value": rr.obs.value}, "info": rr.info, "requests": []})
        if self.path == "/step":
            payload = self._read_json()
            sr = self.env.step(payload.get("action"))
            return self._write_json(
                {
                    "obs": {"value": sr.obs.value},
                    "reward": sr.reward,
                    "done": sr.done,
                    "info": sr.info,
                    "requests": [],
                }
            )
        if self.path == "/close":
            self._read_json()
            return self._write_json({"ok": True})
        self.send_response(404)
        self.end_headers()


def test_http_backend_uses_registered_base_url():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}"
        ew.register(
            "local://counter-http-meta-v0",
            factory=lambda: ew.examples.CounterEnv(),
            base_url=base_url,
            observation_type=ew.examples.CounterObs,
            action_type=ew.examples.CounterAction,
            overwrite=True,
        )

        env = ew.make("local://counter-http-meta-v0", backend="docker_http", autoreset=True)
        rr = env.reset()
        assert rr.obs.value == 0
        sr = env.step({"delta": 1})
        assert sr.obs.value == 1
        env.close()
    finally:
        server.shutdown()
        server.server_close()

