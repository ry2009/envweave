from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import envweave as ew
from envweave.serialization import to_jsonable


def _read_json_from_handler(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("content-length", "0"))
    raw = handler.rfile.read(length) if length else b"{}"
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except Exception:
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}


def _write_json(handler: BaseHTTPRequestHandler, payload: Any, *, status: int = 200) -> None:
    body = json.dumps(to_jsonable(payload)).encode("utf-8")
    handler.send_response(status)
    handler.send_header("content-type", "application/json")
    handler.send_header("content-length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def serve(
    *,
    env_id: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    env_kwargs: dict[str, Any] | None = None,
) -> None:
    env = ew.make(env_id, backend="inproc", **(env_kwargs or {}))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:  # noqa: A002
            return

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                return _write_json(self, {"ok": True})
            if self.path == "/schema":
                return _write_json(
                    self,
                    {
                        "observation_schema": env.observation_schema(),
                        "action_schema": env.action_schema(),
                    },
                )
            self.send_response(404)
            self.end_headers()

        def do_POST(self):  # noqa: N802
            if self.path == "/reset":
                payload = _read_json_from_handler(self)
                rr = env.reset(seed=payload.get("seed"), options=payload.get("options"))
                return _write_json(
                    self, {"obs": rr.obs, "info": rr.info, "requests": rr.requests}
                )
            if self.path == "/step":
                payload = _read_json_from_handler(self)
                sr = env.step(payload.get("action"))
                return _write_json(
                    self,
                    {
                        "obs": sr.obs,
                        "reward": sr.reward,
                        "done": sr.done,
                        "info": sr.info,
                        "requests": sr.requests,
                    },
                )
            if self.path == "/close":
                _read_json_from_handler(self)
                env.close()
                return _write_json(self, {"ok": True})

            self.send_response(404)
            self.end_headers()

    HTTPServer((host, int(port)), Handler).serve_forever()


def _parse_env_kwargs_json(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception as e:
        raise ValueError("ENVWEAVE_ENV_KWARGS_JSON is not valid JSON") from e
    if not isinstance(parsed, dict):
        raise ValueError("ENVWEAVE_ENV_KWARGS_JSON must be a JSON object")
    return parsed


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Serve an envweave env over HTTP.")
    p.add_argument("--env-id", type=str, default=os.environ.get("ENVWEAVE_ENV_ID"))
    p.add_argument("--host", type=str, default=os.environ.get("ENVWEAVE_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.environ.get("ENVWEAVE_PORT", "8000")))
    p.add_argument(
        "--env-kwargs-json",
        type=str,
        default=os.environ.get("ENVWEAVE_ENV_KWARGS_JSON", ""),
        help="JSON object passed as kwargs when constructing the env in-proc.",
    )
    args = p.parse_args(argv)

    if not args.env_id:
        raise SystemExit("--env-id is required (or set ENVWEAVE_ENV_ID).")

    serve(
        env_id=str(args.env_id),
        host=str(args.host),
        port=int(args.port),
        env_kwargs=_parse_env_kwargs_json(str(args.env_kwargs_json)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

