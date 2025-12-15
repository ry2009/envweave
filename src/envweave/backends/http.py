from __future__ import annotations

import dataclasses
import json
import socket
import subprocess
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

from envweave.errors import BackendError, ProtocolError
from envweave.registry import EnvSpec
from envweave.serialization import requests_from_json, to_jsonable
from envweave.types import ResetResult, StepResult


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _http_post_json(url: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = UrlRequest(
        url,
        data=body,
        method="POST",
        headers={"content-type": "application/json"},
    )
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
    except HTTPError as e:
        raise BackendError(f"HTTP {e.code} from {url}: {e.read().decode('utf-8', 'ignore')}") from e
    except URLError as e:
        raise BackendError(f"HTTP error calling {url}: {e}") from e

    try:
        decoded = json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise ProtocolError(f"invalid JSON response from {url}") from e
    if not isinstance(decoded, dict):
        raise ProtocolError(f"expected JSON object from {url}, got {type(decoded).__name__}")
    return decoded


def _http_get_json(url: str, *, timeout_s: float) -> dict[str, Any]:
    req = UrlRequest(
        url,
        method="GET",
        headers={"accept": "application/json"},
    )
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
    except HTTPError as e:
        raise BackendError(f"HTTP {e.code} from {url}: {e.read().decode('utf-8', 'ignore')}") from e
    except URLError as e:
        raise BackendError(f"HTTP error calling {url}: {e}") from e

    if not raw:
        return {}
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise ProtocolError(f"invalid JSON response from {url}") from e
    if not isinstance(decoded, dict):
        raise ProtocolError(f"expected JSON object from {url}, got {type(decoded).__name__}")
    return decoded


def _maybe_dataclass(cls: type | None, payload: Any) -> Any:
    if cls is None:
        return payload
    if not dataclasses.is_dataclass(cls):
        return payload
    if not isinstance(payload, dict):
        return payload
    try:
        return cls(**payload)
    except Exception:
        return payload


@dataclass
class _DockerContainer:
    container_id: str
    base_url: str


class _HTTPEnvClient:
    def __init__(
        self,
        *,
        base_url: str,
        timeout_s: float,
        autoreset: bool,
        observation_type: type | None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._envweave_autoreset = autoreset
        self._observation_type = observation_type

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        payload: dict[str, Any] = {"seed": seed, "options": options}
        data = _http_post_json(f"{self._base_url}/reset", payload, timeout_s=self._timeout_s)
        obs = _maybe_dataclass(self._observation_type, data.get("obs"))
        info = data.get("info") or {}
        return ResetResult(obs=obs, info=dict(info), requests=requests_from_json(data.get("requests")))

    def step(self, action: Any):
        payload: dict[str, Any] = {"action": to_jsonable(action)}
        data = _http_post_json(f"{self._base_url}/step", payload, timeout_s=self._timeout_s)

        if "done" not in data and ("terminated" in data or "truncated" in data):
            terminated = bool(data.get("terminated"))
            truncated = bool(data.get("truncated"))
            data["done"] = terminated or truncated
            info = dict(data.get("info") or {})
            info.setdefault("terminated", terminated)
            info.setdefault("truncated", truncated)
            data["info"] = info

        if "obs" not in data or "reward" not in data or "done" not in data:
            raise ProtocolError("step response missing required keys: obs, reward, done")

        obs = _maybe_dataclass(self._observation_type, data.get("obs"))
        info = data.get("info") or {}
        return StepResult(
            obs=obs,
            reward=data.get("reward"),
            done=data.get("done"),
            info=dict(info),
            requests=requests_from_json(data.get("requests")),
        )

    def close(self) -> None:
        # Best-effort.
        try:
            _http_post_json(f"{self._base_url}/close", {}, timeout_s=self._timeout_s)
        except Exception:
            pass


class DockerHTTPBackend:
    name = "docker_http"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        docker_image: str | None = None,
        container_port: int = 8000,
        host: str = "127.0.0.1",
        timeout_s: float = 30.0,
        startup_timeout_s: float = 30.0,
        health_path: str | None = "/health",
        docker_extra_args: list[str] | None = None,
    ) -> None:
        if base_url is None and docker_image is None:
            raise ValueError("either base_url or docker_image must be provided")
        self._base_url = base_url
        self._docker_image = docker_image
        self._container_port = int(container_port)
        self._host = host
        self._timeout_s = float(timeout_s)
        self._startup_timeout_s = float(startup_timeout_s)
        self._health_path = health_path
        self._docker_extra_args = list(docker_extra_args or [])
        self._container: _DockerContainer | None = None

    def _ensure_container(self) -> str:
        if self._base_url:
            return self._base_url
        if self._container is not None:
            return self._container.base_url
        assert self._docker_image is not None

        host_port = _find_free_port()
        cmd = [
            "docker",
            "run",
            "-d",
            "-p",
            f"{self._host}:{host_port}:{self._container_port}",
            *self._docker_extra_args,
            self._docker_image,
        ]
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        container_id = proc.stdout.strip()
        base_url = f"http://{self._host}:{host_port}"
        self._container = _DockerContainer(container_id=container_id, base_url=base_url)

        if self._health_path:
            deadline = time.time() + self._startup_timeout_s
            while time.time() < deadline:
                try:
                    _http_get_json(
                        f"{base_url}{self._health_path}",
                        timeout_s=min(2.0, self._timeout_s),
                    )
                    break
                except Exception:
                    time.sleep(0.25)
        return base_url

    def create(
        self, spec: EnvSpec, *, autoreset: bool, env_kwargs: dict[str, Any]
    ) -> Any:
        if env_kwargs:
            raise ValueError("docker_http backend does not support env_kwargs (pass via server)")
        base_url = self._ensure_container()
        return _HTTPEnvClient(
            base_url=base_url,
            timeout_s=self._timeout_s,
            autoreset=autoreset,
            observation_type=spec.observation_type,
        )

    def close(self) -> None:
        if self._container is None:
            return
        cid = self._container.container_id
        self._container = None
        subprocess.run(["docker", "rm", "-f", cid], check=False, capture_output=True, text=True)
