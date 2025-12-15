from __future__ import annotations

from typing import Any, Iterable

from envweave.backends.http import DockerHTTPBackend
from envweave.backends.inproc import InProcBackend
from envweave.env import EnvHandle
from envweave.ids import parse_env_id
from envweave.registry import DEFAULT_REGISTRY, EnvFactory, EnvSpec
from envweave.resolvers.base import Resolver
from envweave.resolvers.chain import resolve_env_spec
from envweave.vector import VectorEnv


def register(
    env_id: str,
    factory: EnvFactory,
    *,
    metadata: dict[str, Any] | None = None,
    docker_image: str | None = None,
    base_url: str | None = None,
    observation_type: type | None = None,
    action_type: type | None = None,
    overwrite: bool = False,
) -> None:
    DEFAULT_REGISTRY.register(
        env_id,
        factory,
        metadata=metadata,
        docker_image=docker_image,
        base_url=base_url,
        observation_type=observation_type,
        action_type=action_type,
        overwrite=overwrite,
    )


def make(
    env_id: str,
    *,
    backend: str = "auto",
    autoreset: bool = False,
    instance_id: str | None = None,
    resolvers: Iterable[Resolver] | None = None,
    **kwargs: Any,
) -> EnvHandle:
    spec: EnvSpec = resolve_env_spec(env_id, resolvers=resolvers)
    parsed = parse_env_id(env_id)

    if backend == "auto":
        backend = (
            "docker_http"
            if parsed.scheme in ("http", "https", "docker_http", "openenv")
            else "inproc"
        )

    if instance_id is None:
        instance_id = env_id

    if backend == "inproc":
        impl = InProcBackend().create(spec, autoreset=autoreset, env_kwargs=dict(kwargs))
        return EnvHandle(
            env_id=env_id,
            instance_id=instance_id,
            _impl=impl,
            observation_type=spec.observation_type,
            action_type=spec.action_type,
        )

    if backend == "docker_http":
        backend_keys = {
            "base_url",
            "docker_image",
            "container_port",
            "host",
            "timeout_s",
            "startup_timeout_s",
            "health_path",
            "docker_extra_args",
        }
        backend_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in backend_keys}
        if "base_url" not in backend_kwargs:
            if parsed.scheme in ("http", "https"):
                backend_kwargs["base_url"] = env_id
            else:
                base_url = (spec.metadata or {}).get("base_url")
                if base_url:
                    backend_kwargs["base_url"] = base_url
        if "docker_image" not in backend_kwargs:
            docker_image = (spec.metadata or {}).get("docker_image")
            if docker_image:
                backend_kwargs["docker_image"] = docker_image
        if kwargs:
            raise ValueError(f"unexpected kwargs for docker_http backend: {sorted(kwargs.keys())}")
        http_backend = DockerHTTPBackend(**backend_kwargs)
        impl = http_backend.create(spec, autoreset=autoreset, env_kwargs={})

        class _Managed:
            _envweave_autoreset = autoreset

            def __init__(self, env_impl, backend_obj):
                self._env_impl = env_impl
                self._backend_obj = backend_obj

            def reset(self, *, seed=None, options=None):
                return self._env_impl.reset(seed=seed, options=options)

            def step(self, action):
                return self._env_impl.step(action)

            def close(self):
                try:
                    self._env_impl.close()
                finally:
                    self._backend_obj.close()

            def observation_schema(self):
                return getattr(self._env_impl, "observation_schema", lambda: None)()

            def action_schema(self):
                return getattr(self._env_impl, "action_schema", lambda: None)()

        return EnvHandle(
            env_id=env_id,
            instance_id=instance_id,
            _impl=_Managed(impl, http_backend),
            observation_type=spec.observation_type,
            action_type=spec.action_type,
        )

    raise ValueError(f"unknown backend: {backend!r}")


def make_vector(
    env_id: str,
    num_envs: int,
    *,
    backend: str = "auto",
    autoreset: bool = False,
    resolvers: Iterable[Resolver] | None = None,
    max_workers: int | None = None,
    **kwargs: Any,
) -> VectorEnv:
    envs = [
        make(
            env_id,
            backend=backend,
            autoreset=autoreset,
            instance_id=f"{env_id}#{i}",
            resolvers=resolvers,
            **kwargs,
        )
        for i in range(int(num_envs))
    ]
    return VectorEnv(envs, max_workers=max_workers)
