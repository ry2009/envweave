from __future__ import annotations


class EnvWeaveError(Exception):
    pass


class EnvNotFoundError(EnvWeaveError):
    pass


class ResolveError(EnvWeaveError):
    pass


class BackendError(EnvWeaveError):
    pass


class ProtocolError(EnvWeaveError):
    pass

