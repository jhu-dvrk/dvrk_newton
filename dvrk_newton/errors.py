"""Exception types for the dVRK Newton backend."""

from __future__ import annotations


class NewtonBackendError(RuntimeError):
    """Base error for Newton backend failures."""


class NewtonDependencyError(NewtonBackendError):
    """Raised when Newton or Warp cannot be loaded."""


class GStreamerDependencyError(NewtonBackendError):
    """Raised when GStreamer or Unix-FD plugins are unavailable."""

