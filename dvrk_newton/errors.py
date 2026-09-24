"""Exception types for the dVRK Newton backend."""

from __future__ import annotations

from dvrk_simulator_base.video import VideoSinkError, GStreamerDependencyError as _BaseGstError


class NewtonBackendError(VideoSinkError):
    """Base error for Newton backend failures."""


class NewtonDependencyError(NewtonBackendError):
    """Raised when Newton or Warp cannot be loaded."""


class GStreamerDependencyError(NewtonBackendError, _BaseGstError):
    """Raised when GStreamer or Unix-FD plugins are unavailable."""
