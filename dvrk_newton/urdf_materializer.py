"""Expand and cache Newton-ready dVRK Virtual robot URDF files."""

from __future__ import annotations

import os
from pathlib import Path

from .errors import NewtonBackendError
from dvrk_simulator_base.urdf_materializer import (
    MaterializedUrdf,
    default_generated_root as _default_generated_root,
    materialize_virtual_robot as _materialize_virtual_robot,
)


def default_generated_root() -> Path:
    """Return the user cache directory for Newton artifacts."""
    return _default_generated_root("dvrk_newton")


def materialize_virtual_robot(
    model: str,
    *,
    instrument: str | None = None,
    endoscope: str | None = None,
    parent_link: str = "world",
    generated_root: str | Path | None = None,
    dvrk_model_root: str | Path | None = None,
) -> MaterializedUrdf:
    """Expand and cache a standalone URDF with absolute mesh paths for Newton."""
    return _materialize_virtual_robot(
        model,
        instrument=instrument,
        endoscope=endoscope,
        parent_link=parent_link,
        generated_root=generated_root or default_generated_root(),
        dvrk_model_root=dvrk_model_root,
        error_cls=NewtonBackendError,
    )
