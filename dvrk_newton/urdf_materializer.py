"""Expand and cache Newton-ready dVRK Virtual robot URDF files."""

from __future__ import annotations

import os
from pathlib import Path

from .errors import NewtonBackendError
from dvrk_simulator_base.urdf_materializer import (
    MATERIALIZER_VERSION,
    SUPPORTED_PSMS,
    SUPPORTED_ROBOTS,
    MaterializedUrdf,
    materialize_virtual_robot as _materialize_virtual_robot,
)


def default_generated_root(anchor: str | Path | None = None) -> Path:
    """Return the user cache directory for Newton artifacts."""
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return (cache_root / "dvrk_newton").resolve()


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


def materialize_virtual_psm(
    model: str = "PSM1",
    instrument: str = "420006",
    parent_link: str = "world",
    generated_root: str | Path | None = None,
) -> MaterializedUrdf:
    """Compatibility wrapper for the original PSM-only entry point."""
    return materialize_virtual_robot(
        model,
        instrument=instrument,
        parent_link=parent_link,
        generated_root=generated_root,
    )
