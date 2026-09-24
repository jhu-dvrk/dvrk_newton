"""Newton dependency boundary and backend bootstrap helpers."""

from __future__ import annotations

from types import ModuleType

from .errors import NewtonDependencyError


def load_newton() -> tuple[ModuleType, ModuleType]:
    """Load Newton and Warp lazily so package inspection does not require pip setup."""
    try:
        import warp as wp
        import newton
    except ImportError as error:
        raise NewtonDependencyError(
            "NVIDIA Newton or Warp could not be imported by this executable.\n"
            "From the workspace root, run the bootstrap script:\n\n"
            "  ./src/dvrk/dvrk_newton/scripts/bootstrap_venv.sh\n\n"
            "Then run the launch file or ensure DVRK_NEWTON_PYTHON is set."
        ) from error
    return newton, wp
