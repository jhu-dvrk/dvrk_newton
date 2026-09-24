"""Runtime selection of the Python interpreter used for NVIDIA Newton."""

from __future__ import annotations

from pathlib import Path

from dvrk_simulator_base.python_runtime import (
    SimulatorPython,
    check_python_imports,
    resolve_simulator_python,
)
from .urdf_materializer import default_generated_root


class NewtonPython(SimulatorPython):
    pass


def _imports_newton(python: Path) -> bool:
    return check_python_imports(python, ("newton", "warp"))


def resolve_newton_python(
    generated_root: str | Path | None = None,
) -> NewtonPython:
    """Select a Newton interpreter and record a portable workspace cache.

    1. An explicit ``DVRK_NEWTON_PYTHON`` environment variable override takes precedence.
    2. A valid saved interpreter in ``.generated/newton/python-runtime.json`` is reused.
    3. Workspace-local ``.venv-newton/bin/python3`` is checked if present.
    4. The current running Python interpreter is used if it can import newton.
    """
    return resolve_simulator_python(
        simulator_name="NVIDIA Newton",
        env_var="DVRK_NEWTON_PYTHON",
        generated_root=generated_root,
        default_generated_root=default_generated_root(),
        check_import_fn=_imports_newton,
        workspace_venv_names=(".venv-newton", ".venv"),
        bootstrap_command="./src/dvrk/dvrk_newton/scripts/bootstrap_venv.sh",
        result_factory=NewtonPython,
        source_file=__file__,
    )
