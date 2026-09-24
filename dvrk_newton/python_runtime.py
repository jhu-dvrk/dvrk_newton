"""Runtime selection of the Python interpreter used for NVIDIA Newton."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys

from .urdf_materializer import default_generated_root


_CACHE_FILE = "python-runtime.json"


@dataclass(frozen=True)
class NewtonPython:
    path: Path
    source: str


def _imports_newton(python: Path) -> bool:
    if not python.is_file() or not os.access(python, os.X_OK):
        return False
    result = subprocess.run(
        [str(python), "-c", "import newton, warp"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _cache_path(generated_root: str | Path | None) -> Path:
    root = Path(generated_root or default_generated_root()).expanduser().resolve()
    return root / _CACHE_FILE


def _read_cached_python(cache_path: Path) -> Path | None:
    try:
        document = json.loads(cache_path.read_text(encoding="utf-8"))
        value = document.get("python")
        if not isinstance(value, str) or not value:
            return None
        return Path(value).expanduser().absolute()
    except (OSError, ValueError, TypeError):
        return None


def _save_python(cache_path: Path, python: Path) -> None:
    """Persist a workspace-local interpreter choice without touching source."""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(f".tmp-{os.getpid()}")
        temporary.write_text(
            json.dumps({"python": str(python)}, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(cache_path)
    except OSError:
        return


def resolve_newton_python(
    generated_root: str | Path | None = None,
) -> NewtonPython:
    """Select a Newton interpreter and record a portable workspace cache.

    1. An explicit ``DVRK_NEWTON_PYTHON`` environment variable override takes precedence.
    2. A valid saved interpreter in ``.generated/newton/python-runtime.json`` is reused.
    3. Workspace-local ``.venv-newton/bin/python3`` is checked if present.
    4. The current running Python interpreter is used if it can import newton.
    """
    cache_path = _cache_path(generated_root)
    configured = os.environ.get("DVRK_NEWTON_PYTHON", "").strip()
    if configured:
        candidate = Path(configured).expanduser().absolute()
        if not _imports_newton(candidate):
            raise RuntimeError(
                f"DVRK_NEWTON_PYTHON is '{candidate}', but it cannot import newton"
            )
        _save_python(cache_path, candidate)
        return NewtonPython(candidate, "DVRK_NEWTON_PYTHON")

    cached = _read_cached_python(cache_path)
    if cached is not None and _imports_newton(cached):
        return NewtonPython(cached, f"saved selection in {cache_path}")

    # Check for workspace .venv-newton
    source = Path(__file__).resolve()
    for parent in source.parents:
        cand = parent / ".venv-newton" / "bin" / "python3"
        if cand.is_file():
            venv_python = cand.absolute()
            if _imports_newton(venv_python):
                _save_python(cache_path, venv_python)
                return NewtonPython(venv_python, f"workspace venv at {venv_python}")

    current = Path(sys.executable).absolute()
    if _imports_newton(current):
        _save_python(cache_path, current)
        return NewtonPython(current, "current Python")

    raise RuntimeError(
        "NVIDIA Newton is unavailable in the current Python environment and no valid "
        f"saved interpreter was found in {cache_path}.\n"
        "Run the bootstrap script to create the Newton venv:\n"
        "  ./src/dvrk/dvrk_newton/scripts/bootstrap_venv.sh\n"
        "Or set DVRK_NEWTON_PYTHON to a Python interpreter with newton and warp-lang installed."
    )
