"""Expand and cache Newton-ready dVRK Virtual robot URDF files."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import xacro

from .errors import NewtonBackendError
from dvrk_simulator_base.model_source import locate_dvrk_model


SUPPORTED_PSMS = ("PSM1", "PSM2", "PSM3")
SUPPORTED_ROBOTS = (*SUPPORTED_PSMS, "ECM")
MATERIALIZER_VERSION = 1


@dataclass(frozen=True)
class MaterializedUrdf:
    model: str
    instrument: str | None
    endoscope: str | None
    source_path: Path
    model_root: Path
    urdf_path: Path
    metadata_path: Path
    content_hash: str


def default_generated_root(anchor: str | Path | None = None) -> Path:
    """Return the user cache directory for Newton artifacts."""
    import os
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return (cache_root / "dvrk_newton").resolve()



def _expand_virtual_robot(
    source: Path,
    parent_link: str,
    instrument: str | None,
    endoscope: str | None,
) -> str:
    mappings = {"parent_link_": parent_link, "show_rcm": "false"}
    if instrument is not None:
        mappings.update({"instrument": instrument, "is_virtual": "true"})
    if endoscope is not None:
        mappings["endoscope"] = endoscope
    try:
        document = xacro.process_file(
            str(source),
            mappings=mappings,
        )
    except Exception as error:
        raise NewtonBackendError(f"failed to expand {source}: {error}") from error
    return document.toxml()


def _resolved_urdf(urdf_text: str, model_root: Path) -> bytes:
    try:
        robot = ET.fromstring(urdf_text)
    except ET.ParseError as error:
        raise NewtonBackendError(f"expanded dvrk_model URDF is invalid XML: {error}") from error

    prefix = "package://dvrk_model/"
    for element in robot.iter():
        filename = element.attrib.get("filename")
        if not filename or not filename.startswith(prefix):
            continue
        relative = filename[len(prefix):]
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise NewtonBackendError(f"invalid dvrk_model resource path: {filename}")
        resolved = (model_root / relative).resolve()
        if not resolved.is_file():
            raise NewtonBackendError(f"dvrk_model resource does not exist: {resolved}")
        element.set("filename", str(resolved))

    return ET.tostring(robot, encoding="utf-8", xml_declaration=True)


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def materialize_virtual_robot(
    model: str,
    *,
    instrument: str | None = None,
    endoscope: str | None = None,
    parent_link: str = "world",
    generated_root: str | Path | None = None,
    dvrk_model_root: str | Path | None = None,
) -> MaterializedUrdf:
    """Expand and cache a standalone URDF with absolute mesh paths."""
    normalized_model = model.upper()
    if normalized_model not in SUPPORTED_ROBOTS:
        raise ValueError(f"model must be one of {SUPPORTED_ROBOTS}, got {model!r}")
    if normalized_model in SUPPORTED_PSMS:
        if instrument is None:
            instrument = "420006"
        if endoscope is not None:
            raise ValueError(f"{normalized_model} does not accept an endoscope parameter")
    else:
        if endoscope is None:
            endoscope = "Si_straight"
        if instrument is not None:
            raise ValueError("ECM does not accept an instrument parameter")

    model_root = locate_dvrk_model()
    source_path = model_root / "urdf" / "Virtual" / f"{normalized_model}.urdf.xacro"
    if not source_path.is_file():
        raise NewtonBackendError(f"missing dvrk_model source xacro: {source_path}")

    raw_urdf = _expand_virtual_robot(source_path, parent_link, instrument, endoscope)
    resolved = _resolved_urdf(raw_urdf, model_root)

    digest = hashlib.sha256()
    digest.update(str(MATERIALIZER_VERSION).encode("utf-8"))
    digest.update(b"\0")
    digest.update(normalized_model.encode("utf-8"))
    digest.update(b"\0")
    digest.update((instrument or "").encode("utf-8"))
    digest.update(b"\0")
    digest.update((endoscope or "").encode("utf-8"))
    digest.update(b"\0")
    digest.update(resolved)
    content_hash = digest.hexdigest()

    cache_dir = default_generated_root(generated_root) / content_hash
    urdf_path = cache_dir / "model.urdf"
    metadata_path = cache_dir / "metadata.json"

    if not urdf_path.is_file() or not metadata_path.is_file():
        _atomic_write(urdf_path, resolved)
        metadata = {
            "version": MATERIALIZER_VERSION,
            "model": normalized_model,
            "instrument": instrument,
            "endoscope": endoscope,
            "content_hash": content_hash,
            "source_path": str(source_path),
            "dvrk_model_root": str(model_root),
        }
        _atomic_write(
            metadata_path,
            json.dumps(metadata, indent=2).encode("utf-8") + b"\n",
        )

    return MaterializedUrdf(
        model=normalized_model,
        instrument=instrument,
        endoscope=endoscope,
        source_path=source_path,
        model_root=model_root,
        urdf_path=urdf_path,
        metadata_path=metadata_path,
        content_hash=content_hash,
    )
