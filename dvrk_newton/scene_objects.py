"""Load backend-neutral scene objects into an NVIDIA Newton world."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ament_index_python.packages import get_package_share_directory

from dvrk_simulator_base.scene import SceneObject

from .errors import NewtonBackendError


@dataclass(frozen=True)
class LoadedSceneObject:
    spec: SceneObject
    body_index: int
    shape_indices: tuple[int, ...]
    initial_position: tuple[float, float, float]
    initial_orientation_xyzw: tuple[float, float, float, float]
    is_dynamic: bool


def resolve_asset_uri(asset: str) -> Path:
    """Resolve package://<package>/<relative-path> and absolute asset paths."""
    if asset.startswith("package://"):
        package, separator, relative = asset[len("package://") :].partition("/")
        if not package or not separator or not relative:
            raise NewtonBackendError(f"invalid scene asset URI {asset!r}")
        try:
            path = Path(get_package_share_directory(package)) / relative
        except Exception as error:
            raise NewtonBackendError(
                f"could not locate package for scene asset {asset!r}"
            ) from error
    else:
        path = Path(asset).expanduser()
        if not path.is_absolute():
            raise NewtonBackendError(
                f"scene asset must be package:// URI or absolute path: {asset!r}"
            )
    path = path.resolve()
    if not path.is_file():
        raise NewtonBackendError(f"scene asset does not exist: {path}")
    return path


def add_scene_objects_to_builder(
    builder: Any, objects: Iterable[SceneObject]
) -> dict[str, LoadedSceneObject]:
    """Add configured fixed or dynamic URDF objects into a Newton ModelBuilder."""
    import warp as wp

    loaded: dict[str, LoadedSceneObject] = {}
    for spec in objects:
        asset_path = resolve_asset_uri(spec.asset)
        initial_body_count = builder.body_count
        initial_shape_count = builder.shape_count

        pos = tuple(float(v) for v in spec.position)
        rot = tuple(float(v) for v in spec.orientation_xyzw)
        xform = wp.transform(pos, rot)

        is_dynamic = not spec.fixed
        builder.add_urdf(
            str(asset_path),
            xform=xform,
            floating=is_dynamic,
            enable_self_collisions=False,
        )

        new_body_count = builder.body_count
        new_shape_count = builder.shape_count
        if new_body_count == initial_body_count:
            raise NewtonBackendError(f"Newton could not load scene object {spec.name!r}")

        # In Newton, the body index for a single-link floating URDF is initial_body_count
        body_index = initial_body_count
        shape_indices = tuple(range(initial_shape_count, new_shape_count))

        loaded[spec.name] = LoadedSceneObject(
            spec=spec,
            body_index=body_index,
            shape_indices=shape_indices,
            initial_position=pos,
            initial_orientation_xyzw=rot,
            is_dynamic=is_dynamic,
        )
    return loaded
