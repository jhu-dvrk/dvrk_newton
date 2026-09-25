"""Unit tests for dvrk_newton scene object loading and asset resolution."""

from pathlib import Path
import pytest

from ament_index_python.packages import get_package_share_directory
from dvrk_simulator_base.scene import SceneObject
from dvrk_newton.errors import NewtonBackendError
from dvrk_newton.scene_objects import add_scene_objects_to_builder, resolve_asset_uri


def test_resolve_asset_uri():
    uri = "package://dvrk_simulator_base/share/assets/table/table.urdf"
    resolved = resolve_asset_uri(uri)
    assert resolved.is_file()
    assert resolved.name == "table.urdf"

    # Non-existent package
    with pytest.raises(NewtonBackendError, match="could not locate package"):
        resolve_asset_uri("package://non_existent_pkg_12345/foo.urdf")

    # Non-existent file
    with pytest.raises(NewtonBackendError, match="does not exist"):
        resolve_asset_uri("package://dvrk_simulator_base/share/assets/table/non_existent.urdf")

    # Relative non-package path
    with pytest.raises(NewtonBackendError, match="must be package:// URI"):
        resolve_asset_uri("relative/path/to/asset.urdf")


def test_add_scene_objects_to_builder():
    import newton

    builder = newton.ModelBuilder()
    table_obj = SceneObject(
        name="table",
        asset="package://dvrk_simulator_base/share/assets/table/table.urdf",
        fixed=True,
        position=(0.0, 0.0, 0.03),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    cube_obj = SceneObject(
        name="cube",
        asset="package://dvrk_simulator_base/share/assets/cube/cube.urdf",
        fixed=False,
        position=(0.0, 0.0, 0.055),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )

    loaded = add_scene_objects_to_builder(builder, (table_obj, cube_obj))
    assert "table" in loaded
    assert "cube" in loaded

    assert loaded["table"].is_dynamic is False
    assert loaded["cube"].is_dynamic is True

    assert loaded["table"].initial_position == (0.0, 0.0, 0.03)
    assert loaded["cube"].initial_position == (0.0, 0.0, 0.055)
    assert len(loaded["table"].shape_indices) > 0
    assert len(loaded["cube"].shape_indices) > 0
