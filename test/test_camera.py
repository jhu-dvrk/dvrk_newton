import numpy as np
import os
from pathlib import Path
import subprocess
import sys
import pytest

from dvrk_simulator_base.scene import SceneCamera
from dvrk_simulator_base.types import Pose
from dvrk_newton.camera import CameraOptions, NewtonCameraRenderer, VideoFrame
from dvrk_newton.camera_gl import NewtonOpenGLCameraRenderer
from dvrk_newton.camera_lighting import LIGHT_OFFSET_M, endoscope_light_positions


def test_scene_camera_parses_unixfd():
    camera = SceneCamera(
        mode="stereo",
        owner="ECM",
        settings={
            "mode": "stereo",
            "owner": "ECM",
            "width": 1280,
            "height": 720,
            "horizontal_fov_deg": 60.0,
            "near_clip_m": 0.005,
            "far_clip_m": 10.0,
            "baseline_m": 0.006,
            "publish_rate_hz": 30.0,
            "transports": ["unixfd"],
            "unixfd": {"socket_path": "@dvrk:newton:stereo_source"},
            "endoscope_lights": False,
        },
    )
    options = CameraOptions.from_scene(camera)
    assert options is not None
    assert options.mode == "stereo"
    assert options.socket_path == "@dvrk:newton:stereo_source"
    assert (options.width, options.height, options.rate_hz) == (1280, 720, 30.0)
    assert options.transport_width == 2560
    assert options.baseline_m == 0.006
    assert not options.endoscope_lights
    assert CameraOptions().endoscope_lights


def test_endoscope_lights_follow_camera_roll_and_translation():
    orientation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    pose = Pose(position=np.array([0.2, -0.1, 0.4]), orientation=orientation)
    positions = endoscope_light_positions(pose)
    np.testing.assert_allclose(positions.mean(axis=0), pose.position, atol=1e-7)
    np.testing.assert_allclose(
        positions[0] - positions[1], -2 * LIGHT_OFFSET_M * orientation[:, 1], atol=1e-7
    )


@pytest.mark.parametrize("renderer_type", [
    NewtonCameraRenderer,
    pytest.param(NewtonOpenGLCameraRenderer, marks=pytest.mark.skipif(
        not (os.environ.get("DISPLAY") or os.environ.get("PYGLET_HEADLESS")),
        reason="OpenGL requires a display or PYGLET_HEADLESS=1",
    )),
])
@pytest.mark.parametrize("opacity", [1.0, 0.5])
def test_endoscope_lights_brighten_front_surface(renderer_type, opacity):
    brightness = []
    for enabled in (False, True):
        if renderer_type is NewtonOpenGLCameraRenderer:
            # Newton caches GL resources globally. Separate processes avoid
            # reusing resources after the previous viewer's context closes.
            script = (
                "import runpy; "
                f"test = runpy.run_path({str(Path(__file__).resolve())!r}); "
                "print(test['_front_surface_brightness']("
                f"test['NewtonOpenGLCameraRenderer'], {enabled!r}, {opacity!r}))"
            )
            result = subprocess.run(
                [sys.executable, "-c", script], capture_output=True, text=True,
                check=True, timeout=30,
            )
            brightness.append(float(result.stdout.strip().splitlines()[-1]))
        else:
            brightness.append(_front_surface_brightness(renderer_type, enabled, opacity))
    assert brightness[1] > brightness[0] + 10, brightness


def _front_surface_brightness(renderer_type, enabled, opacity):
    import newton
    import warp as wp

    builder = newton.ModelBuilder()
    builder.add_shape_box(
        -1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()),
        hx=0.04, hy=0.04, hz=0.005, color=(0.3, 0.3, 0.3), opacity=opacity,
    )
    model = builder.finalize("cpu")
    state = model.state()
    pose = Pose(position=np.zeros(3), orientation=np.eye(3))
    renderer = renderer_type(model, CameraOptions(
        mode="stereo", width=32, height=24, endoscope_lights=enabled,
    ), device="cpu")
    try:
        renderer.start()
        frame = renderer.render(state, pose, 0.0)
        # Sample the surface near the center of each stereo eye.
        centers = frame.rgba[10:14, (14, 15, 16, 17, 46, 47, 48, 49), :3]
        assert frame.rgba.shape == (24, 64, 4)
        assert np.all(frame.rgba[10:14, 14:18, 3] == 255)
        return float(centers.mean())
    finally:
        renderer.close()


def test_newton_camera_renderer_stereo():
    import newton
    import warp as wp

    device = "cuda:0" if wp.is_cuda_available() else "cpu"

    builder = newton.ModelBuilder()
    model = builder.finalize(device)
    state = model.state()

    options = CameraOptions(
        mode="stereo",
        width=160,
        height=120,
        rate_hz=30.0,
        baseline_m=0.006,
    )
    renderer = NewtonCameraRenderer(model, options, device=device)

    pose = Pose(
        position=np.array([0.0, -0.2, 0.1]),
        orientation=np.eye(3),
    )
    frame = renderer.render(state, pose, 0.0)
    assert isinstance(frame, VideoFrame)
    assert frame.rgba.shape == (120, 320, 4)
    assert frame.rgba.dtype == np.uint8
    assert frame.sequence == 1
