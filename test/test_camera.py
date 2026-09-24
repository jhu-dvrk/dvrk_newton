from pathlib import Path

import numpy as np
import pytest

from dvrk_simulator_base.scene import SceneCamera
from dvrk_simulator_base.types import Pose
from dvrk_newton.camera import CameraOptions, NewtonCameraRenderer, VideoFrame


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
        },
    )
    options = CameraOptions.from_scene(camera)
    assert options is not None
    assert options.enabled
    assert options.mode == "stereo"
    assert options.socket_path == "@dvrk:newton:stereo_source"
    assert (options.width, options.height, options.rate_hz) == (1280, 720, 30.0)
    assert options.transport_width == 2560
    assert options.baseline_m == 0.006


def test_newton_camera_renderer_stereo():
    import newton
    import warp as wp

    builder = newton.ModelBuilder()
    model = builder.finalize("cuda:0")
    state = model.state()

    options = CameraOptions(
        mode="stereo",
        width=160,
        height=120,
        rate_hz=30.0,
        baseline_m=0.006,
    )
    renderer = NewtonCameraRenderer(model, options, device="cuda:0")

    pose = Pose(
        position=np.array([0.0, -0.2, 0.1]),
        orientation=np.eye(3),
    )
    frame = renderer.render(state, pose, 0.0)
    assert isinstance(frame, VideoFrame)
    assert frame.rgba.shape == (120, 320, 4)
    assert frame.rgba.dtype == np.uint8
    assert frame.sequence == 1
