"""ECM optical camera raytraced rendering in NVIDIA Newton using Warp."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np

from dvrk_simulator_base.rotations import rotation_to_quaternion_xyzw
from dvrk_simulator_base.types import Pose

from .backend import load_newton


@dataclass(frozen=True)
class CameraOptions:
    enabled: bool = True
    mode: str = "stereo"
    socket_path: str | Path = "@dvrk:newton:stereo_source"
    width: int = 1280
    height: int = 720
    rate_hz: float = 30.0
    horizontal_fov_degrees: float = 60.0
    near_m: float = 0.005
    far_m: float = 10.0
    baseline_m: float = 0.006

    def __post_init__(self) -> None:
        mode = str(self.mode).lower()
        socket_reference = str(self.socket_path)
        if mode not in {"mono", "stereo"}:
            raise ValueError("camera mode must be 'mono' or 'stereo'")
        if socket_reference.startswith("@dvrk:"):
            if len(socket_reference.split(":")) != 3:
                raise ValueError(
                    "dVRK camera socket must use @dvrk:<package>:<stream> syntax"
                )
            path: str | Path = socket_reference
        else:
            path = Path(socket_reference).expanduser()
            if not path.is_absolute():
                raise ValueError(
                    "camera socket must be @dvrk:<package>:<stream> or an absolute path"
                )
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera width and height must be positive")
        if not np.isfinite(self.rate_hz) or self.rate_hz <= 0.0:
            raise ValueError("camera rate must be finite and positive")
        if not 0.0 < self.horizontal_fov_degrees < 180.0:
            raise ValueError("camera horizontal FOV must be between 0 and 180 degrees")
        if self.near_m <= 0.0 or self.far_m <= self.near_m:
            raise ValueError("camera clipping planes must satisfy 0 < near < far")
        if not np.isfinite(self.baseline_m) or self.baseline_m <= 0.0:
            raise ValueError("camera baseline must be finite and positive")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "socket_path", path)

    @property
    def transport_width(self) -> int:
        return self.width * (2 if self.mode == "stereo" else 1)

    @classmethod
    def from_scene(cls, camera: Any) -> CameraOptions | None:
        if camera is None or camera.mode == "off":
            return None
        settings = camera.as_dict()
        transports = settings.get("transports", ["unixfd"])
        if "unixfd" not in transports:
            return None
        unixfd = settings.get("unixfd", {}) or {}
        if not isinstance(unixfd, dict):
            raise ValueError("scene.camera.unixfd must be a mapping")
        socket_path = unixfd.get(
            "socket_path",
            f"@dvrk:newton:{camera.mode}_source",
        )
        return cls(
            enabled=True,
            mode=camera.mode,
            socket_path=socket_path,
            width=int(settings.get("width", 1280)),
            height=int(settings.get("height", 720)),
            rate_hz=float(settings.get("publish_rate_hz", 30.0)),
            horizontal_fov_degrees=float(settings.get("horizontal_fov_deg", 60.0)),
            near_m=float(settings.get("near_clip_m", 0.005)),
            far_m=float(settings.get("far_clip_m", 10.0)),
            baseline_m=float(settings.get("baseline_m", 0.006)),
        )


@dataclass(frozen=True)
class VideoFrame:
    rgba: np.ndarray
    simulation_time: float
    sequence: int


class NewtonCameraRenderer:
    """Raytraced camera renderer using Newton SensorTiledCamera."""

    def __init__(self, model: Any, options: CameraOptions, device: str = "cuda:0") -> None:
        self.options = options
        self.device = str(device)
        self.newton, self.wp = load_newton()
        self.model = model

        self.sensor = self.newton.sensors.SensorTiledCamera(model)
        self.camera_count = 2 if options.mode == "stereo" else 1
        fov_rad = math.radians(options.horizontal_fov_degrees)

        self.rays = self.wp.zeros(
            (self.camera_count, options.height, options.width, 2),
            dtype=self.wp.vec3f,
            device=self.device,
        )
        for i in range(self.camera_count):
            self.sensor.utils.compute_camera_rays_pinhole(
                options.width,
                options.height,
                camera_fovs=fov_rad,
                out_rays=self.rays,
                camera_index=i,
            )

        self.color_output = self.sensor.utils.create_color_image_output(
            options.width,
            options.height,
            camera_count=self.camera_count,
        )
        self.sequence = 0

    def render(self, state: Any, optical_pose: Pose, simulation_time: float) -> VideoFrame:
        """Render a mono or stereo frame at the given ECM optical pose."""
        pos = optical_pose.position
        # ECM optical frame conventions:
        # column 0 is forward (+X along endoscope)
        # column 1 is left (+Y)
        # column 2 is up (+Z)
        forward = optical_pose.orientation[:, 0]
        left = optical_pose.orientation[:, 1]
        up = optical_pose.orientation[:, 2]
        right = -left

        # In camera coordinates for pinhole raytracing (OpenGL convention):
        # Camera X is right, Camera Y is up, Camera Z is -forward
        R_cam = np.column_stack((right, up, -forward))
        q = rotation_to_quaternion_xyzw(R_cam)
        quat = self.wp.quat(q[0], q[1], q[2], q[3])

        if self.camera_count == 1:
            cam_xform = self.wp.transform(self.wp.vec3(*pos), quat)
            cam_xforms = self.wp.array([[cam_xform]], dtype=self.wp.transform, device=self.device)
        else:
            half_baseline = 0.5 * self.options.baseline_m
            left_pos = pos + left * half_baseline
            right_pos = pos - left * half_baseline
            left_cam = self.wp.transform(self.wp.vec3(*left_pos), quat)
            right_cam = self.wp.transform(self.wp.vec3(*right_pos), quat)
            cam_xforms = self.wp.array([[left_cam], [right_cam]], dtype=self.wp.transform, device=self.device)

        self.model.bvh_refit_shapes(state)
        self.sensor.update(state, cam_xforms, self.rays, color_image=self.color_output)

        raw = self.color_output.numpy()  # shape (1, camera_count, height, width) of uint32
        h, w = self.options.height, self.options.width

        if self.camera_count == 1:
            rgba = raw[0, 0].view(np.uint8).reshape(h, w, 4)
        else:
            left_img = raw[0, 0].view(np.uint8).reshape(h, w, 4)
            right_img = raw[0, 1].view(np.uint8).reshape(h, w, 4)
            rgba = np.ascontiguousarray(np.concatenate((left_img, right_img), axis=1))

        self.sequence += 1
        return VideoFrame(
            rgba=rgba,
            simulation_time=simulation_time,
            sequence=self.sequence,
        )
