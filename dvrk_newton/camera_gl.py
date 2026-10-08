"""Offscreen OpenGL ECM camera rendering through Newton's ViewerGL."""

from __future__ import annotations

from typing import Any

import numpy as np

from dvrk_simulator_base.types import Pose

from .backend import load_newton
from .camera import CameraOptions, VideoFrame

class NewtonOpenGLCameraRenderer:
    """Render full-pose mono or stereo views on the camera worker's GL context."""

    def __init__(self, model: Any, options: CameraOptions, device: str = "cuda:0") -> None:
        self.model = model
        self.options = options
        self.device = str(device)
        self.newton, self.wp = load_newton()
        self.viewer = None
        self._camera = None
        self._rgb_target = None
        self.sequence = 0

    def start(self) -> None:
        """Create the OpenGL context in the same thread that will render and close it."""
        from newton._src.viewer.camera import Camera
        from newton.viewer import ViewerGL

        class OpticalCamera(Camera):
            """Viewer camera with the ECM's full orientation, including roll."""

            def set_pose(self, position: np.ndarray, forward: np.ndarray, up: np.ndarray) -> None:
                self.pos = self._as_vec3(position)
                self._optical_forward = self._as_vec3(forward)
                self._optical_up = self._as_vec3(up)
                self.pivot = self.pos + self._optical_forward

            def get_front(self):
                return getattr(self, "_optical_forward", None) or super().get_front()

            def get_up(self):
                return getattr(self, "_optical_up", None) or super().get_up()

            def get_view_matrix(self, scaling: float = 1.0) -> np.ndarray:
                from pyglet.math import Mat4, Vec3

                eye = Vec3(*(self.pos / scaling))
                return np.array(
                    Mat4.look_at(eye, eye + self._optical_forward, self._optical_up),
                    dtype=np.float32,
                )

        viewer = ViewerGL(
            width=self.options.width,
            height=self.options.height,
            headless=True,
            vsync=False,
        )
        self.viewer = viewer
        viewer.set_model(self.model)
        # A video camera needs no viewer shadow-map pass.
        viewer.renderer.draw_shadows = False
        self._camera = OpticalCamera(
            width=viewer.camera.width,
            height=viewer.camera.height,
            # The existing Warp helper applies this legacy-named angle as a
            # vertical FOV; keep both renderers framed the same way.
            fov=self.options.horizontal_fov_degrees,
            near=self.options.near_m,
            far=self.options.far_m,
            up_axis=self.model.up_axis,
        )
        viewer.camera = self._camera
        if (self._camera.width, self._camera.height) != (
            self.options.width,
            self.options.height,
        ):
            raise RuntimeError(
                "OpenGL framebuffer size differs from the configured camera size: "
                f"{self._camera.width}x{self._camera.height} versus "
                f"{self.options.width}x{self.options.height}"
            )
        self._rgb_target = self.wp.empty(
            (self.options.height, self.options.width, 3),
            dtype=self.wp.uint8,
            device=self.device,
        )

    def render(self, state: Any, optical_pose: Pose, simulation_time: float) -> VideoFrame:
        """Render both eyes in one GL context and pack the existing RGBA transport."""
        if self.viewer is None or self._camera is None or self._rgb_target is None:
            raise RuntimeError("OpenGL camera renderer has not started")

        position = optical_pose.position
        forward = optical_pose.orientation[:, 0]
        left = optical_pose.orientation[:, 1]
        up = optical_pose.orientation[:, 2]
        if self.options.mode == "stereo":
            half_baseline = 0.5 * self.options.baseline_m
            eye_positions = (
                position + left * half_baseline,
                position - left * half_baseline,
            )
        else:
            eye_positions = (position,)

        height, width = self.options.height, self.options.width
        rgba = np.empty((height, self.options.transport_width, 4), dtype=np.uint8)
        rgba[:, :, 3] = 255

        for eye_index, eye_position in enumerate(eye_positions):
            self._camera.set_pose(eye_position, forward, up)
            self.viewer.begin_frame(simulation_time)
            # ViewerGL keeps logged scene geometry between frames. Both eyes
            # use the same state, and log_state synchronizes CUDA with the CPU;
            # doing it again for the second eye only repeats that work.
            if eye_index == 0:
                self.viewer.log_state(state)

            self.viewer.end_frame()

            rgb = self.viewer.get_frame(self._rgb_target).numpy()

            rgba[:, eye_index * width : (eye_index + 1) * width, :3] = rgb

        self.sequence += 1
        return VideoFrame(rgba=rgba, simulation_time=simulation_time, sequence=self.sequence)

    def close(self) -> None:
        """Destroy the GL context on the same thread that created it."""
        viewer, self.viewer = self.viewer, None
        self._camera = None
        self._rgb_target = None
        if viewer is not None:
            viewer.close()
