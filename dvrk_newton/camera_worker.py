"""Latest-state camera rendering on a worker thread with an independent model."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time

import numpy as np

from dvrk_simulator_base.types import Pose

from .camera import NewtonCameraRenderer
from .camera_gl import NewtonOpenGLCameraRenderer
from .errors import NewtonBackendError
from .video import UnixFdVideoSink


@dataclass(frozen=True)
class _CameraSnapshot:
    body_q: np.ndarray
    particle_q: np.ndarray | None
    optical_pose: Pose
    simulation_time: float


class NewtonCameraWorker:
    """Render the newest completed simulation state, dropping superseded frames."""

    def __init__(
        self,
        renderer: NewtonCameraRenderer | NewtonOpenGLCameraRenderer,
        render_state,
        sink: UnixFdVideoSink,
    ) -> None:
        self.renderer = renderer
        self.render_state = render_state
        self.sink = sink
        self._interval = 1.0 / renderer.options.rate_hz
        self._condition = threading.Condition()
        self._latest: _CameraSnapshot | None = None
        self._stopping = False
        self._error: Exception | None = None
        self._frames_pushed = 0
        self._sample_count = 0
        self._sample_at = time.monotonic()
        self._thread = threading.Thread(
            target=self._run, name="dvrk-newton-camera", daemon=False
        )

    def start(self) -> None:
        self._thread.start()

    def offer(
        self,
        body_q: np.ndarray,
        particle_q: np.ndarray | None,
        optical_pose: Pose,
        simulation_time: float,
    ) -> None:
        # The lock guards only the one-slot mailbox.  The simulation thread
        # never waits for the renderer or the video transport.
        snapshot = _CameraSnapshot(
            body_q=np.array(body_q, copy=True),
            particle_q=None if particle_q is None else np.array(particle_q, copy=True),
            optical_pose=optical_pose,
            simulation_time=simulation_time,
        )
        with self._condition:
            if not self._stopping:
                self._latest = snapshot
                self._condition.notify()

    def _run(self) -> None:
        next_frame_at = 0.0
        try:
            self.renderer.start()
            while True:
                with self._condition:
                    while not self._stopping:
                        now = time.monotonic()
                        if self._latest is not None and now >= next_frame_at:
                            snapshot = self._latest
                            self._latest = None
                            break
                        wait_time = (
                            None if self._latest is None else max(0.0, next_frame_at - now)
                        )
                        self._condition.wait(wait_time)
                    else:
                        return

                next_frame_at = time.monotonic() + self._interval
                wp = self.renderer.wp
                device = self.renderer.device
                self.render_state.body_q.assign(
                    wp.array(snapshot.body_q, dtype=wp.transform, device=device)
                )
                if snapshot.particle_q is not None:
                    self.render_state.particle_q.assign(
                        wp.array(
                            snapshot.particle_q,
                            dtype=self.render_state.particle_q.dtype,
                            device=device,
                        )
                    )
                frame = self.renderer.render(
                    self.render_state, snapshot.optical_pose, snapshot.simulation_time
                )
                self.sink.push(frame)
                with self._condition:
                    self._frames_pushed += 1
        except Exception as error:
            with self._condition:
                self._error = error
                self._stopping = True
                self._condition.notify_all()
        finally:
            try:
                self.renderer.close()
            except Exception as error:
                with self._condition:
                    if self._error is None:
                        self._error = error
                    self._stopping = True
                    self._condition.notify_all()

    def raise_if_failed(self) -> None:
        with self._condition:
            error = self._error
        if error is not None:
            raise NewtonBackendError(f"camera render worker failed: {error}") from error

    def take_rate_hz(self) -> float:
        with self._condition:
            now = time.monotonic()
            elapsed = max(now - self._sample_at, 1e-6)
            count = self._frames_pushed - self._sample_count
            self._sample_at = now
            self._sample_count = self._frames_pushed
        return count / elapsed

    def close(self) -> None:
        with self._condition:
            self._stopping = True
            self._latest = None
            self._condition.notify_all()
        if self._thread.is_alive():
            self._thread.join()
