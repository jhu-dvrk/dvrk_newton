import os
import socket

import numpy as np
import pytest

from dvrk_newton.camera import CameraOptions, VideoFrame
from dvrk_newton.errors import NewtonBackendError
from dvrk_newton.video import UnixFdVideoSink


def test_unixfd_sink_accepts_memfd_rgba_and_cleans_socket(tmp_path):
    path = tmp_path / "camera.sock"
    options = CameraOptions(mode="mono", socket_path=path, width=16, height=12, rate_hz=30.0)
    sink = UnixFdVideoSink(options)
    try:
        sink.start()
        assert path.is_socket()
        rgba = np.full((12, 16, 4), [10, 20, 30, 255], dtype=np.uint8)
        sink.push(VideoFrame(rgba, 0.0, 0))
        assert sink.frames_pushed == 1
    finally:
        sink.close()
    assert not path.exists()


def test_unixfd_sink_accepts_dvrk_abstract_socket():
    reference = f"@dvrk:newton:pytest-{os.getpid()}"
    options = CameraOptions(mode="mono", socket_path=reference, width=16, height=12)
    sink = UnixFdVideoSink(options)
    try:
        sink.start()
        rgba = np.full((12, 16, 4), [10, 20, 30, 255], dtype=np.uint8)
        sink.push(VideoFrame(rgba, 0.0, 0))
        assert sink.frames_pushed == 1
    finally:
        sink.close()


def test_unixfd_sink_reports_busy_abstract_socket():
    reference = f"@dvrk:newton:pytest-busy-{os.getpid()}"
    blocker = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    blocker.bind(f"\0{reference[1:]}")
    blocker.listen()
    sink = UnixFdVideoSink(CameraOptions(mode="mono", socket_path=reference, width=16, height=12))
    try:
        with pytest.raises(NewtonBackendError, match="abstract socket is already in use"):
            sink.start()
    finally:
        sink.close()
        blocker.close()
