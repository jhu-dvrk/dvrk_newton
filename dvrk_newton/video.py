"""Low-latency GStreamer Unix-FD video output backed by Linux memfd buffers."""

from __future__ import annotations

import errno
from fractions import Fraction
import mmap
import os
from pathlib import Path
import socket
import stat

from .camera import CameraOptions, VideoFrame
from .errors import GStreamerDependencyError, NewtonBackendError


def _load_gstreamer():
    try:
        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstAllocators", "1.0")
        from gi.repository import Gio, Gst, GstAllocators
    except (ImportError, ValueError) as error:
        raise GStreamerDependencyError(
            "GStreamer Python bindings and GstAllocators are required; install "
            "python3-gi, gstreamer1.0-plugins-base, and gstreamer1.0-plugins-bad"
        ) from error
    Gst.init(None)
    return Gst, GstAllocators, Gio


def _prepare_socket_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(mode):
        raise NewtonBackendError(f"refusing to replace non-socket path {path}")
    live = False
    for socket_type in (socket.SOCK_STREAM, socket.SOCK_SEQPACKET):
        probe = socket.socket(socket.AF_UNIX, socket_type)
        try:
            if probe.connect_ex(str(path)) == 0:
                live = True
                break
        finally:
            probe.close()
    if live:
        raise NewtonBackendError(f"camera socket is already in use: {path}")
    path.unlink()


def _prepare_abstract_socket(reference: str) -> None:
    """Reject an abstract socket name already owned by another publisher."""
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.bind(f"\0{reference[1:]}")
    except OSError as error:
        if error.errno == errno.EADDRINUSE:
            raise NewtonBackendError(
                f"camera abstract socket is already in use: {reference}"
            ) from error
        raise NewtonBackendError(
            f"could not check camera abstract socket {reference}: {error}"
        ) from error
    finally:
        probe.close()


class UnixFdVideoSink:
    """Push RGBA frames without unbounded queueing; Gst owns each submitted fd."""

    def __init__(self, options: CameraOptions) -> None:
        self.options = options
        self.Gst, self.GstAllocators, self.Gio = _load_gstreamer()
        self.pipeline = None
        self.source = None
        self.frames_pushed = 0
        self._socket_inode: int | None = None

    def start(self) -> None:
        if self.pipeline is not None:
            return
        abstract_socket = isinstance(self.options.socket_path, str)
        if abstract_socket:
            _prepare_abstract_socket(self.options.socket_path)
        else:
            _prepare_socket_path(self.options.socket_path)
        Gst = self.Gst
        pipeline = Gst.Pipeline.new("dvrk-newton-camera")
        source = Gst.ElementFactory.make("appsrc", "camera-source")
        queue = Gst.ElementFactory.make("queue", "latest-frame")
        sink = Gst.ElementFactory.make("unixfdsink", "camera-socket")
        if any(item is None for item in (pipeline, source, queue, sink)):
            raise GStreamerDependencyError(
                "GStreamer appsrc, queue, or unixfdsink is unavailable"
            )
        rate = Fraction(str(self.options.rate_hz)).limit_denominator(1001)
        caps = Gst.Caps.from_string(
            f"video/x-raw,format=RGBA,width={self.options.transport_width},"
            f"height={self.options.height},framerate={rate.numerator}/{rate.denominator}"
        )
        source.set_property("caps", caps)
        source.set_property("is-live", True)
        source.set_property("format", Gst.Format.TIME)
        source.set_property("do-timestamp", True)
        source.set_property("block", False)
        source.set_property("max-buffers", 1)
        source.set_property("leaky-type", 2)  # downstream: discard oldest
        queue.set_property("max-size-buffers", 1)
        queue.set_property("max-size-bytes", 0)
        queue.set_property("max-size-time", 0)
        queue.set_property("leaky", 2)  # downstream: discard the oldest frame
        socket_path = str(self.options.socket_path)
        if abstract_socket:
            socket_path = socket_path[1:]
            sink.set_property("socket-type", self.Gio.UnixSocketAddressType.ABSTRACT)
        sink.set_property("socket-path", socket_path)
        sink.set_property("sync", False)
        sink.set_property("async", False)
        pipeline.add(source)
        pipeline.add(queue)
        pipeline.add(sink)
        if not source.link(queue) or not queue.link(sink):
            raise NewtonBackendError("could not link the camera GStreamer pipeline")
        self.pipeline = pipeline
        self.source = source
        change = pipeline.set_state(Gst.State.PLAYING)
        if change == Gst.StateChangeReturn.FAILURE:
            detail = self._bus_error_detail()
            self.close()
            if detail is not None:
                raise NewtonBackendError(f"camera GStreamer pipeline error: {detail}")
            raise NewtonBackendError("camera GStreamer pipeline failed to start")
        pipeline.get_state(2 * Gst.SECOND)
        self._raise_bus_error()
        if not abstract_socket:
            try:
                self._socket_inode = self.options.socket_path.stat().st_ino
            except FileNotFoundError:
                self.close()
                raise NewtonBackendError(
                    f"GStreamer did not create camera socket {self.options.socket_path}"
                )

    def _raise_bus_error(self) -> None:
        detail = self._bus_error_detail()
        if detail is not None:
            raise NewtonBackendError(f"camera GStreamer pipeline error: {detail}")

    def _bus_error_detail(self) -> str | None:
        if self.pipeline is None:
            return None
        message = self.pipeline.get_bus().pop_filtered(self.Gst.MessageType.ERROR)
        if message is not None:
            error, debug = message.parse_error()
            return f"{error}; {debug or 'no details'}"
        return None

    def push(self, frame: VideoFrame) -> None:
        if self.source is None:
            raise RuntimeError("camera video sink is not started")
        expected = (self.options.height, self.options.transport_width, 4)
        if frame.rgba.shape != expected or frame.rgba.dtype.name != "uint8":
            raise ValueError(f"camera frame must be uint8 RGBA with shape {expected}")
        size = frame.rgba.nbytes
        fd = os.memfd_create("dvrk-newton-camera", os.MFD_CLOEXEC)
        owned_by_gstreamer = False
        try:
            os.ftruncate(fd, size)
            with mmap.mmap(fd, size, access=mmap.ACCESS_WRITE) as mapped:
                mapped[:] = frame.rgba
            allocator = self.GstAllocators.FdAllocator.new()
            memory = self.GstAllocators.FdAllocator.alloc(
                allocator,
                fd,
                size,
                self.GstAllocators.FdMemoryFlags.NONE,
            )
            owned_by_gstreamer = True
            buffer = self.Gst.Buffer.new()
            buffer.append_memory(memory)
            buffer.duration = round(self.Gst.SECOND / self.options.rate_hz)
            result = self.source.emit("push-buffer", buffer)
            if result != self.Gst.FlowReturn.OK:
                raise NewtonBackendError(
                    f"camera pipeline rejected frame: {result.value_nick}"
                )
            self.frames_pushed += 1
            self._raise_bus_error()
        finally:
            if not owned_by_gstreamer:
                os.close(fd)

    def close(self) -> None:
        pipeline, self.pipeline = self.pipeline, None
        source, self.source = self.source, None
        if source is not None:
            source.emit("end-of-stream")
        if pipeline is not None:
            pipeline.set_state(self.Gst.State.NULL)
            pipeline.get_state(2 * self.Gst.SECOND)
        if isinstance(self.options.socket_path, str):
            return
        try:
            current = self.options.socket_path.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISSOCK(current.st_mode) and current.st_ino == self._socket_inode:
            self.options.socket_path.unlink()
