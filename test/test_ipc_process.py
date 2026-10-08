"""Run the actual IPC worker loop in a fresh interpreter with a small backend."""

import time
from types import SimpleNamespace

import numpy as np
import pytest

from dvrk_simulator_base import process as ipc_runtime
from dvrk_simulator_base.cartesian_command import CartesianCommand
from dvrk_simulator_base.command_mailbox import CommandMailboxes
from dvrk_simulator_base.ros_interface import LatestSnapshot
from dvrk_simulator_base.types import Pose
from dvrk_newton.python_runtime import resolve_newton_python


class FakeNode:
    def __init__(self):
        self.arm_interfaces = {name: SimpleNamespace(commands=CommandMailboxes(), snapshots=LatestSnapshot(),
                                                     _publish_warning=lambda text: None)
                               for name in ("ECM", "PSM1")}
        self.states = {}

    def install_initial_snapshots(self, snapshots):
        self.states = snapshots
        for name, state in snapshots.items():
            self.arm_interfaces[name].snapshots.set_initial(state)

    def accept_snapshots(self, snapshots):
        self.states = snapshots
        for name, state in snapshots.items():
            self.arm_interfaces[name].snapshots.set(state)

    def get_logger(self):
        return SimpleNamespace(info=lambda text: None)


def pump_until(worker, node, predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate():
        worker.poll(node)
        if time.monotonic() > deadline:
            raise AssertionError("IPC worker did not reach expected state")
        time.sleep(0.001)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_real_newton_worker_moving_ecm_cartesian_command_and_state(tmp_path, monkeypatch, device):
    from dvrk_simulator_base.cartesian_frames import compose_pose, view_pose_from_optical
    monkeypatch.delenv("DVRK_SIMULATOR_TEST_TIMEOUT", raising=False)
    config = tmp_path / "newton.yaml"
    config.write_text(f"device: {device}\nheadless: true\ngenerated_root: {tmp_path / 'generated'}\n")
    scene = tmp_path / "scene.yaml"
    scene.write_text("""scene:
  name: ipc_test
  robots:
    - {config: ECM.yaml, endoscope: Si_straight}
    - {config: PSM1.yaml, instrument: '420006'}
""")
    python_bin = resolve_newton_python().path
    process = ipc_runtime.SimulationProcess(python_bin, "dvrk_newton.simulation_worker", {
        "config": str(config), "scene": str(scene), "device": device, "headless": True,
    })
    node = FakeNode()
    try:
        pump_until(process, node, lambda: process.ready, timeout=60)
        q = node.states["ECM"].measured_js.position.copy()
        q[0] += 0.02
        node.arm_interfaces["ECM"].commands.submit_servo("servo_jp", q)
        pump_until(process, node, lambda: np.allclose(node.states["ECM"].measured_js.position, q))
        frame_pose = node.states["PSM1"].publication_frames.measured_cp
        target = Pose(frame_pose.position + [0.001, 0, 0], frame_pose.orientation)
        expected = compose_pose(view_pose_from_optical(node.states["ECM"].measured_cp_world), target)
        node.arm_interfaces["PSM1"].commands.submit_servo("servo_cp", CartesianCommand(target, "ECM_view"))
        pump_until(process, node, lambda: np.linalg.norm(node.states["PSM1"].measured_cp_world.position - expected.position) < 1e-4)
        np.testing.assert_allclose(node.states["PSM1"].publication_frames.measured_cp.position, target.position, atol=1e-4)
        assert node.states["ECM"].sequence == node.states["PSM1"].sequence
        node.arm_interfaces["PSM1"].commands.submit_discrete("state_command", "pause")
        pump_until(process, node, lambda: node.states["PSM1"].operating_state.state == "PAUSED")
        _, events = node.arm_interfaces["PSM1"].snapshots.get_with_events()
        assert any(event.state == "PAUSED" for event in events)
    finally:
        process.shutdown()
    assert process.process.returncode == 0
