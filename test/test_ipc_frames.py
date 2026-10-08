"""Newton resolves targets from authoritative completed scenes, independent of ROS."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from dvrk_newton.runtime import NewtonRuntime
from dvrk_simulator_base.cartesian_command import CartesianCommand
from dvrk_simulator_base.publication_frames import with_publication_frames
from dvrk_simulator_base.snapshots import ArmSnapshot, OperatingStateSnapshot
from dvrk_simulator_base.types import JointState, Pose, Twist


def test_newton_resolves_latest_simulation_ecm_and_publication_frames():
    configs = [SimpleNamespace(name=name, type=kind, base_frame=name + "_base", parent_frame="world",
                               base_position=[1, 0, 0], base_orientation_xyzw=[0, 0, 0, 1])
               for name, kind in (("ECM", "ECM"), ("PSM1", "PSM"))]
    runtime = NewtonRuntime.__new__(NewtonRuntime)
    runtime.arms = {config.name: SimpleNamespace(config=config) for config in configs}
    joints = JointState(("yaw",), [0], [0])
    ecm_pose = Pose([2, 0, 0], np.eye(3))
    psm_pose = Pose([2.3, 0.1, 0.2], np.eye(3))
    runtime._frame_snapshots = {
        name: ArmSnapshot(1, 0.1, True, joints, joints, pose, pose,
                          Twist([0, 0, 0], [0, 0, 0]), None, None,
                          OperatingStateSnapshot("ENABLED", True, False))
        for name, pose in (("ECM", ecm_pose), ("PSM1", psm_pose))
    }
    command = CartesianCommand(Pose([0.1, 0.2, 0.3], np.eye(3)), "ECM_view")
    first = runtime.resolve_cartesian_target("PSM1", command)
    np.testing.assert_allclose(first.position, psm_pose.position)
    converted = with_publication_frames(runtime._frame_snapshots, configs)
    np.testing.assert_allclose(converted["PSM1"].publication_frames.measured_cp.position, command.pose.position)
    np.testing.assert_allclose(converted["PSM1"].publication_frames.local_measured_cp.position, [1.3, 0.1, 0.2])
    assert converted["PSM1"].publication_frames.frame_id == "ECM_view"
    assert converted["ECM"].publication_frames.frame_id == "world"
    runtime._frame_snapshots["ECM"] = replace(runtime._frame_snapshots["ECM"],
                                            measured_cp_world=Pose([4, 0, 0], np.eye(3)))
    second = runtime.resolve_cartesian_target("PSM1", command)
    np.testing.assert_allclose(second.position - first.position, [2, 0, 0])
