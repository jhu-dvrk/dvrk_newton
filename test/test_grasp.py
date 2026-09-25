"""Unit and integration tests for dvrk_newton grasping and physics dynamics."""

import numpy as np
import pytest

from dvrk_simulator_base.command_mailbox import CommandMailboxes
from dvrk_newton.configuration import load_installed_scene_config, resolve_scene_path
from dvrk_newton.runtime import NewtonRuntime, NewtonRuntimeOptions


@pytest.fixture
def tray_cubes_scene():
    resolved = resolve_scene_path("/tmp/test.yaml", ["PSM1_420006.yaml", "tray_cubes.yaml"])
    return load_installed_scene_config(resolved)


def test_grasp_and_release_lifecycle(tray_cubes_scene):
    commands = {"PSM1": CommandMailboxes()}
    options = NewtonRuntimeOptions(
        device="cuda:0",
        headless=True,
        scene_objects=tuple(tray_cubes_scene.objects),
    )
    runtime = NewtonRuntime(tray_cubes_scene.robots, options, commands)
    snaps = runtime.initialize()

    cube = runtime.scene_objects["cube"]
    cube_b_idx = cube.body_index

    # 1. Closed jaws on the cube at home position form a grasp attachment
    runtime.step()
    assert "PSM1" in runtime.grasp_manager.attachments
    attachment = runtime.grasp_manager.attachments["PSM1"]
    assert attachment.object_name == "cube"

    # 2. Command PSM1 arm up by 2 cm; cube moves rigidly with the arm
    cur_pose = snaps["PSM1"].measured_cp_world
    target_pos = cur_pose.position + np.array([0.0, 0.0, 0.02])
    target_pose = (target_pos, cur_pose.orientation)
    ik_res = runtime.compute_ik("PSM1", target_pose)
    assert ik_res.success

    commands["PSM1"].submit_servo("servo_jp", ik_res.position)
    runtime.step()

    lifted_z = runtime.state.body_q.numpy()[cube_b_idx, 2]
    assert lifted_z > 0.065, f"Cube should lift with arm: {lifted_z}"

    # 3. Open jaws past release threshold (0.2 rad > 0.08 rad); attachment is released
    commands["PSM1"].submit_servo("jaw/servo_jp", 0.2)
    runtime.step()
    assert "PSM1" not in runtime.grasp_manager.attachments

    # 4. Step physics solver; cube falls under gravity and rests back on the tray
    for _ in range(60):
        runtime.step()

    dropped_z = runtime.state.body_q.numpy()[cube_b_idx, 2]
    assert dropped_z < lifted_z, f"Cube should drop after release: {dropped_z} vs {lifted_z}"
    assert abs(dropped_z - 0.040) < 0.005, f"Cube should rest on tray surface: {dropped_z}"


def test_scene_reset(tray_cubes_scene):
    commands = {"PSM1": CommandMailboxes()}
    options = NewtonRuntimeOptions(
        device="cuda:0",
        headless=True,
        scene_objects=tuple(tray_cubes_scene.objects),
    )
    runtime = NewtonRuntime(tray_cubes_scene.robots, options, commands)
    runtime.initialize()

    cube = runtime.scene_objects["cube"]
    initial_z = runtime.state.body_q.numpy()[cube.body_index, 2]

    # Open jaws and let cube fall
    commands["PSM1"].submit_servo("jaw/servo_jp", 0.3)
    for _ in range(40):
        runtime.step()

    fallen_z = runtime.state.body_q.numpy()[cube.body_index, 2]
    assert fallen_z < initial_z

    # Request scene reset
    runtime.request_reset()
    runtime.step()

    restored_z = runtime.state.body_q.numpy()[cube.body_index, 2]
    assert abs(restored_z - initial_z) < 1e-4


def test_far_object_not_grasped_in_empty_air(tray_cubes_scene):
    """Ensure distant dynamic objects are not qualified for grasp."""
    commands = {"PSM1": CommandMailboxes()}
    options = NewtonRuntimeOptions(
        device="cuda:0",
        headless=True,
        scene_objects=tuple(tray_cubes_scene.objects),
    )
    runtime = NewtonRuntime(tray_cubes_scene.robots, options, commands)
    runtime.initialize()

    # Initial step: cube is between jaws and gets grasped; grasp_cube is 35 mm away and ignored
    runtime.step()
    assert "PSM1" in runtime.grasp_manager.attachments
    assert runtime.grasp_manager.attachments["PSM1"].object_name == "cube"

    # Open jaws to release cube
    commands["PSM1"].submit_servo("jaw/servo_jp", 0.2)
    runtime.step()
    assert len(runtime.grasp_manager.attachments) == 0

    # Move arm away into empty space
    cur_pose = runtime.snapshots()["PSM1"].measured_cp_world
    target_pos = cur_pose.position + np.array([0.05, 0.05, 0.05])
    ik_res = runtime.compute_ik("PSM1", (target_pos, cur_pose.orientation))
    assert ik_res.success
    commands["PSM1"].submit_servo("servo_jp", ik_res.position)
    runtime.step()

    # Close jaws while in empty air; neither cube nor grasp_cube should be grasped
    commands["PSM1"].submit_servo("jaw/servo_jp", 0.0)
    runtime.step()
    assert len(runtime.grasp_manager.attachments) == 0


def test_peg_board_cuhk_grasp_distance_qualification():
    """triangular_block in peg_board_CUHK is 35 mm away at home and must not be grasped until arm moves to it."""
    resolved = resolve_scene_path("/tmp/test.yaml", ["PSM1_420006.yaml", "peg_board_CUHK.yaml"])
    scene = load_installed_scene_config(resolved)
    commands = {"PSM1": CommandMailboxes()}
    options = NewtonRuntimeOptions(
        device="cuda:0",
        headless=True,
        scene_objects=tuple(scene.objects),
    )
    runtime = NewtonRuntime(scene.robots, options, commands)
    runtime.initialize()

    # 1. At home position, jaws closed: distant triangular_block is NOT grasped
    runtime.step()
    assert len(runtime.grasp_manager.attachments) == 0, (
        f"triangular_block 35 mm away must not be grasped at home: {runtime.grasp_manager.attachments}"
    )

    # 2. Open jaws and command arm to triangular block
    commands["PSM1"].submit_servo("jaw/servo_jp", 0.3)
    runtime.step()

    block = runtime.scene_objects["triangular_block"]
    block_pos = runtime.state.body_q.numpy()[block.body_index, :3]
    cur_pose = runtime.snapshots()["PSM1"].measured_cp_world
    target_pos = block_pos.copy()
    target_pos[2] += 0.003
    ik_res = runtime.compute_ik("PSM1", (target_pos, cur_pose.orientation))
    assert ik_res.success
    commands["PSM1"].submit_servo("servo_jp", ik_res.position)
    runtime.step()

    for _ in range(10):
        runtime.step()

    # 3. Close jaws on triangular block: successfully grasps
    commands["PSM1"].submit_servo("jaw/servo_jp", 0.0)
    runtime.step()
    assert "PSM1" in runtime.grasp_manager.attachments
    assert runtime.grasp_manager.attachments["PSM1"].object_name == "triangular_block"

    # 4. Lift arm: triangular_block moves with it
    lift_pos = target_pos + np.array([0.0, 0.0, 0.02])
    ik_lift = runtime.compute_ik("PSM1", (lift_pos, cur_pose.orientation))
    assert ik_lift.success
    commands["PSM1"].submit_servo("servo_jp", ik_lift.position)
    runtime.step()

    lifted_z = runtime.state.body_q.numpy()[block.body_index, 2]
    assert lifted_z > block_pos[2] + 0.01

    # 5. Open jaws: releases
    commands["PSM1"].submit_servo("jaw/servo_jp", 0.3)
    runtime.step()
    assert "PSM1" not in runtime.grasp_manager.attachments

