"""Unit tests for dvrk_newton materializer, robot mapping, and runtime."""

import numpy as np
import pytest

from dvrk_simulator_base.command_mailbox import CommandMailboxes
from dvrk_newton.configuration import load_installed_robot_config
from dvrk_newton.robot import add_robot_to_builder, build_robot_mapping
from dvrk_newton.runtime import NewtonRuntime, NewtonRuntimeOptions
from dvrk_newton.urdf_materializer import materialize_virtual_robot


@pytest.fixture
def psm1_config():
    return load_installed_robot_config("PSM1", "420006")


@pytest.fixture
def ecm_config():
    return load_installed_robot_config("ECM", "Si_straight")


def test_materializer_psm_and_ecm():
    psm_art = materialize_virtual_robot("PSM1", instrument="420006")
    assert psm_art.urdf_path.is_file()
    assert psm_art.metadata_path.is_file()

    ecm_art = materialize_virtual_robot("ECM", endoscope="Si_straight")
    assert ecm_art.urdf_path.is_file()
    assert ecm_art.metadata_path.is_file()


def test_robot_mapping_psm1(psm1_config):
    import newton
    art = materialize_virtual_robot("PSM1", instrument="420006")
    builder = newton.ModelBuilder()
    add_robot_to_builder(builder, art.urdf_path, base_position=psm1_config.base_position)
    model = builder.finalize("cpu")

    loaded = build_robot_mapping(
        model, "PSM1", art.urdf_path,
        [j.name for j in psm1_config.joints],
        tool_frame=psm1_config.tool_frame,
    )
    assert len(loaded.controlled_joint_names) == 6
    assert len(loaded.controlled_q_indices) == 6
    assert loaded.jaw_joint_name == "jaw"
    assert loaded.jaw_q_index is not None
    assert len(loaded.mimic_joints) == 2
    assert loaded.tool_link_index >= 0


def test_runtime_stepping_and_servo_jp(psm1_config):
    commands = {"PSM1": CommandMailboxes()}
    options = NewtonRuntimeOptions(device="cuda:0", simulation_rate_hz=120.0, headless=True)
    runtime = NewtonRuntime([psm1_config], options, commands)

    initial_snapshots = runtime.initialize()
    assert "PSM1" in initial_snapshots
    snap0 = initial_snapshots["PSM1"]
    assert snap0.valid
    assert len(snap0.measured_js.position) == 6
    assert snap0.jaw_measured is not None

    # Post servo_jp command
    target_q = [0.05, 0.02, 0.14, 0.1, -0.05, 0.02]
    commands["PSM1"].submit_servo("servo_jp", target_q)
    snap1 = runtime.step()["PSM1"]

    np.testing.assert_allclose(snap1.measured_js.position, target_q, atol=1e-5)
    assert snap1.measured_cp_world.position.shape == (3,)
    assert snap1.measured_cp_world.orientation.shape == (3, 3)


def test_runtime_inverse_kinematics(psm1_config):
    commands = {"PSM1": CommandMailboxes()}
    options = NewtonRuntimeOptions(device="cuda:0", simulation_rate_hz=120.0, headless=True)
    runtime = NewtonRuntime([psm1_config], options, commands)
    snap0 = runtime.initialize()["PSM1"]

    initial_pose = snap0.measured_cp_world
    # Perturb target pose by 1cm along X
    target_pos = initial_pose.position + np.array([0.01, 0.0, 0.0])
    target_pose = (target_pos, initial_pose.orientation)

    ik_res = runtime.compute_ik("PSM1", target_pose)
    assert ik_res.success
    assert ik_res.position_error < 1e-4


def test_simulator_config_headless(tmp_path):
    from dvrk_newton.configuration import load_simulator_config

    cfg_file = tmp_path / "sim.yaml"
    cfg_file.write_text("headless: true\n")
    cfg = load_simulator_config(cfg_file)
    assert cfg.headless is True

    cfg_file.write_text("headless: false\n")
    cfg = load_simulator_config(cfg_file)
    assert cfg.headless is False

