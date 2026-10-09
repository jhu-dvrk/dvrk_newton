"""NVIDIA Newton physics and kinematics runtime for dVRK robots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np

from dvrk_arm_description import RobotConfig
from dvrk_simulator_base.command_mailbox import CommandMailboxes
from dvrk_simulator_base.cartesian_command import CartesianCommand, resolve_cartesian_command
from dvrk_simulator_base.publication_frames import with_publication_frames
from dvrk_simulator_base.arm_controller import ArmController
from dvrk_simulator_base.rotations import quaternion_matrix_xyzw
from dvrk_simulator_base.scene import SceneObject
from dvrk_simulator_base.snapshots import ArmSnapshot, OperatingStateSnapshot
from dvrk_simulator_base.types import IKResult, JointState, Pose, Twist

from .backend import load_newton
from .camera import CameraOptions, NewtonCameraRenderer
from .camera_gl import NewtonOpenGLCameraRenderer
from .camera_worker import NewtonCameraWorker
from .configuration import GraspConfig
from .errors import NewtonBackendError
from .grasp import NewtonGraspManager
from .robot import (
    LoadedRobot,
    add_robot_to_builder,
    build_robot_mapping,
)
from .scene_objects import LoadedSceneObject, add_scene_objects_to_builder
from .urdf_materializer import MaterializedUrdf, materialize_virtual_robot
from dvrk_simulator_base.urdf_chain import UrdfChain
from .video import UnixFdVideoSink


@dataclass(frozen=True)
class NewtonRuntimeOptions:
    device: str = "cuda:0"
    headless: bool = False
    simulation_rate_hz: float = 120.0
    generated_root: Path | None = None
    camera_options: CameraOptions | None = None
    scene_objects: tuple[SceneObject, ...] = ()
    grasp_config: GraspConfig | None = None
    rigid_gap_m: float = 0.005


class _NewtonArmState(ArmController):
    """Common command state plus Newton's robot and kinematic artifacts."""

    def __init__(self, config, commands):
        super().__init__(config, commands)
        self.artifact: MaterializedUrdf | None = None
        self.robot: LoadedRobot | None = None
        self.ik_chain: UrdfChain | None = None


class NewtonRuntime:
    """Manages the NVIDIA Newton model, simulation steps, and CRTK arm states."""

    def __init__(
        self,
        configs: Sequence[RobotConfig],
        options: NewtonRuntimeOptions,
        commands: Mapping[str, CommandMailboxes],
    ) -> None:
        if not configs:
            raise ValueError("NewtonRuntime requires at least one robot configuration")
        if options.simulation_rate_hz <= 0.0:
            raise ValueError("simulation_rate_hz must be positive")

        self.options = options
        self.newton, self.wp = load_newton()
        self.device = options.device

        self.arms: dict[str, _NewtonArmState] = {
            cfg.name: _NewtonArmState(cfg, commands[cfg.name]) for cfg in configs
        }

        self.builder = None
        self.model = None
        self.state = None
        self._state_out = None
        self.control = None
        self.solver = None
        self.collision = None
        self.contacts = None
        self.scene_objects: dict[str, LoadedSceneObject] = {}
        self.grasp_manager: NewtonGraspManager | None = None
        self._dynamic_body_indices: list[int] = []
        self._initial_body_q: np.ndarray | None = None
        self._initial_body_qd: np.ndarray | None = None
        self._reset_requested = False

        self.viewer = None
        self.camera_renderer: NewtonCameraRenderer | NewtonOpenGLCameraRenderer | None = None
        self.camera_sink: UnixFdVideoSink | None = None
        self.camera_worker: NewtonCameraWorker | None = None
        self._camera_model = None
        self._camera_interval = 0.0
        self._next_camera_offer_at = 0.0
        self._q_buffer = None
        self._sequence = 0
        self._simulation_time = 0.0
        self._is_initialized = False
        self._frame_snapshots: dict[str, ArmSnapshot] = {}
        self.command_warnings: list[tuple[str, str]] = []

    def initialize(self) -> dict[str, ArmSnapshot]:
        """Materialize URDFs, build unified Newton model, and evaluate initial FK."""
        self.builder = self.newton.ModelBuilder()
        self.builder.gravity = (0.0, 0.0, -9.81)
        self.builder.rigid_gap = self.options.rigid_gap_m

        for arm_name, arm in self.arms.items():
            cfg = arm.config
            arm.artifact = materialize_virtual_robot(
                cfg.name,
                instrument=cfg.instrument,
                endoscope=cfg.endoscope,
                generated_root=self.options.generated_root,
            )
            add_robot_to_builder(
                self.builder,
                arm.artifact.urdf_path,
                base_position=cfg.base_position,
                base_orientation_xyzw=cfg.base_orientation_xyzw,
                enable_self_collisions=False,
            )

        # Mark all robot bodies as kinematic
        for b_idx in range(self.builder.body_count):
            self.builder.body_flags[b_idx] = self.newton.BodyFlags.KINEMATIC

        # Add scene objects
        self.scene_objects = add_scene_objects_to_builder(
            self.builder, self.options.scene_objects
        )

        self.model = self.builder.finalize(self.device)
        self.state = self.model.state()
        self._state_out = self.model.state()
        self.control = self.model.control()
        self._q_buffer = self.state.joint_q.numpy()
        self._dynamic_body_indices = [
            obj.body_index for obj in self.scene_objects.values() if obj.is_dynamic
        ]
        self._dynamic_body_joint_q: dict[int, int] = {}
        for j_idx in range(self.builder.joint_count):
            child_b = self.builder.joint_child[j_idx]
            if child_b in self._dynamic_body_indices:
                self._dynamic_body_joint_q[child_b] = self.builder.joint_q_start[j_idx]

        # Collision pipeline and solver for dynamic objects
        self.collision = self.newton.CollisionPipeline(self.model)
        self.contacts = self.collision.contacts()
        if self._dynamic_body_indices:
            self.solver = self.newton.solvers.SolverXPBD(self.model)

        for arm_name, arm in self.arms.items():
            cfg = arm.config
            arm.robot = build_robot_mapping(
                self.model,
                cfg.name,
                arm.artifact.urdf_path,
                (j.name for j in cfg.joints),
                tool_frame=cfg.tool_frame,
            )
            # Apply initial joint setpoints
            for i, q_idx in enumerate(arm.robot.controlled_q_indices):
                self._q_buffer[q_idx] = arm.joint_setpoint[i]
            # Apply jaw and mimic joints
            if arm.robot.jaw_q_index is not None:
                self._q_buffer[arm.robot.jaw_q_index] = arm.jaw_setpoint
            for m in arm.robot.mimic_joints:
                self._q_buffer[m.q_index] = m.multiplier * self._q_buffer[m.source_q_index] + m.offset

        # Commit to GPU state and evaluate forward kinematics
        self.state.joint_q.assign(self.wp.array(self._q_buffer, dtype=float, device=self.device))
        self.newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)

        # Initialize grasp manager
        self.grasp_manager = NewtonGraspManager(
            self.model,
            self.arms,
            self.scene_objects,
            self.options.grasp_config,
        )

        # Cache initial state for resets
        self._initial_body_q = self.state.body_q.numpy().copy()
        self._initial_body_qd = self.state.body_qd.numpy().copy()
        self._initialize_ik_chains()
        self._is_initialized = True

        if not self.options.headless:
            try:
                from newton.viewer import ViewerGL
                from pyglet.math import Vec3
                self.viewer = ViewerGL(width=1280, height=720, headless=False)
                self.viewer.set_model(self.model)
                self.viewer.camera.pos = Vec3(0.5, -0.8, 0.4)
                self.viewer.camera.look_at(Vec3(0.0, 0.0, 0.1))
                self.viewer.begin_frame(0.0)
                self.viewer.log_state(self.state)
                self.viewer.end_frame()
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to initialize Newton ViewerGL: {e}")
                self.viewer = None

        if self.options.camera_options is not None and "ECM" in self.arms:
            # The camera worker owns its model and state. The Warp renderer
            # refits a mutable BVH, and ViewerGL caches render geometry.
            self._camera_model = self.builder.finalize(self.device)
            if (
                self._camera_model.body_count != self.model.body_count
                or self._camera_model.particle_count != self.model.particle_count
            ):
                raise NewtonBackendError("camera model does not match simulation model")
            renderer_type = (
                NewtonOpenGLCameraRenderer
                if self.options.camera_options.renderer == "opengl"
                else NewtonCameraRenderer
            )
            self.camera_renderer = renderer_type(
                self._camera_model, self.options.camera_options, device=self.device
            )
            self.camera_sink = UnixFdVideoSink(self.options.camera_options)
            self.camera_sink.start()
            self.camera_worker = NewtonCameraWorker(
                self.camera_renderer, self._camera_model.state(), self.camera_sink
            )
            self.camera_worker.start()
            self._camera_interval = 1.0 / self.options.camera_options.rate_hz
            self._next_camera_offer_at = 0.0

        self._frame_snapshots = with_publication_frames(
            self.snapshots(), [arm.config for arm in self.arms.values()]
        )
        return self._frame_snapshots

    def _initialize_ik_chains(self) -> None:
        """Use CPU IK only when its URDF poses agree with Newton at startup."""
        import warnings

        validation_state = self.model.state()
        for arm_name, arm in self.arms.items():
            try:
                tool_label = self.model.body_label[arm.robot.tool_link_index].split("/")[-1]
                chain = UrdfChain(
                    arm.artifact.urdf_path,
                    tool_label,
                    tuple(joint.name for joint in arm.config.joints),
                    arm.config.base_position,
                    arm.config.base_orientation_xyzw,
                )

                def matches_newton(q: np.ndarray, body_tf: np.ndarray) -> bool:
                    position, rotation, _ = chain.forward(q)
                    return bool(
                        np.linalg.norm(position - body_tf[:3]) < 5e-4
                        and np.linalg.norm(
                            rotation - quaternion_matrix_xyzw(body_tf[3:])
                        ) < 5e-3
                    )

                if not matches_newton(
                    arm.joint_setpoint,
                    self._initial_body_q[arm.robot.tool_link_index],
                ):
                    raise ValueError("home pose differs from Newton FK")

                probe = arm.joint_setpoint.copy()
                for index, joint in enumerate(arm.config.joints):
                    amount = 0.002 if joint.type == "prismatic" else 0.02
                    if joint.upper - probe[index] >= amount:
                        probe[index] += amount
                    elif probe[index] - joint.lower >= amount:
                        probe[index] -= amount
                if not np.array_equal(probe, arm.joint_setpoint):
                    probe_joint_q = self._q_buffer.copy()
                    for index, q_index in enumerate(arm.robot.controlled_q_indices):
                        probe_joint_q[q_index] = probe[index]
                    validation_state.joint_q.assign(
                        self.wp.array(probe_joint_q, dtype=float, device=self.device)
                    )
                    self.newton.eval_fk(
                        self.model,
                        validation_state.joint_q,
                        validation_state.joint_qd,
                        validation_state,
                    )
                    body_tf = validation_state.body_q.numpy()[arm.robot.tool_link_index]
                    if not matches_newton(probe, body_tf):
                        raise ValueError("perturbed pose differs from Newton FK")
                arm.ik_chain = chain
            except (ValueError, KeyError, ET.ParseError) as error:
                warnings.warn(
                    f"{arm_name}: CPU IK unavailable ({error}); using slower Newton FK IK",
                    stacklevel=2,
                )

    def is_connected(self) -> bool:
        return self._is_initialized and (self.viewer is None or self.viewer.is_running())

    def prepare_step(self, now: float) -> None:
        """Process incoming CRTK commands and advance trajectories."""
        for arm_name, arm in self.arms.items():
            arm.advance_commands(
                now,
                lambda target, seed: self.compute_ik(arm_name, target, seed),
                lambda target: self.resolve_cartesian_target(arm_name, target),
            )
            self.command_warnings.extend((arm_name, warning) for warning in arm.command_warnings)
            arm.command_warnings.clear()
            for i, q_idx in enumerate(arm.robot.controlled_q_indices):
                self._q_buffer[q_idx] = arm.joint_setpoint[i]
            if arm.robot.jaw_q_index is not None:
                self._q_buffer[arm.robot.jaw_q_index] = arm.jaw_setpoint
            for m in arm.robot.mimic_joints:
                self._q_buffer[m.q_index] = m.multiplier * self._q_buffer[m.source_q_index] + m.offset

    def request_reset(self) -> None:
        self._reset_requested = True

    def _reset_scene(self) -> None:
        if self.grasp_manager is not None:
            self.grasp_manager.release_all()
        for arm in self.arms.values():
            arm.reset_motion()
            for i, q_idx in enumerate(arm.robot.controlled_q_indices):
                self._q_buffer[q_idx] = arm.joint_setpoint[i]
            if arm.robot.jaw_q_index is not None:
                self._q_buffer[arm.robot.jaw_q_index] = arm.jaw_setpoint
            for m in arm.robot.mimic_joints:
                self._q_buffer[m.q_index] = m.multiplier * self._q_buffer[m.source_q_index] + m.offset

        if self._initial_body_q is not None and self._initial_body_qd is not None:
            self.state.body_q.assign(
                self.wp.array(self._initial_body_q, dtype=self.wp.transform, device=self.device)
            )
            self.state.body_qd.assign(
                self.wp.array(self._initial_body_qd, dtype=self.wp.spatial_vector, device=self.device)
            )
            for b_idx, q_start in self._dynamic_body_joint_q.items():
                self._q_buffer[q_start : q_start + 7] = self._initial_body_q[b_idx]
        self.state.joint_q.assign(self.wp.array(self._q_buffer, dtype=float, device=self.device))
        self.newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)

    def step(self) -> dict[str, ArmSnapshot]:
        """Consume commands and advance one complete scene step."""
        if self._reset_requested:
            self._reset_scene()
            self._reset_requested = False
        self.prepare_step(time.monotonic())
        self._frame_snapshots = self.finish_step()
        return self._frame_snapshots

    def finish_step(self) -> dict[str, ArmSnapshot]:
        """Apply joint configuration to state, compute FK, step dynamics & grasp, and create snapshots."""
        joint_qd = self.state.joint_qd.numpy()
        joint_qd.fill(0.0)
        for arm in self.arms.values():
            for i, q_idx in enumerate(arm.robot.controlled_q_indices):
                joint_qd[q_idx] = arm.joint_velocity[i]
            if arm.robot.jaw_q_index is not None:
                joint_qd[arm.robot.jaw_q_index] = arm.jaw_velocity
            for m in arm.robot.mimic_joints:
                joint_qd[m.q_index] = m.multiplier * joint_qd[m.source_q_index]
        self.state.joint_q.assign(self.wp.array(self._q_buffer, dtype=float, device=self.device))
        self.state.joint_qd.assign(self.wp.array(joint_qd, dtype=float, device=self.device))
        self.newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
        snapshots = self.snapshots()
        if self.collision is not None:
            self.collision.collide(self.state, self.contacts)
        body_q_np = self.state.body_q.numpy()
        body_qd_np = self.state.body_qd.numpy()
        if self.grasp_manager is not None:
            self.grasp_manager.step(snapshots, self.contacts, body_q_np, body_qd_np)
            self.state.body_q.assign(self.wp.array(body_q_np, dtype=self.wp.transform, device=self.device))
            self.state.body_qd.assign(self.wp.array(body_qd_np, dtype=self.wp.spatial_vector, device=self.device))
        if self.solver is not None and self._dynamic_body_indices:
            dt = 1.0 / self.options.simulation_rate_hz
            self.solver.step(self.state, self._state_out, self.control, self.contacts, dt)
            out_q = self._state_out.body_q.numpy()
            out_qd = self._state_out.body_qd.numpy()
            grasped_objects = {att.object_body_index for att in (self.grasp_manager.attachments.values() if self.grasp_manager else ())}
            for b_idx in self._dynamic_body_indices:
                if b_idx not in grasped_objects:
                    body_q_np[b_idx] = out_q[b_idx]
                    body_qd_np[b_idx] = out_qd[b_idx]
            if self.grasp_manager is not None and self.grasp_manager.attachments:
                self.grasp_manager.step(snapshots, None, body_q_np, body_qd_np)
            for b_idx, q_start in self._dynamic_body_joint_q.items():
                self._q_buffer[q_start:q_start + 7] = body_q_np[b_idx]
            self.state.body_q.assign(self.wp.array(body_q_np, dtype=self.wp.transform, device=self.device))
            self.state.body_qd.assign(self.wp.array(body_qd_np, dtype=self.wp.spatial_vector, device=self.device))
        self._simulation_time += 1.0 / self.options.simulation_rate_hz
        self._sequence += 1
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.begin_frame(self._simulation_time)
            self.viewer.log_state(self.state)
            self.viewer.end_frame()
        snapshots = self.snapshots(body_q_np)
        if self.camera_worker is not None:
            self.camera_worker.raise_if_failed()
        if self.camera_worker is not None and 'ECM' in snapshots and (snapshots['ECM'].measured_cp_world is not None):
            now = time.monotonic()
            if now >= self._next_camera_offer_at:
                particle_q = self.state.particle_q.numpy() if self.model.particle_count else None
                self.camera_worker.offer(body_q_np, particle_q, snapshots['ECM'].measured_cp_world, self._simulation_time)
                self._next_camera_offer_at += self._camera_interval
                if self._next_camera_offer_at <= now:
                    self._next_camera_offer_at = now + self._camera_interval
        for arm in self.arms.values():
            arm.operating_state_event_pending = False
        return with_publication_frames(snapshots, [arm.config for arm in self.arms.values()])

    def snapshots(self, body_poses: np.ndarray | None = None) -> dict[str, ArmSnapshot]:
        """Extract ArmSnapshot for each loaded robot arm."""
        if body_poses is None:
            body_poses = self.state.body_q.numpy()
        result: dict[str, ArmSnapshot] = {}

        for arm_name, arm in self.arms.items():
            robot = arm.robot
            names = tuple(joint.name for joint in arm.config.joints)
            positions = arm.joint_setpoint.copy()
            velocities = arm.joint_velocity.copy()

            measured_js = JointState(names, positions, velocities)
            setpoint_js = JointState(names, arm.joint_setpoint.copy(), arm.joint_velocity.copy())

            # Tool tip link pose from Newton FK
            tf = body_poses[robot.tool_link_index]
            pos = np.asarray(tf[:3], dtype=float)
            rot = quaternion_matrix_xyzw(tf[3:])
            pose = Pose(pos, rot)
            twist = Twist(np.zeros(3, dtype=float), np.zeros(3, dtype=float))

            operating_state_snapshot = OperatingStateSnapshot(
                state=arm.operating_state.state,
                is_homed=arm.operating_state.is_homed,
                is_busy=(
                    arm.joint_trajectory is not None
                    or arm.jaw_trajectory is not None
                    or arm.move_failure_pending
                ),
            )

            result[arm_name] = ArmSnapshot(
                sequence=self._sequence,
                simulation_time=self._simulation_time,
                valid=True,
                measured_js=measured_js,
                setpoint_js=setpoint_js,
                measured_cp_world=pose,
                setpoint_cp_world=pose,
                measured_cv_world=twist,
                jaw_measured=float(arm.jaw_setpoint) if robot.jaw_q_index is not None else None,
                jaw_setpoint=float(arm.jaw_setpoint) if robot.jaw_q_index is not None else None,
                operating_state=operating_state_snapshot,
                operating_state_event=arm.operating_state_event_pending,
            )

        return result

    def resolve_cartesian_target(self, arm_name: str, command: CartesianCommand) -> Pose:
        """Use the completed scene at this step's start, independent of ROS lag.

        All commands in a step use the same measured ECM pose. An ECM command
        received alongside a PSM command takes effect in the subsequent state.
        """
        config = self.arms[arm_name].config
        ecm = next((arm for arm in self.arms.values() if arm.config.type == "ECM"), None)
        has_ecm = config.type == "PSM" and ecm is not None
        snapshot = self._frame_snapshots.get(ecm.config.name) if has_ecm else None
        return resolve_cartesian_command(
            command, config, None if snapshot is None else snapshot.measured_cp_world,
            has_ecm=has_ecm,
        )

    def compute_ik(
        self,
        arm_name: str,
        target_pose: Pose | tuple[Iterable[float], Iterable[float]],
        seed: Iterable[float] | None = None,
        *,
        max_iterations: int = 30,
        position_tolerance_m: float = 1e-4,
        orientation_tolerance_rad: float = 1e-3,
    ) -> IKResult:
        """Solve Cartesian IK using validated CPU kinematics or Newton FK."""
        arm = self.arms.get(arm_name)
        if arm is None or arm.robot is None:
            raise NewtonBackendError(f"robot {arm_name} is not loaded")

        if isinstance(target_pose, Pose):
            target_pos = target_pose.position
            target_rot = target_pose.orientation
        else:
            target_pos = np.asarray(target_pose[0], dtype=float)
            target_rot = (
                np.asarray(target_pose[1], dtype=float)
                if np.asarray(target_pose[1]).shape == (3, 3)
                else quaternion_matrix_xyzw(target_pose[1])
            )

        q = np.array(seed if seed is not None else arm.joint_setpoint, dtype=float, copy=True)
        lower = arm.lower_limits
        upper = arm.upper_limits
        controlled_q = arm.robot.controlled_q_indices
        tool_idx = arm.robot.tool_link_index

        if arm.ik_chain is not None:
            pos_err = 0.0
            rot_err = 0.0
            for iteration in range(max_iterations):
                pos, rot, jacobian = arm.ik_chain.forward(q)
                dp = target_pos - pos
                dr = 0.5 * (
                    np.cross(rot[:, 0], target_rot[:, 0])
                    + np.cross(rot[:, 1], target_rot[:, 1])
                    + np.cross(rot[:, 2], target_rot[:, 2])
                )
                pos_err = float(np.linalg.norm(dp))
                rot_err = float(np.linalg.norm(dr))
                if pos_err <= position_tolerance_m and rot_err <= orientation_tolerance_rad:
                    return IKResult(q, True, iteration, pos_err, rot_err, "converged")

                error = np.concatenate((dp, dr))
                damping = 1e-4 * np.eye(6)
                delta = jacobian.T @ np.linalg.solve(
                    jacobian @ jacobian.T + damping, error
                )
                q = np.clip(q + np.clip(delta, -0.05, 0.05), lower, upper)
            return IKResult(q, False, max_iterations, pos_err, rot_err, "did not converge")

        temp_q = self._q_buffer.copy()

        def fk(q_cand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            for i, idx in enumerate(controlled_q):
                temp_q[idx] = q_cand[i]
            self.state.joint_q.assign(self.wp.array(temp_q, dtype=float, device=self.device))
            self.newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
            body_tf = self.state.body_q.numpy()[tool_idx]
            return body_tf[:3], quaternion_matrix_xyzw(body_tf[3:])

        pos_err = 0.0
        rot_err = 0.0
        try:
            for iteration in range(max_iterations):
                pos, rot = fk(q)
                dp = pos - target_pos
                dr = 0.5 * (
                    np.cross(rot[:, 0], target_rot[:, 0])
                    + np.cross(rot[:, 1], target_rot[:, 1])
                    + np.cross(rot[:, 2], target_rot[:, 2])
                )
                err = np.concatenate([dp, dr])
                pos_err = float(np.linalg.norm(dp))
                rot_err = float(np.linalg.norm(dr))

                if pos_err <= position_tolerance_m and rot_err <= orientation_tolerance_rad:
                    return IKResult(q, True, iteration, pos_err, rot_err, "converged")

                # Numerical Jacobian via finite differences
                J = np.zeros((6, len(q)), dtype=float)
                eps = 1e-5
                for j in range(len(q)):
                    q_pert = q.copy()
                    q_pert[j] += eps
                    p_p, r_p = fk(q_pert)
                    e_p = np.concatenate([
                        p_p - target_pos,
                        0.5 * (
                            np.cross(r_p[:, 0], target_rot[:, 0])
                            + np.cross(r_p[:, 1], target_rot[:, 1])
                            + np.cross(r_p[:, 2], target_rot[:, 2])
                        ),
                    ])
                    J[:, j] = (e_p - err) / eps

                damping = 1e-4 * np.eye(6)
                delta = -J.T @ np.linalg.solve(J @ J.T + damping, err)
                delta = np.clip(delta, -0.05, 0.05)
                q = np.clip(q + delta, lower, upper)
        finally:
            # Restore state joint_q buffer
            self.state.joint_q.assign(self.wp.array(self._q_buffer, dtype=float, device=self.device))
            self.newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)

        return IKResult(q, False, max_iterations, pos_err, rot_err, "did not converge")

    def take_camera_rate_hz(self) -> float:
        """Return the camera frames pushed to the video sink over the last interval."""
        return self.camera_worker.take_rate_hz() if self.camera_worker else 0.0

    def shutdown(self) -> None:
        """Cleanup runtime resources."""
        if self.camera_worker is not None:
            self.camera_worker.close()
            self.camera_worker = None
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.camera_sink is not None:
            self.camera_sink.close()
            self.camera_sink = None
        self.camera_renderer = None
        self._camera_model = None
        self._is_initialized = False
        self.model = None
        self.state = None
        self.builder = None
