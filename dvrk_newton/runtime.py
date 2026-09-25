"""NVIDIA Newton physics and kinematics runtime for dVRK robots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np

from dvrk_arm_description import RobotConfig
from dvrk_simulator_base.command_mailbox import CommandMailboxes
from dvrk_simulator_base.operating_state import CRTKOperatingState
from dvrk_simulator_base.rotations import quaternion_matrix_xyzw, rotation_to_quaternion_xyzw
from dvrk_simulator_base.scene import SceneObject
from dvrk_simulator_base.snapshots import ArmSnapshot, OperatingStateSnapshot
from dvrk_simulator_base.trajectory import JointTrajectory
from dvrk_simulator_base.types import IKResult, JointState, Pose, Twist

from .backend import load_newton
from .camera import CameraOptions, NewtonCameraRenderer
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


class _NewtonArmState:
    """Per-arm state and command tracking."""

    def __init__(
        self,
        config: RobotConfig,
        commands: CommandMailboxes,
    ) -> None:
        self.config = config
        self.commands = commands
        self.artifact: MaterializedUrdf | None = None
        self.robot: LoadedRobot | None = None

        self.joint_setpoint = np.array(config.home_position, dtype=float, copy=True)
        self.joint_velocity = np.zeros_like(self.joint_setpoint)
        self.jaw_setpoint = 0.0
        self.jaw_velocity = 0.0

        self.joint_trajectory: JointTrajectory | None = None
        self.jaw_trajectory: JointTrajectory | None = None

        self.operating_state = CRTKOperatingState(CRTKOperatingState.ENABLED)
        self.operating_state_event_pending = False
        self.move_failure_pending = False

        jaw_raw = config.raw.get("robot", {}).get("jaw", {})
        self.jaw_lower = float(jaw_raw.get("lower", -0.349066))
        self.jaw_upper = float(jaw_raw.get("upper", 1.39626))
        self.jaw_speed = float(jaw_raw.get("velocity", 0.4))

        self.commands_applied = 0
        self.commands_rejected = 0
        self.commands_canceled = 0

        self.lower_limits = np.array([j.lower for j in config.joints], dtype=float)
        self.upper_limits = np.array([j.upper for j in config.joints], dtype=float)
        self.max_velocities = np.array([j.velocity for j in config.joints], dtype=float)

    def valid_joint_target(self, target: np.ndarray) -> bool:
        if len(target) != len(self.config.joints):
            return False
        return bool(
            np.all(np.isfinite(target))
            and np.all(target >= self.lower_limits - 1e-6)
            and np.all(target <= self.upper_limits + 1e-6)
        )

    def cancel_motion(self) -> None:
        if self.joint_trajectory is not None or self.jaw_trajectory is not None:
            self.commands_canceled += 1
        self.joint_trajectory = None
        self.jaw_trajectory = None
        self.joint_velocity.fill(0.0)
        self.jaw_velocity = 0.0


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
        self.camera_renderer: NewtonCameraRenderer | None = None
        self.camera_sink: UnixFdVideoSink | None = None
        self._last_camera_render_time = -1.0
        self._camera_interval = 0.0
        self._q_buffer = None
        self._sequence = 0
        self._simulation_time = 0.0
        self._is_initialized = False

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
        self._is_initialized = True

        if not self.options.headless:
            try:
                from newton.viewer import ViewerGL
                self.viewer = ViewerGL(width=1280, height=720, headless=False)
                self.viewer.set_model(self.model)
                self.viewer.camera.pos = self.wp.vec3(0.5, -0.8, 0.4)
                self.viewer.camera.look_at(self.wp.vec3(0.0, 0.0, 0.1))
                self.viewer.begin_frame(0.0)
                self.viewer.log_state(self.state)
                self.viewer.end_frame()
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to initialize Newton ViewerGL: {e}")
                self.viewer = None

        if self.options.camera_options is not None and "ECM" in self.arms:
            self.camera_renderer = NewtonCameraRenderer(
                self.model, self.options.camera_options, device=self.device
            )
            self.camera_sink = UnixFdVideoSink(self.options.camera_options)
            self.camera_sink.start()
            self._camera_interval = 1.0 / self.options.camera_options.rate_hz
            self._last_camera_render_time = -1.0

        return self.snapshots()

    def is_connected(self) -> bool:
        return self._is_initialized and (self.viewer is None or self.viewer.is_running())

    def prepare_step(self, now_ns: int, now: float) -> None:
        """Process incoming CRTK commands and advance trajectories."""
        dt = 1.0 / self.options.simulation_rate_hz

        for arm_name, arm in self.arms.items():
            if arm.move_failure_pending:
                arm.move_failure_pending = False

            # Drain commands
            for command in arm.commands.drain():
                if command.channel == "state_command":
                    success, _ = arm.operating_state.command(command.payload)
                    if not success:
                        arm.commands_rejected += 1
                        continue
                    arm.commands_applied += 1
                    arm.operating_state_event_pending = True
                    if not arm.operating_state.accepts_motion:
                        arm.cancel_motion()
                    continue

                if not arm.operating_state.accepts_motion:
                    arm.commands_rejected += 1
                    if command.channel in {"move_jp", "move_cp", "jaw/move_jp"}:
                        arm.move_failure_pending = True
                    continue

                # Joint position commands
                if command.channel in {"servo_jp", "move_jp"}:
                    target = np.asarray(command.payload, dtype=float)
                    if not arm.valid_joint_target(target):
                        arm.commands_rejected += 1
                        if command.channel == "move_jp":
                            arm.move_failure_pending = True
                        continue
                    if command.channel == "servo_jp":
                        if arm.joint_trajectory is not None:
                            arm.commands_canceled += 1
                        arm.joint_trajectory = None
                        arm.joint_setpoint = target.copy()
                        arm.joint_velocity.fill(0.0)
                    else:
                        if arm.joint_trajectory is not None:
                            arm.commands_canceled += 1
                        arm.joint_trajectory = JointTrajectory(
                            arm.joint_setpoint,
                            target,
                            arm.max_velocities,
                            now,
                        )
                    arm.commands_applied += 1
                    continue

                # Cartesian position commands
                if command.channel in {"servo_cp", "move_cp"}:
                    ik_res = self.compute_ik(arm_name, command.payload, arm.joint_setpoint)
                    if not ik_res.success or not arm.valid_joint_target(ik_res.position):
                        arm.commands_rejected += 1
                        if command.channel == "move_cp":
                            arm.move_failure_pending = True
                        continue
                    if command.channel == "servo_cp":
                        if arm.joint_trajectory is not None:
                            arm.commands_canceled += 1
                        arm.joint_trajectory = None
                        arm.joint_setpoint = ik_res.position.copy()
                        arm.joint_velocity.fill(0.0)
                    else:
                        if arm.joint_trajectory is not None:
                            arm.commands_canceled += 1
                        arm.joint_trajectory = JointTrajectory(
                            arm.joint_setpoint,
                            ik_res.position,
                            arm.max_velocities,
                            now,
                        )
                    arm.commands_applied += 1
                    continue

                # Jaw commands
                if command.channel in {"jaw/servo_jp", "jaw/move_jp"}:
                    target = float(command.payload)
                    if not np.isfinite(target) or not (arm.jaw_lower <= target <= arm.jaw_upper):
                        arm.commands_rejected += 1
                        if command.channel == "jaw/move_jp":
                            arm.move_failure_pending = True
                        continue
                    if command.channel == "jaw/servo_jp":
                        if arm.jaw_trajectory is not None:
                            arm.commands_canceled += 1
                        arm.jaw_trajectory = None
                        arm.jaw_setpoint = target
                        arm.jaw_velocity = 0.0
                    else:
                        if arm.jaw_trajectory is not None:
                            arm.commands_canceled += 1
                        arm.jaw_trajectory = JointTrajectory(
                            np.array([arm.jaw_setpoint]),
                            np.array([target]),
                            [arm.jaw_speed],
                            now,
                        )
                    arm.commands_applied += 1
                    continue

                arm.commands_rejected += 1

            # Advance trajectories
            if arm.joint_trajectory is not None:
                sample = arm.joint_trajectory.sample(now)
                arm.joint_setpoint = sample.position.copy()
                arm.joint_velocity = sample.velocity.copy()
                if sample.complete:
                    arm.joint_trajectory = None
                    arm.joint_velocity.fill(0.0)

            if arm.jaw_trajectory is not None:
                jaw_sample = arm.jaw_trajectory.sample(now)
                arm.jaw_setpoint = float(jaw_sample.position[0])
                arm.jaw_velocity = float(jaw_sample.velocity[0])
                if jaw_sample.complete:
                    arm.jaw_trajectory = None
                    arm.jaw_velocity = 0.0

            # Update q buffer
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
            arm.joint_setpoint = np.array(arm.config.home_position, dtype=float, copy=True)
            arm.joint_velocity.fill(0.0)
            arm.jaw_setpoint = 0.0
            arm.jaw_velocity = 0.0
            arm.cancel_motion()
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
        """Perform one complete simulation update cycle."""
        if self._reset_requested:
            self._reset_scene()
            self._reset_requested = False
        now_ns = time.monotonic_ns()
        self.prepare_step(now_ns, now_ns * 1e-9)
        return self.finish_step()

    def finish_step(self) -> dict[str, ArmSnapshot]:
        """Apply joint configuration to state, compute FK, step dynamics & grasp, and create snapshots."""
        # 1. Update robot kinematics and velocities. Newton derives body_qd from
        # joint_qd during FK; leaving joint_qd stale makes grasped objects appear
        # stationary even while the commanded arm is moving.
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

        # 2. Collision detection and grasp management
        snapshots = self.snapshots()

        if self.collision is not None:
            self.collision.collide(self.state, self.contacts)

        body_q_np = self.state.body_q.numpy()
        body_qd_np = self.state.body_qd.numpy()

        if self.grasp_manager is not None:
            self.grasp_manager.step(snapshots, self.contacts, body_q_np, body_qd_np)
            self.state.body_q.assign(self.wp.array(body_q_np, dtype=self.wp.transform, device=self.device))
            self.state.body_qd.assign(self.wp.array(body_qd_np, dtype=self.wp.spatial_vector, device=self.device))

        # 3. Step physics solver for dynamic objects
        if self.solver is not None and self._dynamic_body_indices:
            dt = 1.0 / self.options.simulation_rate_hz
            self.solver.step(self.state, self._state_out, self.control, self.contacts, dt)
            out_q = self._state_out.body_q.numpy()
            out_qd = self._state_out.body_qd.numpy()

            grasped_objects = {
                att.object_body_index for att in (
                    self.grasp_manager.attachments.values() if self.grasp_manager else ()
                )
            }
            for b_idx in self._dynamic_body_indices:
                if b_idx not in grasped_objects:
                    body_q_np[b_idx] = out_q[b_idx]
                    body_qd_np[b_idx] = out_qd[b_idx]

            if self.grasp_manager is not None and self.grasp_manager.attachments:
                self.grasp_manager.step(snapshots, None, body_q_np, body_qd_np)

            for b_idx, q_start in self._dynamic_body_joint_q.items():
                self._q_buffer[q_start : q_start + 7] = body_q_np[b_idx]

            self.state.body_q.assign(self.wp.array(body_q_np, dtype=self.wp.transform, device=self.device))
            self.state.body_qd.assign(self.wp.array(body_qd_np, dtype=self.wp.spatial_vector, device=self.device))

        self._simulation_time += 1.0 / self.options.simulation_rate_hz
        self._sequence += 1

        if self.viewer is not None and self.viewer.is_running():
            self.viewer.begin_frame(self._simulation_time)
            self.viewer.log_state(self.state)
            self.viewer.end_frame()

        snapshots = self.snapshots()
        if (
            self.camera_sink is not None
            and self.camera_renderer is not None
            and "ECM" in snapshots
            and snapshots["ECM"].measured_cp_world is not None
        ):
            if (self._simulation_time - self._last_camera_render_time) >= (self._camera_interval - 1e-6):
                frame = self.camera_renderer.render(
                    self.state,
                    snapshots["ECM"].measured_cp_world,
                    self._simulation_time,
                )
                self.camera_sink.push(frame)
                self._last_camera_render_time = self._simulation_time

        for arm in self.arms.values():
            arm.operating_state_event_pending = False
        return snapshots

    def snapshots(self) -> dict[str, ArmSnapshot]:
        """Extract ArmSnapshot for each loaded robot arm."""
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
        """Compute numerical damped least-squares IK using Newton forward kinematics."""
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

    def run(self, publish_snapshots, should_continue=None) -> None:
        """Paced simulation stepping loop."""
        period = 1.0 / self.options.simulation_rate_hz
        deadline = time.monotonic()
        while self.is_connected() and (should_continue is None or should_continue()):
            publish_snapshots(self.step())
            deadline += period
            remaining = deadline - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)
            else:
                deadline = time.monotonic()

    def shutdown(self) -> None:
        """Cleanup runtime resources."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.camera_sink is not None:
            self.camera_sink.close()
            self.camera_sink = None
        self.camera_renderer = None
        self._is_initialized = False
        self.model = None
        self.state = None
        self.builder = None
