"""Contact-qualified grasping for kinematic PSM robots in NVIDIA Newton."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
import warp as wp

from .configuration import GraspConfig
from .scene_objects import LoadedSceneObject


@dataclass(frozen=True)
class NewtonGraspAttachment:
    arm_name: str
    object_name: str
    tool_body_index: int
    object_body_index: int
    relative_position: tuple[float, float, float]
    relative_orientation_xyzw: tuple[float, float, float, float]


class NewtonGraspManager:
    """Manages contact detection and rigid attachment of scene objects to PSM jaws."""

    def __init__(
        self,
        model: Any,
        arms: Mapping[str, Any],
        scene_objects: Mapping[str, LoadedSceneObject],
        grasp_config: GraspConfig | None = None,
    ) -> None:
        self.model = model
        self.arms = arms
        self.scene_objects = scene_objects
        self.dynamic_objects = {
            name: obj for name, obj in scene_objects.items() if obj.is_dynamic
        }
        self.config = grasp_config or GraspConfig()

        self.close_threshold = self.config.close_threshold_rad
        self.release_threshold = self.config.release_threshold_rad
        self.break_distance = self.config.break_distance_m
        self.break_orientation = self.config.break_orientation_rad
        self.max_grasps_per_object = self.config.max_grasps_per_object
        self.show_markers = self.config.show_grasps

        self.attachments: dict[str, NewtonGraspAttachment] = {}
        self.regrasp_blocked_arms: set[str] = set()
        self._last_tool_positions: dict[str, np.ndarray] = {}

        self.shape_body = model.shape_body.numpy()

        # Map each PSM to its jaw link body indices and shapes
        self.arm_jaw_bodies: dict[str, tuple[int, int]] = {}
        self.arm_jaw_shapes: dict[str, tuple[set[int], set[int]]] = {}
        self.arm_tool_body: dict[str, int] = {}

        for arm_name, arm_state in arms.items():
            robot = arm_state.robot
            if robot is None:
                continue
            jaw_bodies = self._find_jaw_bodies(arm_name, robot)
            if jaw_bodies is not None:
                j1_body, j2_body = jaw_bodies
                self.arm_jaw_bodies[arm_name] = jaw_bodies
                j1_shapes = set(int(idx) for idx in np.where(self.shape_body == j1_body)[0])
                j2_shapes = set(int(idx) for idx in np.where(self.shape_body == j2_body)[0])
                self.arm_jaw_shapes[arm_name] = (j1_shapes, j2_shapes)
                self.arm_tool_body[arm_name] = robot.tool_link_index

    @staticmethod
    def _find_jaw_bodies(arm_name: str, robot: Any) -> tuple[int, int] | None:
        prefix = f"{arm_name}_"
        links = robot.link_indices
        for p in (prefix, ""):
            j1 = f"{p}jaw_1_link"
            j2 = f"{p}jaw_2_link"
            if j1 in links and j2 in links:
                return links[j1], links[j2]
        return None

    def release_all(self) -> None:
        """Release all active grasp attachments."""
        self.attachments.clear()
        self.regrasp_blocked_arms.clear()
        self._last_tool_positions.clear()

    def _release(self, arm_name: str, require_reopen: bool = False) -> None:
        if arm_name in self.attachments:
            del self.attachments[arm_name]
            if require_reopen:
                self.regrasp_blocked_arms.add(arm_name)

    def _pose_break_exceeded(self, attachment: NewtonGraspAttachment, body_q_np: np.ndarray) -> bool:
        """Return whether an attachment has drifted beyond configured pose limits."""
        tool_q = body_q_np[attachment.tool_body_index]
        object_q = body_q_np[attachment.object_body_index]
        tool_xform = wp.transform(tuple(tool_q[:3]), tuple(tool_q[3:7]))
        object_xform = wp.transform(tuple(object_q[:3]), tuple(object_q[3:7]))
        expected_rel = wp.transform(attachment.relative_position, attachment.relative_orientation_xyzw)
        expected_object = wp.transform_multiply(tool_xform, expected_rel)
        expected_pos = np.asarray(wp.transform_get_translation(expected_object), dtype=float)
        if np.linalg.norm(np.asarray(object_q[:3], dtype=float) - expected_pos) > self.break_distance:
            return True
        actual_rel = wp.transform_multiply(wp.transform_inverse(tool_xform), object_xform)
        actual_rot = np.asarray(wp.transform_get_rotation(actual_rel), dtype=float)
        expected_rot = np.asarray(attachment.relative_orientation_xyzw, dtype=float)
        actual_rot /= max(np.linalg.norm(actual_rot), 1e-12)
        expected_rot /= max(np.linalg.norm(expected_rot), 1e-12)
        dot = float(abs(np.dot(actual_rot, expected_rot)))
        return 2.0 * math.acos(float(np.clip(dot, -1.0, 1.0))) > self.break_orientation

    def step(
        self,
        snapshots: Mapping[str, Any],
        contacts: Any,
        body_q_np: np.ndarray,
        body_qd_np: np.ndarray,
    ) -> None:
        """Process release conditions, detect new two-jaw contacts, and enforce attachments."""
        # 1. Unblock regrasp if jaws opened
        for arm_name, snapshot in snapshots.items():
            if (
                snapshot.jaw_measured is not None
                and snapshot.jaw_measured >= self.release_threshold
            ):
                self.regrasp_blocked_arms.discard(arm_name)

        # 2. Check release on open jaws
        for arm_name in tuple(self.attachments.keys()):
            jaw = snapshots[arm_name].jaw_measured if arm_name in snapshots else None
            tool_position = np.asarray(body_q_np[self.attachments[arm_name].tool_body_index, :3], dtype=float)
            previous_tool_position = self._last_tool_positions.get(arm_name)
            tool_moved = (
                previous_tool_position is not None
                and np.linalg.norm(tool_position - previous_tool_position) > self.break_distance
            )
            pose_break = (
                not tool_moved
                and self._pose_break_exceeded(self.attachments[arm_name], body_q_np)
            )
            if jaw is None or jaw >= self.release_threshold or pose_break:
                self._release(arm_name, require_reopen=False)
                if pose_break and jaw is not None and jaw < self.release_threshold:
                    self.regrasp_blocked_arms.add(arm_name)

        # 3. Detect new two-jaw contacts on dynamic objects
        if self.dynamic_objects and contacts is not None:
            contact_count = int(contacts.rigid_contact_count.numpy()[0])
            if contact_count > 0:
                s0_all = contacts.rigid_contact_shape0.numpy()[:contact_count]
                s1_all = contacts.rigid_contact_shape1.numpy()[:contact_count]
                p0_all = contacts.rigid_contact_point0.numpy()[:contact_count]
                p1_all = contacts.rigid_contact_point1.numpy()[:contact_count]

                grasp_counts: dict[str, int] = {}
                for att in self.attachments.values():
                    grasp_counts[att.object_name] = grasp_counts.get(att.object_name, 0) + 1

                for arm_name, (j1_shapes, j2_shapes) in self.arm_jaw_shapes.items():
                    if arm_name in self.attachments or arm_name in self.regrasp_blocked_arms:
                        continue
                    jaw_pos = (
                        snapshots[arm_name].jaw_measured
                        if arm_name in snapshots
                        else None
                    )
                    if jaw_pos is None or jaw_pos > self.close_threshold:
                        continue

                    tool_idx = self.arm_tool_body[arm_name]
                    tool_xform = wp.transform(
                        tuple(body_q_np[tool_idx, :3]),
                        tuple(body_q_np[tool_idx, 3:7]),
                    )
                    offset_xform = wp.transform(
                        self.config.contact_region_offset_m, (0.0, 0.0, 0.0, 1.0)
                    )
                    grasp_center = np.array(
                        wp.transform_get_translation(wp.transform_multiply(tool_xform, offset_xform))
                    )
                    max_jaw_dist = self.config.contact_region_radius_m
                    
                    # Find which dynamic objects are in contact with jaw1 and jaw2
                    for obj_name, obj in self.dynamic_objects.items():
                        if (
                            self.max_grasps_per_object > 0
                            and grasp_counts.get(obj_name, 0) >= self.max_grasps_per_object
                        ):
                            continue

                        obj_shapes = set(obj.shape_indices)
                        j1_touch = False
                        j2_touch = False

                        for c in range(contact_count):
                            s0 = int(s0_all[c])
                            s1 = int(s1_all[c])
                            in_j1 = (s0 in j1_shapes and s1 in obj_shapes) or (
                                s1 in j1_shapes and s0 in obj_shapes
                            )
                            in_j2 = (s0 in j2_shapes and s1 in obj_shapes) or (
                                s1 in j2_shapes and s0 in obj_shapes
                            )
                            if not (in_j1 or in_j2):
                                continue

                            # Check contact point on jaw in world frame
                            jaw_shape = s0 if (s0 in j1_shapes or s0 in j2_shapes) else s1
                            jaw_body = int(self.shape_body[jaw_shape])
                            jaw_local_p = p0_all[c] if jaw_shape == s0 else p1_all[c]

                            tf_jaw = wp.transform(
                                tuple(body_q_np[jaw_body, :3]),
                                tuple(body_q_np[jaw_body, 3:7]),
                            )
                            world_jaw_p = np.array(
                                wp.transform_point(
                                    tf_jaw,
                                    wp.vec3(
                                        float(jaw_local_p[0]),
                                        float(jaw_local_p[1]),
                                        float(jaw_local_p[2]),
                                    ),
                                )
                            )

                            if np.linalg.norm(world_jaw_p - grasp_center) <= max_jaw_dist:
                                if in_j1:
                                    j1_touch = True
                                if in_j2:
                                    j2_touch = True

                            if j1_touch and j2_touch:
                                break

                        if j1_touch and j2_touch:
                            # Form attachment!
                            obj_idx = obj.body_index
                            obj_xform = wp.transform(
                                tuple(body_q_np[obj_idx, :3]),
                                tuple(body_q_np[obj_idx, 3:7]),
                            )
                            rel_xform = wp.transform_multiply(
                                wp.transform_inverse(tool_xform), obj_xform
                            )
                            rel_pos = tuple(
                                float(v) for v in wp.transform_get_translation(rel_xform)
                            )
                            rel_rot = tuple(
                                float(v) for v in wp.transform_get_rotation(rel_xform)
                            )

                            self.attachments[arm_name] = NewtonGraspAttachment(
                                arm_name=arm_name,
                                object_name=obj_name,
                                tool_body_index=tool_idx,
                                object_body_index=obj_idx,
                                relative_position=rel_pos,
                                relative_orientation_xyzw=rel_rot,
                            )
                            grasp_counts[obj_name] = grasp_counts.get(obj_name, 0) + 1
                            self._last_tool_positions[arm_name] = np.asarray(
                                body_q_np[tool_idx, :3], dtype=float
                            ).copy()
                            break

        # 4. Enforce transforms for all active attachments
        for attachment in self.attachments.values():
            tool_idx = attachment.tool_body_index
            obj_idx = attachment.object_body_index
            tool_xform = wp.transform(
                tuple(body_q_np[tool_idx, :3]), tuple(body_q_np[tool_idx, 3:7])
            )
            rel_xform = wp.transform(
                attachment.relative_position, attachment.relative_orientation_xyzw
            )
            new_obj_xform = wp.transform_multiply(tool_xform, rel_xform)
            new_p = wp.transform_get_translation(new_obj_xform)
            new_q = wp.transform_get_rotation(new_obj_xform)

            body_q_np[obj_idx, :3] = [new_p[0], new_p[1], new_p[2]]
            body_q_np[obj_idx, 3:7] = [new_q[0], new_q[1], new_q[2], new_q[3]]
            # Newton spatial velocities are linear then angular. Account for the
            # rigid offset: v_obj = v_tool + omega_tool x r_world.
            tool_qd = np.asarray(body_qd_np[tool_idx], dtype=float)
            r_world = np.asarray(new_p, dtype=float) - np.asarray(body_q_np[tool_idx, :3], dtype=float)
            object_qd = np.asarray(body_qd_np[obj_idx], dtype=float).copy()
            object_qd[:3] = tool_qd[:3] + np.cross(tool_qd[3:6], r_world)
            object_qd[3:6] = tool_qd[3:6]
            body_qd_np[obj_idx] = object_qd
            self._last_tool_positions[attachment.arm_name] = np.asarray(
                body_q_np[tool_idx, :3], dtype=float
            ).copy()

    def marker_poses(self, body_q_np: np.ndarray) -> dict[str, tuple]:
        """Return visual marker poses for active grasps."""
        if not self.show_markers:
            return {}
        markers = {}
        for arm_name, attachment in self.attachments.items():
            obj_idx = attachment.object_body_index
            pos = tuple(float(v) for v in body_q_np[obj_idx, :3])
            rot = tuple(float(v) for v in body_q_np[obj_idx, 3:7])
            markers[arm_name] = (pos, rot)
        return markers

