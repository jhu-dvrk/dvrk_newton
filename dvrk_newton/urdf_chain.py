"""CPU forward kinematics and analytic Jacobian for a materialized URDF chain."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from dvrk_simulator_base.rotations import quaternion_matrix_xyzw


def _vector(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=float)
    values = np.fromstring(text, sep=" ", dtype=float)
    if values.shape != (3,):
        raise ValueError(f"expected three URDF coordinates, got {text!r}")
    return values


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    v = 1.0 - c
    return np.array(
        [
            [c + x * x * v, x * y * v - z * s, x * z * v + y * s],
            [y * x * v + z * s, c + y * y * v, y * z * v - x * s],
            [z * x * v - y * s, z * y * v + x * s, c + z * z * v],
        ]
    )


@dataclass(frozen=True)
class _Joint:
    kind: str
    origin_position: np.ndarray
    origin_rotation: np.ndarray
    axis: np.ndarray
    control_index: int | None
    multiplier: float = 1.0
    offset: float = 0.0


class UrdfChain:
    """Evaluate one root-to-tool chain in the same world frame as Newton."""

    def __init__(
        self,
        urdf_path: str | Path,
        tool_link: str,
        joint_names: tuple[str, ...],
        base_position: tuple[float, float, float],
        base_orientation_xyzw: tuple[float, float, float, float],
    ) -> None:
        root = ET.parse(urdf_path).getroot()
        by_child: dict[str, ET.Element] = {}
        for joint in root.findall("joint"):
            child = joint.find("child")
            if child is not None:
                by_child[child.attrib["link"]] = joint

        chain: list[ET.Element] = []
        link = tool_link
        while link in by_child:
            joint = by_child[link]
            chain.append(joint)
            link = joint.find("parent").attrib["link"]
        if not chain:
            raise ValueError(f"tool link {tool_link!r} is absent from {urdf_path}")

        control_indices = {name: index for index, name in enumerate(joint_names)}
        found: set[str] = set()
        parsed: list[_Joint] = []
        for element in reversed(chain):
            name = element.attrib["name"]
            kind = element.attrib.get("type", "fixed")
            if kind not in {"fixed", "revolute", "continuous", "prismatic"}:
                raise ValueError(f"unsupported URDF joint type {kind!r} in {name}")
            origin = element.find("origin")
            origin_position = _vector(
                None if origin is None else origin.attrib.get("xyz"), (0.0, 0.0, 0.0)
            )
            origin_rotation = _rpy_matrix(
                _vector(None if origin is None else origin.attrib.get("rpy"), (0.0, 0.0, 0.0))
            )
            axis_element = element.find("axis")
            axis = _vector(
                None if axis_element is None else axis_element.attrib.get("xyz"),
                (1.0, 0.0, 0.0),
            )
            axis_norm = float(np.linalg.norm(axis))
            if kind != "fixed" and axis_norm == 0.0:
                raise ValueError(f"zero URDF joint axis in {name}")
            if axis_norm > 0.0:
                axis = axis / axis_norm

            control_index = None
            multiplier = 1.0
            offset = 0.0
            if kind != "fixed":
                mimic = element.find("mimic")
                if mimic is not None:
                    source = mimic.attrib["joint"]
                    multiplier = float(mimic.attrib.get("multiplier", "1.0"))
                    offset = float(mimic.attrib.get("offset", "0.0"))
                else:
                    source = name
                if source not in control_indices:
                    raise ValueError(f"uncontrolled URDF joint {name!r} in tool chain")
                control_index = control_indices[source]
                found.add(source)
            parsed.append(
                _Joint(kind, origin_position, origin_rotation, axis, control_index, multiplier, offset)
            )

        missing = set(joint_names) - found
        if missing:
            raise ValueError(f"controlled joints outside tool chain: {', '.join(sorted(missing))}")
        self.joints = tuple(parsed)
        self.joint_count = len(joint_names)
        self.base_position = np.asarray(base_position, dtype=float)
        self.base_rotation = quaternion_matrix_xyzw(base_orientation_xyzw)

    def forward(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return world position, orientation, and 6-by-N spatial Jacobian."""
        if q.shape != (self.joint_count,):
            raise ValueError("joint position has the wrong size")
        position = self.base_position.copy()
        rotation = self.base_rotation.copy()
        axes: list[tuple[int, str, np.ndarray, np.ndarray, float]] = []
        for joint in self.joints:
            position = position + rotation @ joint.origin_position
            rotation = rotation @ joint.origin_rotation
            if joint.control_index is None:
                continue
            axis_world = rotation @ joint.axis
            axes.append(
                (joint.control_index, joint.kind, axis_world, position.copy(), joint.multiplier)
            )
            amount = q[joint.control_index] * joint.multiplier + joint.offset
            if joint.kind == "prismatic":
                position = position + axis_world * amount
            else:
                rotation = rotation @ _axis_rotation(joint.axis, amount)

        jacobian = np.zeros((6, self.joint_count), dtype=float)
        for index, kind, axis, origin, multiplier in axes:
            if kind == "prismatic":
                jacobian[:3, index] += axis * multiplier
            else:
                jacobian[:3, index] += np.cross(axis, position - origin) * multiplier
                jacobian[3:, index] += axis * multiplier
        return position, rotation, jacobian
