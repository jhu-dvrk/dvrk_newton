"""Load a materialized dVRK model and build name-based NVIDIA Newton mappings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import xml.etree.ElementTree as ET

import numpy as np

from .errors import NewtonBackendError


@dataclass(frozen=True)
class MimicJoint:
    joint_name: str
    source_joint_name: str
    multiplier: float
    offset: float
    joint_index: int
    source_joint_index: int
    q_index: int
    source_q_index: int


@dataclass(frozen=True)
class LoadedRobot:
    name: str
    joint_indices: dict[str, int]
    link_indices: dict[str, int]
    controlled_joint_names: tuple[str, ...]
    controlled_joint_indices: tuple[int, ...]
    controlled_q_indices: tuple[int, ...]
    jaw_joint_name: str | None
    jaw_joint_index: int | None
    jaw_q_index: int | None
    mimic_joints: tuple[MimicJoint, ...]
    tool_link_index: int


def _normalize_name(name: str) -> str:
    """Strip namespaces and slashes for flexible matching."""
    return name.split("/")[-1].strip()


def add_robot_to_builder(
    builder: Any,
    urdf_path: str | Path,
    *,
    base_position: Iterable[float] = (0.0, 0.0, 0.0),
    base_orientation_xyzw: Iterable[float] = (0.0, 0.0, 0.0, 1.0),
    enable_self_collisions: bool = False,
) -> None:
    """Add a robot URDF into a Newton ModelBuilder."""
    import warp as wp

    path = Path(urdf_path).resolve()
    if not path.is_file():
        raise NewtonBackendError(f"materialized URDF does not exist: {path}")

    pos = tuple(float(v) for v in base_position)
    rot = tuple(float(v) for v in base_orientation_xyzw)
    xform = wp.transform(pos, rot)

    builder.add_urdf(
        str(path),
        xform=xform,
        floating=False,
        collapse_fixed_joints=False,
        enable_self_collisions=enable_self_collisions,
    )


def build_robot_mapping(
    model: Any,
    robot_name: str,
    urdf_path: str | Path,
    expected_joint_names: Iterable[str],
    tool_frame: str | None = None,
) -> LoadedRobot:
    """Extract name-based joint and body mappings from a finalized Newton Model."""
    path = Path(urdf_path).resolve()
    prefix = f"{robot_name}/"

    joint_indices: dict[str, int] = {}
    joint_q_indices: dict[str, int] = {}
    link_indices: dict[str, int] = {}

    joint_labels = list(model.joint_label)
    joint_types = model.joint_type.numpy()
    joint_q_starts = model.joint_q_start.numpy()

    for idx, label in enumerate(joint_labels):
        # Register full label
        joint_indices[label] = idx
        # Register unprefixed label if it belongs to this robot
        if label.startswith(prefix):
            unprefixed = label[len(prefix):]
            joint_indices[unprefixed] = idx
            joint_indices[f"{robot_name}_{unprefixed}"] = idx
        else:
            base = _normalize_name(label)
            if base not in joint_indices:
                joint_indices[base] = idx

        # Record DOF q-index if not a fixed joint (type 3)
        if joint_types[idx] != 3:
            q_idx = int(joint_q_starts[idx])
            joint_q_indices[label] = q_idx
            if label.startswith(prefix):
                unprefixed = label[len(prefix):]
                joint_q_indices[unprefixed] = q_idx
                joint_q_indices[f"{robot_name}_{unprefixed}"] = q_idx
            else:
                base = _normalize_name(label)
                if base not in joint_q_indices:
                    joint_q_indices[base] = q_idx

    body_labels = list(model.body_label)
    for idx, label in enumerate(body_labels):
        link_indices[label] = idx
        if label.startswith(prefix):
            unprefixed = label[len(prefix):]
            link_indices[unprefixed] = idx
            link_indices[f"{robot_name}_{unprefixed}"] = idx
            # If unprefixed has _link suffix, also index without it
            if unprefixed.endswith("_link"):
                link_indices[unprefixed[:-5]] = idx
                link_indices[f"{robot_name}_{unprefixed[:-5]}"] = idx
        else:
            base = _normalize_name(label)
            if base not in link_indices:
                link_indices[base] = idx
            if base.endswith("_link") and base[:-5] not in link_indices:
                link_indices[base[:-5]] = idx

    # Validate expected controlled joints
    expected = tuple(str(name) for name in expected_joint_names)
    missing = tuple(name for name in expected if name not in joint_indices)
    if missing:
        raise NewtonBackendError(
            f"materialized robot {robot_name} is missing configured joints: {', '.join(missing)}"
        )

    controlled_indices = tuple(joint_indices[name] for name in expected)
    controlled_q = tuple(joint_q_indices[name] for name in expected)

    # Check for jaw joint specific to this robot
    jaw_name: str | None = None
    jaw_joint_idx: int | None = None
    jaw_q_idx: int | None = None
    jaw_cand = f"{prefix}jaw"
    if jaw_cand in joint_indices and jaw_cand in joint_q_indices:
        jaw_name = "jaw"
        jaw_joint_idx = joint_indices[jaw_cand]
        jaw_q_idx = joint_q_indices[jaw_cand]
    elif f"{robot_name}_jaw" in joint_indices and f"{robot_name}_jaw" in joint_q_indices:
        jaw_name = "jaw"
        jaw_joint_idx = joint_indices[f"{robot_name}_jaw"]
        jaw_q_idx = joint_q_indices[f"{robot_name}_jaw"]

    # Parse mimic joints from URDF
    mimic_joints: list[MimicJoint] = []
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as error:
        raise NewtonBackendError(f"materialized URDF is invalid XML: {error}") from error

    for joint_element in root.findall("joint"):
        mimic = joint_element.find("mimic")
        if mimic is None:
            continue
        j_name = joint_element.attrib.get("name", "")
        src_name = mimic.attrib.get("joint", "")

        j_key = f"{prefix}{j_name}" if f"{prefix}{j_name}" in joint_indices else j_name
        src_key = f"{prefix}{src_name}" if f"{prefix}{src_name}" in joint_indices else src_name

        if j_key in joint_indices and src_key in joint_indices:
            j_idx = joint_indices[j_key]
            src_idx = joint_indices[src_key]
            if j_key in joint_q_indices and src_key in joint_q_indices:
                multiplier = float(mimic.attrib.get("multiplier", "1.0"))
                offset = float(mimic.attrib.get("offset", "0.0"))
                mimic_joints.append(
                    MimicJoint(
                        joint_name=j_name,
                        source_joint_name=src_name,
                        multiplier=multiplier,
                        offset=offset,
                        joint_index=j_idx,
                        source_joint_index=src_idx,
                        q_index=joint_q_indices[j_key],
                        source_q_index=joint_q_indices[src_key],
                    )
                )

    # Locate tool tip link
    candidates = (
        f"{robot_name}_tool_tip_link",
        f"{prefix}{robot_name}_tool_tip_link",
        f"{robot_name}_tool_tip",
        f"{prefix}tool_tip_link",
        tool_frame,
        f"{tool_frame}_link" if tool_frame else None,
        f"{robot_name}_tip_link",
        f"{prefix}{robot_name}_tip_link",
        f"{robot_name}_endoscope_frame_link",
        f"{prefix}{robot_name}_endoscope_frame_link",
        f"{robot_name}_endoscope_link",
    )
    tool_link_idx: int | None = None
    for cand in candidates:
        if cand and cand in link_indices:
            tool_link_idx = link_indices[cand]
            break

    if tool_link_idx is None:
        raise NewtonBackendError(
            f"could not resolve tool link for robot {robot_name}; tried: "
            + ", ".join(c for c in candidates if c)
        )

    return LoadedRobot(
        name=robot_name,
        joint_indices=joint_indices,
        link_indices=link_indices,
        controlled_joint_names=expected,
        controlled_joint_indices=controlled_indices,
        controlled_q_indices=controlled_q,
        jaw_joint_name=jaw_name,
        jaw_joint_index=jaw_joint_idx,
        jaw_q_index=jaw_q_idx,
        mimic_joints=tuple(mimic_joints),
        tool_link_index=tool_link_idx,
    )
