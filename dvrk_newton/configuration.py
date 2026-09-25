"""Resolve robot and simulator configuration for the NVIDIA Newton backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml

from ament_index_python.packages import get_package_share_directory

from dvrk_arm_description import RobotConfig, load_robot_config
from dvrk_simulator_base.scene import SceneConfig, SceneResolver, load_scene_config


@dataclass(frozen=True)
class GraspConfig:
    show_grasps: bool = True
    max_grasps_per_object: int = 1
    policy: str = "pose_error"
    arm_policies: dict[str, str] | None = None
    close_threshold_rad: float = 0.04
    release_threshold_rad: float = 0.08
    break_distance_m: float = 0.005
    break_orientation_rad: float = 0.2617993877991494
    break_tension_force_n: float = 10.0
    break_shear_force_n: float = 10.0
    break_torque_nm: float = 0.25
    break_load_duration_s: float = 0.05
    max_force_n: float = 100.0
    constraint_erp: float = 0.8
    contact_region_offset_m: tuple[float, float, float] = (0.0, 0.0, -0.003)
    contact_region_radius_m: float = 0.005


@dataclass(frozen=True)
class NewtonSimulatorConfig:
    device: str = "cuda:0"
    headless: bool = False
    simulation_rate_hz: float = 120.0
    state_publish_rate_hz: float = 100.0
    generated_root: Path | None = None
    command_queue_capacity: int = 32
    rigid_gap_m: float = 0.005
    grasp: GraspConfig | None = None
    scene: str | None = None


def _boolean(value, *, source: Path, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{source}: {field} must be true or false")
    return value


def load_simulator_config(path: str | Path) -> NewtonSimulatorConfig:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    if not isinstance(document, dict):
        raise ValueError(f"{source}: simulator configuration must be a mapping")

    device = str(document.get("device", "cuda:0")).strip()
    simulation_rate = float(document.get("simulation_rate_hz", 120.0))
    state_rate = float(document.get("state_publish_rate_hz", 100.0))
    if simulation_rate <= 0.0 or state_rate <= 0.0:
        raise ValueError(f"{source}: simulation and state publish rates must be positive")

    capacity = int(document.get("command_queue_capacity", 32))
    if capacity <= 0:
        raise ValueError(f"{source}: command_queue_capacity must be positive")

    generated = document.get("generated_root")
    generated_root = None
    if generated not in (None, ""):
        generated_root = Path(str(generated)).expanduser()
        if not generated_root.is_absolute():
            generated_root = (source.parent / generated_root).resolve()
    scene = document.get("scene")

    headless_val = document.get("headless", False)

    grasp_doc = document.get("grasp")
    if grasp_doc is None:
        grasp_doc = {}
    if not isinstance(grasp_doc, dict):
        raise ValueError(f"{source}: grasp must be a mapping")

    policy = str(grasp_doc.get("policy", "pose_error"))
    arms_doc = grasp_doc.get("arms", {})
    arm_policies = {}
    if isinstance(arms_doc, dict):
        for arm_name, arm_subdoc in arms_doc.items():
            if isinstance(arm_subdoc, dict):
                arm_policies[str(arm_name)] = str(arm_subdoc.get("policy", policy))

    offset = tuple(float(v) for v in grasp_doc.get("contact_region_offset_m", (0.0, 0.0, -0.003)))
    if len(offset) != 3:
        raise ValueError(f"{source}: grasp.contact_region_offset_m must contain three values")

    grasp_config = GraspConfig(
        show_grasps=_boolean(grasp_doc.get("show_grasps", True), source=source, field="grasp.show_grasps"),
        max_grasps_per_object=int(grasp_doc.get("max_grasps_per_object", 1)),
        policy=policy,
        arm_policies=arm_policies or None,
        close_threshold_rad=float(grasp_doc.get("close_threshold_rad", 0.04)),
        release_threshold_rad=float(grasp_doc.get("release_threshold_rad", 0.08)),
        break_distance_m=float(grasp_doc.get("break_distance_m", 0.005)),
        break_orientation_rad=float(grasp_doc.get("break_orientation_rad", 0.2617993877991494)),
        break_tension_force_n=float(grasp_doc.get("break_tension_force_n", 10.0)),
        break_shear_force_n=float(grasp_doc.get("break_shear_force_n", 10.0)),
        break_torque_nm=float(grasp_doc.get("break_torque_nm", 0.25)),
        break_load_duration_s=float(grasp_doc.get("break_load_duration_s", 0.05)),
        max_force_n=float(grasp_doc.get("max_force_n", 100.0)),
        constraint_erp=float(grasp_doc.get("constraint_erp", 0.8)),
        contact_region_offset_m=offset,
        contact_region_radius_m=float(grasp_doc.get("contact_region_radius_m", 0.005)),
    )

    if grasp_config.contact_region_radius_m <= 0.0:
        raise ValueError(f"{source}: grasp.contact_region_radius_m must be positive")

    rigid_gap_m = float(document.get("rigid_gap_m", 0.005))
    if rigid_gap_m <= 0.0:
        raise ValueError(f"{source}: rigid_gap_m must be positive")

    return NewtonSimulatorConfig(
        device=device,
        headless=_boolean(headless_val, source=source, field="headless"),
        simulation_rate_hz=simulation_rate,
        state_publish_rate_hz=state_rate,
        generated_root=generated_root,
        command_queue_capacity=capacity,
        rigid_gap_m=rigid_gap_m,
        grasp=grasp_config,
        scene=None if scene in (None, "") else str(scene),
    )


def load_installed_robot_config(model: str, instrument: str) -> RobotConfig:
    share = Path(get_package_share_directory("dvrk_arm_description"))
    path = share / "arms" / f"{model}.yaml"
    if not path.is_file():
        raise RuntimeError(f"installed robot configuration does not exist: {path}")
    if str(model).upper() == "ECM":
        return load_robot_config(path, endoscope=instrument)
    return load_robot_config(path, instrument=instrument)


def scene_search_paths(config_path: str | Path) -> tuple[Path, ...]:
    config = Path(config_path).expanduser().resolve()
    package_share = Path(get_package_share_directory("dvrk_newton"))
    pybullet_share = Path(get_package_share_directory("dvrk_pybullet"))
    simulator_base_share = Path(get_package_share_directory("dvrk_simulator_base"))
    candidates = (
        config.parent / "scenes",
        package_share / "share" / "scenes",
        simulator_base_share / "share" / "scenes",
        pybullet_share / "share" / "scenes",
        simulator_base_share / "share" / "exercises",
    )
    paths = []
    for path in candidates:
        path = path.resolve()
        if path.is_dir() and path not in paths:
            paths.append(path)
    return tuple(paths)


def resolve_scene_path(
    config_path: str | Path,
    selection: str | Path | Sequence[str | Path],
) -> Path | tuple[Path, ...]:
    config = Path(config_path).expanduser().resolve()
    resolver = SceneResolver(scene_search_paths(config), relative_root=config.parent)
    if isinstance(selection, (list, tuple)):
        return resolver.resolve_all(selection)
    return resolver.resolve(selection)


def load_installed_scene_config(
    path: str | Path | Sequence[str | Path],
    *,
    search_paths: Sequence[Path] | None = None,
) -> SceneConfig:
    share = Path(get_package_share_directory("dvrk_simulator_base"))
    arm_description_share = Path(get_package_share_directory("dvrk_arm_description"))
    newton_share = Path(get_package_share_directory("dvrk_newton"))
    pybullet_share = Path(get_package_share_directory("dvrk_pybullet"))
    default_search = (
        newton_share / "share" / "scenes",
        share / "share" / "scenes",
        pybullet_share / "share" / "scenes",
        share / "share" / "exercises",
    )
    resolver = SceneResolver(tuple(search_paths or default_search))
    return load_scene_config(
        path,
        robot_config_root=arm_description_share / "arms",
        resolver=resolver,
    )
