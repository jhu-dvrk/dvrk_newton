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
class NewtonSimulatorConfig:
    device: str = "cuda:0"
    headless: bool = False
    simulation_rate_hz: float = 120.0
    state_publish_rate_hz: float = 100.0
    generated_root: Path | None = None
    command_queue_capacity: int = 32
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
    return NewtonSimulatorConfig(
        device=device,
        headless=_boolean(headless_val, source=source, field="headless"),
        simulation_rate_hz=simulation_rate,
        state_publish_rate_hz=state_rate,
        generated_root=generated_root,
        command_queue_capacity=capacity,
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
        pybullet_share / "share" / "scenes",
        share / "share" / "exercises",
    )
    resolver = SceneResolver(tuple(search_paths or default_search))
    return load_scene_config(
        path,
        robot_config_root=arm_description_share / "arms",
        resolver=resolver,
    )
