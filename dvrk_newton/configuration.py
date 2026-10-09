"""Runtime options for the newton backend."""

from dataclasses import dataclass
from dvrk_simulator_base.configuration import RuntimeConfig, load_runtime_config
from dvrk_simulator_base import configuration as _shared
from dvrk_simulator_base.configuration import PoseGraspConfig as GraspConfig


@dataclass(frozen=True)
class NewtonSimulatorConfig(RuntimeConfig):
    device: str = "cuda:0"
    rigid_gap_m: float = 0.005
    grasp: GraspConfig | None = None


def load_simulator_config(path):
    return load_runtime_config(path, NewtonSimulatorConfig, grasp_type=GraspConfig)


def scene_search_paths(config_path):
    return _shared.scene_search_paths("dvrk_newton", config_path)


def resolve_scene_path(config_path, selection):
    return _shared.resolve_scene_path("dvrk_newton", config_path, selection)


def load_installed_scene_config(path, *, search_paths=None):
    return _shared.load_installed_scene_config("dvrk_newton", path, search_paths=search_paths)
