"""Initialize a Newton scene inside the shared IPC simulation worker."""

from dvrk_simulator_base.command_mailbox import CommandMailboxes
from dvrk_simulator_base.process_worker import worker_main


def create_runtime(start):
    from .camera import CameraOptions
    from .configuration import load_installed_scene_config, load_simulator_config
    from .runtime import NewtonRuntime, NewtonRuntimeOptions

    config = load_simulator_config(start["config"])
    scene = load_installed_scene_config(start["scene"])
    commands = {item.name: CommandMailboxes(config.command_queue_capacity) for item in scene.robots}
    runtime = NewtonRuntime(scene.robots, NewtonRuntimeOptions(
        device=start["device"], headless=start["headless"],
        simulation_rate_hz=config.simulation_rate_hz, generated_root=config.generated_root,
        camera_options=CameraOptions.from_scene(scene.camera), scene_objects=tuple(scene.objects),
        grasp_config=config.grasp, rigid_gap_m=config.rigid_gap_m,
    ), commands)
    return runtime, commands


if __name__ == "__main__":
    raise SystemExit(worker_main(create_runtime))
