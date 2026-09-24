"""Start configured dVRK robots in NVIDIA Newton using the configured Python interpreter."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from dvrk_newton.configuration import (
    load_installed_scene_config,
    load_simulator_config,
    resolve_scene_path,
)
from dvrk_newton.python_runtime import resolve_newton_python
from dvrk_newton.urdf_materializer import default_generated_root
from dvrk_simulator_base.rqt_perspective import (
    existing_ament_prefix_path,
    write_monitor_perspective,
)


def _start_sim(context):
    package_share = Path(get_package_share_directory("dvrk_newton"))
    config_path = Path(LaunchConfiguration("config").perform(context)).expanduser().resolve()
    simulator_config = load_simulator_config(config_path)

    selection = resolve_newton_python(simulator_config.generated_root)
    script = package_share / "scripts" / "simulator.py"

    scene = LaunchConfiguration("scene").perform(context) or simulator_config.scene
    model = LaunchConfiguration("model").perform(context)
    instrument = LaunchConfiguration("instrument").perform(context)
    device = LaunchConfiguration("device").perform(context) or simulator_config.device

    headless_arg = LaunchConfiguration("headless").perform(context)
    if headless_arg:
        headless = "true" if headless_arg.lower() == "true" else "false"
    else:
        headless = "true" if simulator_config.headless else "false"

    cmd = [
        str(selection.path),
        str(script),
        "--config", str(config_path),
        "--device", str(device),
        "--headless", str(headless),
    ]

    if scene:
        cmd.extend(["--scene", str(scene)])
    else:
        cmd.extend(["--model", str(model), "--instrument", str(instrument)])

    actions = [
        LogInfo(
            msg=(
                f"Starting NVIDIA Newton simulator on device '{device}' with Python {selection.path} "
                f"(selected via {selection.source})"
            )
        ),
        ExecuteProcess(cmd=cmd, output="screen"),
    ]

    if LaunchConfiguration("rqt").perform(context).lower() == "true":
        if scene:
            scene_config = resolve_scene_path(config_path, scene)
            scene_description = load_installed_scene_config(scene_config)
            arms = [robot.name for robot in scene_description.robots]
        else:
            arms = [model]

        perspective = write_monitor_perspective(
            (simulator_config.generated_root or default_generated_root()) / "rqt" / "monitor.perspective",
            arms,
            include_console=LaunchConfiguration("rqt_console").perform(context).lower() == "true",
        )
        rqt_environment = {"DVRK_RQT_ARMS": ",".join(arms)}
        if prefix_path := existing_ament_prefix_path():
            rqt_environment["AMENT_PREFIX_PATH"] = prefix_path
        actions.append(ExecuteProcess(
            cmd=["rqt", "--perspective-file", str(perspective)], output="screen",
            additional_env=rqt_environment,
        ))

    return actions


def generate_launch_description():
    package_share = Path(get_package_share_directory("dvrk_newton"))
    default_config = package_share / "share" / "newton.yaml"

    return LaunchDescription([
        DeclareLaunchArgument(
            "config", default_value=str(default_config),
            description="Backend runtime configuration YAML",
        ),
        DeclareLaunchArgument(
            "scene", default_value="",
            description="Scene YAML path or installed scene filename",
        ),
        DeclareLaunchArgument(
            "model", default_value="PSM1",
            description="Robot model (PSM1, PSM2, PSM3, ECM) when no scene is specified",
        ),
        DeclareLaunchArgument(
            "instrument", default_value="420006",
            description="Instrument type for PSM when no scene is specified",
        ),
        DeclareLaunchArgument(
            "device", default_value="",
            description="Warp compute device, e.g. cuda:0 or cpu (defaults to config)",
        ),
        DeclareLaunchArgument(
            "headless", default_value="false",
            description="Run without GUI window (true/false, default: false)",
        ),
        DeclareLaunchArgument(
            "rqt", default_value="false",
            description="Start a dockable dVRK rqt monitor",
        ),
        DeclareLaunchArgument(
            "rqt_console", default_value="false",
            description="Include the dVRK Console widget in the rqt monitor",
        ),
        OpaqueFunction(function=_start_sim),
    ])
