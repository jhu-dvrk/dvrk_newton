"""Start the NVIDIA Newton patient cart and the optional Quest/OpenXR console."""

from __future__ import annotations

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, LogInfo, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = Path(get_package_share_directory("dvrk_newton"))
    open_xr_directory = package_share / "share" / "open-xr"
    simulator_config = open_xr_directory / "newton.yaml"
    system_config = (
        open_xr_directory / "system-MTML-MTMR-OpenXR-patient-cart-ROS.json"
    )
    overlay_config = open_xr_directory / "dvrk-console-overlay.json"

    from dvrk_newton.configuration import load_simulator_config, resolve_scene_path
    from dvrk_newton.python_runtime import resolve_newton_python
    from dvrk_newton.urdf_materializer import default_generated_root
    from dvrk_simulator_base.rqt_perspective import (
        existing_ament_prefix_path,
        write_monitor_perspective,
    )
    newton_config = load_simulator_config(simulator_config)
    selection = resolve_newton_python(newton_config.generated_root)
    scene = resolve_scene_path(simulator_config, "ECM_PSM1_PSM2_PSM3.yaml")
    rqt_perspective = write_monitor_perspective(
        (newton_config.generated_root or default_generated_root()) / "rqt" / "open-xr.perspective",
        ("ECM", "PSM1", "PSM2", "PSM3"), include_console=True,
    )

    simulator = ExecuteProcess(
        cmd=[
            str(selection.path),
            str(package_share / "scripts" / "simulator.py"),
            "--config", str(simulator_config),
            "--scene", str(scene),
            "--scene", LaunchConfiguration("scene"),
            "--headless", LaunchConfiguration("headless"),
        ],
        output="screen",
    )
    dvrk_system = Node(
        package="dvrk_robot",
        executable="dvrk_system",
        name="dvrk_system",
        output="screen",
        cwd=str(open_xr_directory),
        arguments=["--json-config", str(system_config)],
    )
    console_overlay = Node(
        package="dvrk_console",
        executable="stereo_display",
        name="newton_console_overlay",
        output="screen",
        arguments=["-c", str(overlay_config)],
    )
    start_system = Node(
        package="dvrk_simulator_base",
        executable="start_dvrk_system",
        output="screen",
        arguments=["--console", LaunchConfiguration("console")],
    )
    rqt_environment = {
        "DVRK_RQT_ARMS": "ECM,PSM1,PSM2,PSM3",
        "DVRK_RQT_CONSOLE": LaunchConfiguration("console"),
    }
    if prefix_path := existing_ament_prefix_path():
        rqt_environment["AMENT_PREFIX_PATH"] = prefix_path
    rqt_monitor = ExecuteProcess(
        cmd=["rqt", "--perspective-file", str(rqt_perspective)],
        additional_env=rqt_environment,
        condition=IfCondition(LaunchConfiguration("rqt")), output="screen",
    )

    stop_with_simulator = RegisterEventHandler(
        OnProcessExit(
            target_action=simulator,
            on_exit=[EmitEvent(event=Shutdown(reason="NVIDIA Newton simulator exited"))],
        )
    )
    stop_with_console = RegisterEventHandler(
        OnProcessExit(
            target_action=dvrk_system,
            on_exit=[EmitEvent(event=Shutdown(reason="dvrk_system exited"))],
        )
    )
    stop_with_overlay = RegisterEventHandler(
        OnProcessExit(
            target_action=console_overlay,
            on_exit=[EmitEvent(event=Shutdown(reason="console video overlay exited"))],
        )
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "scene",
                default_value="tray_cubes.yaml",
                description="Exercise scene YAML path or installed exercise filename",
            ),
            DeclareLaunchArgument(
                "console",
                default_value="console",
                description="dVRK console ROS namespace",
            ),
            DeclareLaunchArgument(
                "headless",
                default_value="true",
                description="run without desktop ViewerGL window (HMD provides the display)",
            ),
            DeclareLaunchArgument(
                "rqt", default_value="false",
                description="start a dockable dVRK and CRTK rqt monitor",
            ),
            LogInfo(
                msg=(
                    f"Starting NVIDIA Newton simulator with Python {selection.path} "
                    f"(selected via {selection.source})"
                )
            ),
            simulator,
            console_overlay,
            dvrk_system,
            start_system,
            rqt_monitor,
            stop_with_simulator,
            stop_with_console,
            stop_with_overlay,
        ]
    )
