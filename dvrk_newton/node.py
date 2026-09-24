"""ROS 2 CRTK node for the NVIDIA Newton physics backend."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import threading
import time
from typing import Sequence

# Check if Newton is available in current interpreter; re-exec into venv if not
try:
    import newton  # noqa: F401
    import warp  # noqa: F401
except ImportError:
    from dvrk_newton.python_runtime import resolve_newton_python
    _sel = resolve_newton_python()
    if Path(sys.executable).absolute() != _sel.path.absolute():
        os.execv(str(_sel.path), [str(_sel.path), "-m", "dvrk_newton.node"] + sys.argv[1:])

from ament_index_python.packages import get_package_share_directory
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.utilities import remove_ros_args

from dvrk_simulator_base.ros_interface import ArmRosInterface
from dvrk_simulator_base.snapshots import ArmSnapshot

from .configuration import (
    NewtonSimulatorConfig,
    load_installed_robot_config,
    load_installed_scene_config,
    load_simulator_config,
    resolve_scene_path,
)
from .camera import CameraOptions
from .errors import NewtonDependencyError
from .runtime import NewtonRuntime, NewtonRuntimeOptions
from .urdf_materializer import SUPPORTED_ROBOTS


class DvrkNewtonNode(Node):
    """ROS 2 node exposing CRTK interfaces for dVRK robots simulated in Newton."""

    def __init__(
        self,
        *,
        scene_path: Path | Sequence[Path] | None = None,
        model: str = "PSM1",
        instrument: str = "420006",
        endoscope: str = "Si_straight",
        device: str = "cuda:0",
        headless: bool = False,
        simulation_rate_hz: float = 120.0,
        state_publish_rate_hz: float = 100.0,
        generated_root: Path | None = None,
        command_queue_capacity: int = 32,
    ) -> None:
        super().__init__("dvrk_newton")
        self.headless = bool(headless)

        if scene_path is not None:
            scene = load_installed_scene_config(scene_path)
            configs = scene.robots
            self.scene = scene
            self.camera_options = CameraOptions.from_scene(scene.camera)
        else:
            model = model.upper()
            if model not in SUPPORTED_ROBOTS:
                raise ValueError(f"model must be one of {SUPPORTED_ROBOTS}")
            asset = endoscope if model == "ECM" else instrument
            configs = (load_installed_robot_config(model, asset),)
            self.scene = None
            self.camera_options = None

        self.configs = tuple(configs)
        self.device = str(device)
        self.simulation_rate_hz = float(simulation_rate_hz)
        state_rate = float(state_publish_rate_hz)
        if self.simulation_rate_hz <= 0.0 or state_rate <= 0.0:
            raise ValueError("simulation and state publish rates must be positive")
        self.generated_root = generated_root
        capacity = int(command_queue_capacity)
        if capacity <= 0:
            raise ValueError("command queue capacity must be positive")

        # Set up CRTK ROS interfaces for each arm
        ecm_config = next((item for item in self.configs if item.type == "ECM"), None)
        self.arm_interfaces: dict[str, ArmRosInterface] = {}
        if ecm_config is not None:
            ecm = ArmRosInterface(self, ecm_config, capacity)
            self.arm_interfaces[ecm_config.name] = ecm
        else:
            ecm = None

        for config in self.configs:
            if config.type != "ECM":
                self.arm_interfaces[config.name] = ArmRosInterface(
                    self, config, capacity, ecm_interface=ecm
                )

        self._publishing_enabled = True
        self._state_publish_rate_hz = state_rate
        self._state_publish_timer = self.create_timer(1.0 / state_rate, self._publish_latest)
        self._diagnostics = self.create_publisher(DiagnosticArray, "/diagnostics", 10)
        self._diagnostic_started_at = time.monotonic()
        self._diagnostic_snapshot_count = 0
        self._diagnostics_timer = self.create_timer(1.0, self._publish_diagnostics)

        # Single-arm compatibility if only one robot
        self._install_single_arm_compatibility(self.arm_interfaces[self.configs[0].name])

    def _install_single_arm_compatibility(self, interface: ArmRosInterface) -> None:
        self.config = interface.config
        self.commands = interface.commands
        self.snapshots = interface.snapshots
        self.frame_id = interface.frame_id
        for name in (
            "measured_js", "setpoint_js", "measured_cp", "setpoint_cp", "measured_cv",
            "jaw_measured_js", "jaw_setpoint_js", "operating_state", "state", "tool_type",
            "info", "warning", "error", "servo_jp", "move_jp", "servo_cp", "move_cp",
            "jaw_servo_jp", "jaw_move_jp", "state_command",
        ):
            setattr(self, name, getattr(interface, name))
        self._primary_interface = interface

    def install_initial_snapshots(self, snapshots: dict[str, ArmSnapshot]) -> None:
        for name, snapshot in snapshots.items():
            self.arm_interfaces[name].install_initial_snapshot(snapshot)

    def accept_snapshots(self, snapshots: dict[str, ArmSnapshot]) -> None:
        for name, snapshot in snapshots.items():
            self.arm_interfaces[name].snapshots.set(snapshot)
        self._diagnostic_snapshot_count += 1

    def _publish_diagnostics(self) -> None:
        if not self._publishing_enabled or not rclpy.ok():
            return
        now = time.monotonic()
        elapsed = max(now - self._diagnostic_started_at, 1e-6)
        simulation_hz = self._diagnostic_snapshot_count / elapsed
        self._diagnostic_started_at = now
        self._diagnostic_snapshot_count = 0

        status = DiagnosticStatus()
        status.name = "dvrk_newton/runtime"
        status.hardware_id = f"newton_{self.device}"
        status.level = DiagnosticStatus.OK if simulation_hz > 0.0 else DiagnosticStatus.WARN
        status.message = "running" if simulation_hz > 0.0 else "waiting for simulation"
        status.values = [
            KeyValue(key="simulation_hz", value=f"{simulation_hz:.1f}"),
            KeyValue(key="state_publish_hz", value=f"{self._state_publish_rate_hz:.1f}"),
            KeyValue(key="device", value=str(self.device)),
            KeyValue(key="arms", value=str(len(self.arm_interfaces))),
        ]
        msg = DiagnosticArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.status = [status]
        self._diagnostics.publish(msg)

    def _publish_latest(self) -> None:
        if not self._publishing_enabled or not rclpy.ok():
            return
        try:
            for interface in self.arm_interfaces.values():
                interface.publish_latest()
        except Exception:
            if not rclpy.ok():
                return
            raise

    def stop_publishing(self) -> None:
        self._publishing_enabled = False
        self._state_publish_timer.cancel()
        self._diagnostics_timer.cancel()


def _spin_executor(executor: SingleThreadedExecutor) -> None:
    try:
        executor.spin()
    except ExternalShutdownException:
        pass


def _parse_command_line(args: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, help="simulator settings YAML (default: installed newton.yaml)"
    )
    parser.add_argument(
        "--scene", type=Path, action="append", metavar="FILE",
        help="scene YAML path or installed scene filename; may be repeated internally",
    )
    parser.add_argument(
        "--model", default="PSM1", choices=SUPPORTED_ROBOTS,
        help="robot model when no scene is specified (default: PSM1)",
    )
    parser.add_argument(
        "--instrument", default="420006",
        help="instrument type for PSM (default: 420006)",
    )
    parser.add_argument(
        "--endoscope", default="Si_straight",
        help="endoscope type for ECM (default: Si_straight)",
    )
    parser.add_argument(
        "--device", default=None,
        help="Warp computation device, e.g. cuda:0 or cpu (default: from config)",
    )
    parser.add_argument(
        "--headless", choices=("true", "false"),
        help="override headless setting from config (true/false, default: from config)",
    )
    return parser.parse_args(remove_ros_args(args))


def main(args=None) -> int:
    raw_args = list(sys.argv[1:] if args is None else args)
    options = _parse_command_line(raw_args)

    config_path = options.config
    if config_path is None:
        config_path = (
            Path(get_package_share_directory("dvrk_newton")) / "share" / "newton.yaml"
        )

    try:
        config = load_simulator_config(config_path)
        scene_path = None
        if options.scene is not None:
            scene_path = resolve_scene_path(
                config_path,
                options.scene[0] if len(options.scene) == 1 else options.scene,
            )
        elif config.scene is not None:
            scene_path = resolve_scene_path(config_path, config.scene)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    device = options.device or config.device
    if "DVRK_SIMULATOR_TEST_TIMEOUT" in os.environ:
        headless = True
    elif options.headless is not None:
        headless = options.headless.lower() == "true"
    else:
        headless = config.headless

    rclpy.init(args=raw_args)
    node = None
    runtime = None
    executor = None
    executor_thread = None

    try:
        node = DvrkNewtonNode(
            scene_path=scene_path,
            model=options.model,
            instrument=options.instrument,
            endoscope=options.endoscope,
            device=device,
            headless=headless,
            simulation_rate_hz=config.simulation_rate_hz,
            state_publish_rate_hz=config.state_publish_rate_hz,
            generated_root=(
                None if config.generated_root is None else str(config.generated_root)
            ),
            command_queue_capacity=config.command_queue_capacity,
        )

        runtime = NewtonRuntime(
            node.configs,
            NewtonRuntimeOptions(
                device=device,
                headless=node.headless,
                simulation_rate_hz=node.simulation_rate_hz,
                generated_root=node.generated_root,
                camera_options=node.camera_options,
            ),
            {name: interface.commands for name, interface in node.arm_interfaces.items()},
        )

        node.install_initial_snapshots(runtime.initialize())
        node.get_logger().info(
            f"loaded NVIDIA Newton simulation on device '{device}': "
            + ", ".join(node.arm_interfaces)
        )

        executor = SingleThreadedExecutor()
        executor.add_node(node)
        executor_thread = threading.Thread(
            target=_spin_executor, args=(executor,), daemon=True
        )
        executor_thread.start()

        timeout = os.environ.get("DVRK_SIMULATOR_TEST_TIMEOUT")
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        runtime.run(
            node.accept_snapshots,
            lambda: rclpy.ok() and (deadline is None or time.monotonic() < deadline),
        )
    except KeyboardInterrupt:
        pass
    except NewtonDependencyError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    finally:
        if node is not None:
            node.stop_publishing()
        if runtime is not None:
            runtime.shutdown()
        if executor is not None:
            executor.shutdown()
        if executor_thread is not None:
            executor_thread.join(timeout=2.0)
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
