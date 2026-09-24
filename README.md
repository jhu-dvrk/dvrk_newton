# dvrk_newton

NVIDIA Newton physics and kinematics backend for the common [dvrk_simulator_base](file:///home/anton/wss/dvrk/src/dvrk/dvrk_simulator_base) package.

This package runs hardware-accelerated robot kinematics and physics simulation on NVIDIA GPUs using NVIDIA Newton (`newton`) and NVIDIA Warp (`warp-lang`).

---

## Architecture Overview

```
                      +-----------------------------+
                      |   ROS 2 / CRTK Ecosystem    |
                      |  Topics, Services, Clients  |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |       DvrkNewtonNode        |
                      |    (ArmRosInterface x N)    |
                      +--------------+--------------+
                                     | CommandMailboxes
                                     v
                      +-----------------------------+
                      |        NewtonRuntime        |
                      |     (120 Hz GPU Paced)      |
                      +--------------+--------------+
                                     |
        +----------------------------+----------------------------+
        |                                                         |
        v                                                         v
+-------------------------------+                         +-------------------------------+
|     Warp / Newton Articulation|                         |     Damped Least-Squares IK   |
|   - Multi-robot URDF builder  |                         |   - GPU eval_fk in ~0.5 ms    |
|   - Mimic joint constraints   |                         |   - Real-time convergence     |
|   - GPU forward kinematics    |                         |   - Sub-micrometer precision  |
+-------------------------------+                         +-------------------------------+
```

### Key Features
- **GPU-Accelerated**: Runs directly on CUDA-enabled NVIDIA GPUs (e.g. GeForce RTX 4080) with Warp kernels.
- **CRTK Contract Compliant**: Implements the full CRTK joint and Cartesian interfaces (`servo_jp`, `move_jp`, `servo_cp`, `move_cp`, `jaw/servo_jp`, `jaw/move_jp`, `state_command`).
- **Name-Based URDF Mapping**: Automatically inspects Newton articulations and maps joints and links by name, handling namespaces (`PSM1/`, `ECM/`) transparently.
- **Mimic Joint Support**: Extracts `<mimic>` joint relationships (such as dVRK PSM `jaw_1` and `jaw_2` mirroring `jaw`) and keeps them synchronized in simulation and state publishing.
- **Fast Inverse Kinematics**: Uses Jacobian-based damped least-squares IK evaluated directly against GPU forward kinematics (`newton.eval_fk`).
- **Multi-Arm Scenes**: Supports multi-arm setups (e.g. `ECM_PSM1_PSM2`) in a single combined simulation world.

---

## Environment Setup & Bootstrap

Because Newton and Warp require specific GPU runtime libraries, a dedicated workspace virtual environment (`.venv-newton`) is used and cached in `.generated/newton/python-runtime.json`.

To bootstrap or update the virtual environment:
```bash
./src/dvrk/dvrk_newton/scripts/bootstrap_venv.sh
```

To build the ROS 2 package:
```bash
colcon build --symlink-install --packages-select dvrk_newton
source install/setup.bash
```

---

## Running the Simulator

### Single Arm (PSM1)
```bash
# Default (interactive ViewerGL window):
ros2 launch dvrk_newton simulator.launch.py model:=PSM1

# Run headless without window:
ros2 launch dvrk_newton simulator.launch.py model:=PSM1 headless:=true

# Run with dVRK rqt monitor:
ros2 launch dvrk_newton simulator.launch.py model:=PSM1 rqt:=true
```

### Single Arm (ECM)
```bash
ros2 launch dvrk_newton simulator.launch.py model:=ECM endoscope:=Si_straight
```

### Multi-Arm Scene (ECM + PSM1 + PSM2)
```bash
ros2 launch dvrk_newton simulator.launch.py scene:=ECM_PSM1_PSM2.yaml
```

### Full Patient Cart with Stereo Camera (ECM + PSM1 + PSM2 + PSM3)
```bash
ros2 launch dvrk_newton simulator.launch.py scene:=ECM_PSM1_PSM2_PSM3.yaml
```

### OpenXR Quest Surgeon Console
Connects the patient cart simulation with the OpenXR surgeon console (`sawOpenXR`) and dVRK console video overlay over zero-copy GStreamer Unix-FD:
```bash
ros2 launch dvrk_newton open_xr.launch.py
```
Options for `open_xr.launch.py`:
- `scene:=<exercise.yaml>`: Exercise scene overlaid onto the patient cart (default: `tray_cubes.yaml`)
- `headless:=true` (default): HMD provides the visual output without opening an additional desktop viewer window
- `rqt:=true`: Starts the dockable dVRK Console, CRTK Arms tabs, and diagnostics monitor

### Headless & Viewer Options
Standardized on the `headless` option (matching `dvrk_isaac_sim`):
- `headless:=false` (default): Opens Newton's interactive `ViewerGL` OpenGL window with the camera framed directly on the robot workspace.
- `headless:=true`: Runs without the display window for headless training, testing, or server environments.

### Direct Node Invocation
The executable automatically re-executes into the Newton virtual environment if invoked with system Python:
```bash
ros2 run dvrk_newton simulator_node --model PSM1 --headless true
```

---

## Published CRTK Topics

When running (for example with `PSM1`), the node provides:
- `/PSM1/measured_js`: Measured joint state (angles and velocities)
- `/PSM1/setpoint_js`: Commanded joint setpoint
- `/PSM1/measured_cp`: Measured tool tip Cartesian pose in world frame
- `/PSM1/setpoint_cp`: Commanded Cartesian setpoint pose
- `/PSM1/measured_cv`: Measured tool Cartesian twist
- `/PSM1/jaw/measured_js`: Measured jaw angle
- `/PSM1/operating_state`: CRTK operating state (`ENABLED`, `PAUSED`, etc.)
- `/diagnostics`: Simulator diagnostic metrics (`simulation_hz`, `state_publish_hz`, `device`)

### Command Topics
- `/PSM1/servo_jp` (`sensor_msgs/msg/JointState`): Real-time joint servoing
- `/PSM1/move_jp` (`sensor_msgs/msg/JointState`): Velocity-limited joint trajectories
- `/PSM1/servo_cp` (`geometry_msgs/msg/PoseStamped`): Cartesian pose servoing (IK)
- `/PSM1/move_cp` (`geometry_msgs/msg/PoseStamped`): Cartesian trajectory moves (IK)
- `/PSM1/jaw/servo_jp` (`sensor_msgs/msg/JointState`): Jaw angle servoing
- `/PSM1/jaw/move_jp` (`sensor_msgs/msg/JointState`): Jaw angle trajectory moves
- `/PSM1/state_command` (`crtk_msgs/msg/StringStamped`): State commands (`enable`, `pause`, etc.)

---

## Running Tests

Run the test suite using pytest in the virtual environment:
```bash
source install/setup.bash
./.venv-newton/bin/pytest src/dvrk/dvrk_newton/test/ -v
```
