# dvrk_newton

NVIDIA Newton physics and kinematics backend for the common [dvrk_simulator_base](../dvrk_simulator_base) package.

This package runs hardware-accelerated robot kinematics and physics simulation on NVIDIA GPUs using NVIDIA Newton (`newton`) and NVIDIA Warp (`warp-lang`).

---

## Architecture Overview

```text
ROS / CRTK clients
        |
ROS process: DvrkNewtonNode + ArmRosInterface
        |
        | private AF_UNIX stream socket pair (versioned JSON)
        | commands -->     <-- complete scene snapshots / events / metrics
        |
Newton process: NewtonRuntime
        | command acceptance, reference frames, trajectories, operating state
        | backend FK/IK, physics, contacts, grasping
        + camera worker and existing video transport
```

The ROS frontend starts a fresh simulation interpreter selected by
`DVRK_NEWTON_PYTHON` / the existing runtime resolver. It does not import Newton
or Warp. ROS launch uses its own Python interpreter, and the child receives
one inherited Unix socket descriptor. No socket pathname, shared memory, or
pickle is used. Both interpreters need access to the built dVRK packages.

Commands carry plain joint values or a Cartesian pose plus `frame_id`. The
simulation process resolves Cartesian targets against the last completed
simulation scene at the start of a step. All targets consumed in that step
use that same measured ECM pose; an ECM command applied in the same step
changes the reference for the next step. Empty PSM frames default to
`ECM_view` when an ECM is present; explicit world/parent and arm-base frames
keep their existing meanings. Newton also computes the world-to-view and
world-to-base values for publication, using one complete scene snapshot.
The ROS process performs no Cartesian reference-frame conversions.

ROS mailboxes retain bounded state-command queues and superseding arm/jaw
motion commands. Only one command batch is in flight until the simulator
acknowledges consumption at a step boundary. State snapshots may replace
unsent older snapshots; operating-state transitions and warnings travel as
ordered reliable messages. Transport writes are nonblocking and bounded, so
simulation steps do not wait for ROS publications. A stalled reliable queue
or disconnected peer ends the session with a diagnostic error. Startup
silence is limited to 180 seconds and running-worker silence to 30 seconds.

Closing the viewer or reaching the test timeout stops both processes. On
shutdown the frontend requests simulator cleanup, waits for the worker, and
terminates/reaps it if cleanup does not finish. Video stays in the Newton
process and uses its existing transport.

### Key Features
- **GPU-Accelerated**: Runs directly on CUDA-enabled NVIDIA GPUs (e.g. GeForce RTX 4080) with Warp kernels.
- **CRTK Contract Compliant**: Implements the full CRTK joint and Cartesian interfaces (`servo_jp`, `move_jp`, `servo_cp`, `move_cp`, `jaw/servo_jp`, `jaw/move_jp`, `state_command`).
- **Name-Based URDF Mapping**: Automatically inspects Newton articulations and maps joints and links by name, handling namespaces (`PSM1/`, `ECM/`) transparently.
- **Mimic Joint Support**: Extracts `<mimic>` joint relationships (such as dVRK PSM `jaw_1` and `jaw_2` mirroring `jaw`) and keeps them synchronized in simulation and state publishing.
- **Fast Inverse Kinematics**: Uses a CPU URDF chain with an analytic Jacobian for Cartesian commands. Startup checks its home and perturbed tool poses against Newton FK; if they disagree, the simulator warns and uses the original GPU finite-difference solver for that arm.
- **Multi-Arm Scenes**: Supports multi-arm setups (e.g. `ECM_PSM1_PSM2`) in a single combined simulation world.
- **Independent camera worker**: The simulation loop offers completed body poses to a one-slot mailbox. A camera worker renders the newest pose at the configured wall-clock rate and drops older pending poses when rendering is slow. The worker owns a separate Newton model because the Warp raytracer refits its shape BVH. Rendering and simulation still share the GPU, so camera work can still affect simulation throughput through GPU contention.

`/diagnostics` reports the five shared metrics: actual `simulation_hz`,
`camera_hz`, `state_publish_hz`, `snapshot_receive_hz`, and `snapshot_age_ms`.
See [the shared IPC architecture](../dvrk_simulator_base/README.md#unix-socket-process-boundary)
for their definitions. Detailed phase timings and command counters are omitted.

---

## Environment Setup & Bootstrap

Because Newton and Warp require specific GPU runtime libraries, dependencies are listed in `requirements.txt`. You can configure the environment in either of two ways:

1. **Use the bootstrap script** to create a dedicated workspace virtual environment (`.venv-newton`) with `--system-site-packages` and install dependencies with pip:
   ```bash
   ./src/dvrk/dvrk_newton/scripts/bootstrap_venv.sh
   ```
   The script prompts for confirmation before creating the environment and installing packages from `requirements.txt`. (Pass `-y` or `--yes` to proceed non-interactively).

2. **Use your own Python environment** and install dependencies using pip:
   ```bash
   pip install -r src/dvrk/dvrk_newton/requirements.txt
   ```

The selected interpreter is cached in `~/.cache/dvrk_newton/python-runtime.json`.

To build the ROS 2 package:
```bash
colcon build --symlink-install --packages-select dvrk_newton
source install/setup.bash
```

---

## Running the Simulator

### Multi-Arm Scene (ECM + PSM1 + PSM2)
```bash
ros2 launch dvrk_newton simulator.launch.py scene:=ECM_PSM1_PSM2.yaml
```

### Full Patient Cart with Stereo Camera (ECM + PSM1 + PSM2 + PSM3)
```bash
ros2 launch dvrk_newton simulator.launch.py scene:=ECM_PSM1_PSM2_PSM3.yaml
```

The scene YAML selects the camera renderer under `scene.camera`:

```yaml
scene:
  camera:
    renderer: opengl  # or warp_raytrace (default when omitted)
```

`opengl` uses Newton's offscreen `ViewerGL` in the camera worker and keeps the same
mono or side-by-side stereo Unix-FD output. It requires a working OpenGL context;
on a machine without a display, set `PYGLET_HEADLESS=1` before starting ROS.
The existing `horizontal_fov_deg` setting is applied as a vertical FOV by the
Warp camera helper. OpenGL retains that framing for compatibility.

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
The ROS executable starts a separate Newton worker using the selected environment:
```bash
ros2 run dvrk_newton simulator_node --scene ECM_PSM1_PSM2_PSM3.yaml --headless true
```

---

## Running Tests

Run the test suite using pytest in the virtual environment:
```bash
source install/setup.bash
./.venv-newton/bin/pytest src/dvrk/dvrk_newton/test/ -v
```
