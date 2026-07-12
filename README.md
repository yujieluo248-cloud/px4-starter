# px4-starter

`px4-starter` is a PX4 + ROS 2 + Gazebo learning workspace based on
[`SathanBERNARD/PX4-ROS2-Gazebo-Drone-Simulation-Template`](https://github.com/SathanBERNARD/PX4-ROS2-Gazebo-Drone-Simulation-Template).

This fork extends the original template for:

- PX4 SITL
- ROS 2 Offboard control
- Gazebo simulation
- waypoint flight
- continuous traversal of a wide gate at known coordinates

The current verified gate experiment uses the `test_world_gate_wide` Gazebo
world with the `gz_x500_mono_cam` model. The latest successful traversal node is
`offboard_wide_gate_traversal_v3`, which performs automatic takeoff, smooth
approach, continuous gate crossing, after-gate hold, and automatic landing.

This is not vision-based autonomous gate traversal. The v3 traversal uses known
gate geometry and fixed PX4 local-frame trajectory parameters.

## Verified Environment

The project has been verified with:

- Windows + WSL2
- Ubuntu 22.04
- ROS 2 Humble
- PX4 v1.15.2
- `px4_msgs` branch: `release/1.15`
- `px4_msgs` commit: `a1045ec4feb6d709bdecaf3895f1d5b43a5dabb8`
- `px4_msgs` tag: `v1.15.4`
- Gazebo Sim 8.14.0
- MicroXRCEAgent UDP port `8888`
- simulation model: `gz_x500_mono_cam`
- PX4 autostart airframe: `PX4_SYS_AUTOSTART=4010`

## Repository Structure

```text
px4-starter/
├── PX4-Autopilot_PATCH/
│   └── Tools/simulation/gz/
│       ├── worlds/test_world_gate_wide.sdf
│       └── models/test_world_gate_wide/
├── ws_ros2/
│   └── src/
│       ├── my_offboard_ctrl/
│       └── px4_msgs/                 # Git submodule
├── install_px4_gz_ros2_for_ubuntu.sh
├── README.md
└── README.txt
```

Important paths:

- `ws_ros2/src/my_offboard_ctrl/`  
  ROS 2 Python package containing PX4 Offboard control examples and gate
  traversal experiments.

- `ws_ros2/src/my_offboard_ctrl/my_offboard_ctrl/offboard_takeoff_hover.py`  
  Basic takeoff and hover example.

- `ws_ros2/src/my_offboard_ctrl/my_offboard_ctrl/offboard_single_waypoint.py`  
  Single local waypoint example.

- `ws_ros2/src/my_offboard_ctrl/my_offboard_ctrl/offboard_waypoint_patrol.py`  
  Multi-waypoint patrol example.

- `ws_ros2/src/my_offboard_ctrl/my_offboard_ctrl/offboard_wide_gate_traversal_v1.py`  
  Wide-gate local direction verification. This does not cross the gate.

- `ws_ros2/src/my_offboard_ctrl/my_offboard_ctrl/offboard_wide_gate_traversal_v2.py`  
  Earlier staged wide-gate traversal attempt.

- `ws_ros2/src/my_offboard_ctrl/my_offboard_ctrl/offboard_wide_gate_traversal_v3.py`  
  Current verified continuous wide-gate traversal version.

- `ws_ros2/src/my_offboard_ctrl/my_offboard_ctrl/archive/`  
  Failed or obsolete experiment versions kept for reference.

- `PX4-Autopilot_PATCH/`  
  Custom Gazebo world and model files that must be copied into a local
  `PX4-Autopilot` checkout.

- `ws_ros2/src/px4_msgs`  
  Git submodule for PX4 message definitions.

## Clone on a New Computer

Clone with submodules:

```bash
git clone --recursive https://github.com/yujieluo248-cloud/px4-starter.git
cd px4-starter
```

If the repository was cloned without `--recursive`, initialize the submodule:

```bash
git submodule update --init --recursive
```

Verify `px4_msgs`:

```bash
cd ws_ros2/src/px4_msgs
git branch --show-current
git rev-parse HEAD
git describe --tags --exact-match HEAD
```

Expected:

```text
release/1.15
a1045ec4feb6d709bdecaf3895f1d5b43a5dabb8
v1.15.4
```

## Install PX4, Gazebo, ROS 2, and MicroXRCEAgent

The repository includes the original install helper:

```bash
cd ~/px4-starter
./install_px4_gz_ros2_for_ubuntu.sh
```

The script installs Gazebo Harmonic, PX4 v1.15.2, ROS 2 Humble, Python
dependencies, and MicroXRCEAgent. Review the script before running it on a new
machine because it installs system packages and clones PX4 into `~/PX4-Autopilot`.

If installing manually, keep these version relationships:

- PX4: `v1.15.2`
- ROS 2: Humble
- `px4_msgs`: `release/1.15`, commit
  `a1045ec4feb6d709bdecaf3895f1d5b43a5dabb8`
- Gazebo: Harmonic / Gazebo Sim 8

## Apply the Custom Wide-Gate World

Copy the patch files into the local PX4 checkout:

```bash
cd ~/px4-starter
cp -r PX4-Autopilot_PATCH/* ~/PX4-Autopilot/
```

After copying, these files should exist:

```text
~/PX4-Autopilot/Tools/simulation/gz/worlds/test_world_gate_wide.sdf
~/PX4-Autopilot/Tools/simulation/gz/models/test_world_gate_wide/model.sdf
~/PX4-Autopilot/Tools/simulation/gz/models/test_world_gate_wide/meshes/test_terrain_wide_gate.dae
```

The wide-gate model uses:

- terrain visual mesh:
  `model://test_world_gate_wide/meshes/test_terrain_wide_gate.dae`
- terrain collision mesh:
  `model://test_world_gate_wide/meshes/test_terrain_wide_gate.dae`
- explicit box collisions for the left post, right post, bottom beam, and top
  beam.

The gate opening is approximately:

- width: `0.98 m`
- height: `1.48 m`
- gate plane world Y: `3.227722`
- opening world X: `0.05 .. 1.03`
- opening world Z: `0.20 .. 1.68`

## Build the ROS 2 Workspace

```bash
cd ~/px4-starter/ws_ros2
source /opt/ros/humble/setup.bash
colcon build --packages-select my_offboard_ctrl --symlink-install --parallel-workers 2
source install/setup.bash
```

Check the available executables:

```bash
ros2 pkg executables my_offboard_ctrl
```

The wide-gate traversal entry should include:

```text
my_offboard_ctrl offboard_wide_gate_traversal_v3
```

## Run the Simulation

Use separate terminals.

### Terminal 1: MicroXRCEAgent

```bash
MicroXRCEAgent udp4 -p 8888
```

### Terminal 2: PX4 SITL + Gazebo

```bash
PX4_SYS_AUTOSTART=4010 \
PX4_SIM_MODEL=gz_x500_mono_cam \
PX4_GZ_MODEL_POSE="1,1,0.1,0,0,0" \
PX4_GZ_WORLD=test_world_gate_wide \
~/PX4-Autopilot/build/px4_sitl_default/bin/px4
```

Notes:

- `PX4_SYS_AUTOSTART=4010` selects the `gz_x500_mono_cam` airframe.
- `PX4_SIM_MODEL=gz_x500_mono_cam` loads the X500 model with monocular camera.
- `PX4_GZ_MODEL_POSE="1,1,0.1,0,0,0"` is the verified start pose for the
  wide-gate experiment.
- `PX4_GZ_WORLD=test_world_gate_wide` loads the custom world from the patch.

### Terminal 3: ROS 2 Offboard Node

Build and source the workspace first:

```bash
cd ~/px4-starter/ws_ros2
source /opt/ros/humble/setup.bash
source install/setup.bash
```

Run the current verified traversal:

```bash
ros2 run my_offboard_ctrl offboard_wide_gate_traversal_v3
```

## Expected v3 Behavior

`offboard_wide_gate_traversal_v3` should execute this state machine:

```text
WAIT_FOR_POSITION
PRESTREAM
REQUEST_OFFBOARD_ARM
WAIT_FOR_OFFBOARD_ARM
SETTLE_HOME_AFTER_ARM
TAKEOFF
SMOOTH_APPROACH
PRE_GATE_ALIGNMENT_CHECK
SMOOTH_GATE_CROSS
AFTER_GATE_HOLD
LAND
DONE
```

The node:

- waits for valid PX4 local position;
- streams Offboard setpoints before requesting OFFBOARD and ARM;
- confirms `VehicleCommandAck`, `ARMED`, and `OFFBOARD`;
- waits for local position to settle after arming;
- locks a stable takeoff home;
- takes off using a fixed position setpoint;
- approaches the gate with smooth position setpoints and velocity feedforward;
- checks lateral and vertical alignment before crossing;
- crosses the gate continuously without stopping in the gate plane;
- holds briefly after the gate;
- requests AUTO LAND;
- stops its timer after disarm / DONE.

Important v3 parameters:

```text
FLIGHT_UP_M = 0.55
GATE_PLANE_DX_M = 2.23
GATE_CENTER_DY_M = -0.45
PRE_GATE_DX_M = 1.45
PRE_GATE_DY_M = -0.45
AFTER_GATE_DX_M = 2.75
AFTER_GATE_DY_M = -0.45
SMOOTH_APPROACH_SECONDS = 8.00
SMOOTH_GATE_CROSS_SECONDS = 6.50
AFTER_GATE_HOLD_SECONDS = 1.50
```

PX4 local-frame convention used by these scripts:

- PX4 local position is NED.
- Upward motion means local `z` becomes more negative.
- Relative height is computed as `home_z - current_z`.

## Other Useful Nodes

```bash
ros2 run my_offboard_ctrl offboard_takeoff_hover
ros2 run my_offboard_ctrl offboard_single_waypoint
ros2 run my_offboard_ctrl offboard_waypoint_patrol
ros2 run my_offboard_ctrl offboard_wide_gate_traversal_v1
ros2 run my_offboard_ctrl offboard_wide_gate_traversal_v2
ros2 run my_offboard_ctrl offboard_wide_gate_traversal_v3
```

Use `v1` for direction verification, not gate traversal. Use `v3` for the
current known-coordinate wide-gate traversal.

## Safety and Troubleshooting

- Do not run traversal nodes until PX4, Gazebo, and MicroXRCEAgent are already
  running and `/fmu/out/vehicle_local_position` is publishing.
- If the node does not print accepted OFFBOARD / ARM acknowledgements, stop and
  check MicroXRCEAgent and ROS 2 topic connectivity.
- If `xy_valid` or `z_valid` becomes false, or `dead_reckoning` becomes true,
  the traversal node requests AUTO LAND.
- If OFFBOARD or ARM is lost during flight, the traversal node requests AUTO
  LAND.
- If local position jumps abruptly, v3 logs
  `POSITION_DISCONTINUITY_OR_COLLISION` and requests AUTO LAND.
- If the vehicle does not reach a safe takeoff height in time, v3 logs
  `TAKEOFF_FAILED_TO_REACH_SAFE_HEIGHT` and requests AUTO LAND.
- The current traversal is based on fixed geometry. If the world, start pose,
  vehicle model, or gate geometry changes, re-check the route parameters before
  flying.

## Development Checks

Compile-check the current traversal node:

```bash
cd ~/px4-starter/ws_ros2
python3 -m py_compile src/my_offboard_ctrl/my_offboard_ctrl/offboard_wide_gate_traversal_v3.py
```

Build the ROS 2 package:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select my_offboard_ctrl --symlink-install --parallel-workers 2
```

Confirm the entry point:

```bash
source install/setup.bash
ros2 pkg executables my_offboard_ctrl | grep offboard_wide_gate_traversal_v3
```

## Notes on Scope

This repository is for simulation and Offboard-control learning. The wide-gate
experiment is a controlled Gazebo SITL experiment with known coordinates. It is
not a perception stack, visual servoing system, or real-aircraft deployment
package.
