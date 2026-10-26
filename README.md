# Robot Arm Control — xArm7 MuJoCo simulation

This repository is the robot-arm algorithm and control part of the optics robotics project. The current work runs without physical hardware. Team repository boundaries, ownership, and the contribution workflow are recorded in [docs/project-management.md](docs/project-management.md).

This repository starts with the UFACTORY xArm7 model from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/ufactory_xarm7). The model is a simplified MJCF description under BSD-3-Clause; the real laboratory gripper and calibrated robot parameters still need to be confirmed.

## macOS Apple Silicon setup

Use native arm64 Python 3.11. In this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The first model load downloads the pinned Menagerie release into its user cache. To check the model and control loop without opening a window:

```bash
./.venv/bin/python simulation/run_sim.py --headless
```

To watch the Home → joint 1 +20° → Home demonstration, run:

```bash
./.venv/bin/python simulation/run_sim.py
```

The arm returns Home after five seconds; close the viewer when finished. Other examples: `./.venv/bin/python simulation/run_sim.py --joint 2 --delta-deg 10` or `./.venv/bin/python simulation/run_sim.py --headless --joint 2 --delta-deg 10`. Calling `.venv/bin/python` directly also ensures the installed `mujoco_menagerie` package is used even if the shell has not activated the virtual environment.

The stock position actuators have a small steady error under gravity: holding the model's Home command produced up to about 0.0155 rad (0.89°) of arm-joint error in a headless check. The script reports the final error; its output should not be interpreted as precision placement performance.

The script uses MuJoCo's managed viewer, which runs with `python` on macOS. The passive viewer requires `mjpython`; its generated launcher fails when this project's virtual environment path contains spaces. The simulation uses the Menagerie `home` keyframe and sends smooth position targets to one joint while holding the other six at Home. The model includes a gripper actuator, but this demonstration does not move it.

## Five-component line assembly

Run the tabletop pick-and-place demonstration with a new scattered starting layout each time:

```bash
./.venv/bin/python simulation/line_assembly.py
```

For a reproducible run without a viewer:

```bash
./.venv/bin/python simulation/line_assembly.py --headless --seed 7
```

Watch the same layout, or use a fixed staging row for comparison:

```bash
./.venv/bin/python simulation/line_assembly.py --seed 7
./.venv/bin/python simulation/line_assembly.py --fixed
```

The default is an online Cartesian controller. It reads the current tool pose each physics step, uses the xArm7 Jacobian to calculate a damped joint velocity, and updates the seven joint position commands. The task phases are approach, grasp, lift, transfer, lower, release, and retreat. It does not solve a complete IK path in advance or replay a stored joint trajectory. After the fifth placement, the arm returns to a high park pose and maintains its commanded joint angles against gravity. The placement targets and initial object poses still come from simulator state.

You can run the same controller directly with `simulation/task_space_control.py`. To inspect the historical trajectory planner explicitly, use:

```bash
./.venv/bin/python simulation/line_assembly.py --seed 0 --legacy-planner bounded
```

The seed makes one random scene repeatable. The arm base is on the table's horizontal centreline (`x=0`) and faces positive Y. All five starting cubes are scattered in front of it (`x=-0.35–0.35 m`, `y=0.08–0.56 m`); the target row is directly in front at `y=0.35 m`, with `x=-0.18, -0.09, 0, 0.09, 0.18 m`. Its centre is aligned with the arm base. Random starts keep at least 14 cm between cubes and 14 cm from target centres. A conservative radial range excludes the arm base and distant edges. Yaw is within ±30° by default; change it with `--yaw-range-deg` (0–45). The controller reads each cube's actual simulated starting pose and rotates it toward 0° yaw during transfer.

The arm is mounted on a dark optical breadboard with a 25 mm visual hole pattern and moves five 5 × 5 × 8 cm proxy housings to a straight target line in front of it: lens 1, pinhole, lens 2, beam splitter, and mirror. The holes are visual only; the collision surface remains flat. Each target is marked on the table. The tool rotates 90° about the vertical axis at grasp and placement so the fingers open perpendicular to the target row. Headless runs check both finger contacts, lift, sustained grip and table clearance during transfer, placement within 1 cm in XY and 5° in yaw and tilt, and unwanted arm/table or arm/other-object contacts. See [docs/paper-components.md](docs/paper-components.md) for how this scene maps to the paper, [docs/task-space-control.md](docs/task-space-control.md) for the new controller, and [docs/planner-comparison.md](docs/planner-comparison.md) for the historical planner experiments.

This is a handling demonstration, not a physically valid optical path. The source poses are simulator ground truth; the patterns on top of the cubes are **visual placeholders, not valid ArUco markers**. Object geometry, mass, friction, and goal coordinates are illustrative. The real gripper, holder dimensions, camera calibration, and collision constraints must be checked when hardware is chosen. The reported placement errors are simulation results and should not be interpreted as real-world accuracy or the paper's 0.5 mm correction threshold.

## Current scope

The single-joint demonstration is the M1 control baseline. The line assembly adds Cartesian feedback control and object handling, but it does not yet include general obstacle avoidance, camera calibration, actual ArUco detection, or real robot commands. The Menagerie gripper should not be assumed to match the laboratory end effector.
