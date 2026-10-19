# Robot Arm Control — xArm7 MuJoCo baseline

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

## Current scope

This is the M1 control baseline only. It does not yet provide inverse kinematics, collision-aware trajectories, object grasping, camera calibration, or real robot commands. The Menagerie gripper should not be assumed to match the laboratory end effector.
