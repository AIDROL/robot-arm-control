# Robot Arm Control for Optical Experiments

This repository contains the robot control and simulation components of the AIDROL optics project. Using an xArm7 model, it explores vision-based localization, optical component assembly, and closed-loop alignment with a fine alignment tool (FAT). Simulations use Isaac Sim and MuJoCo, with LightPipes for optical modeling.

The demonstration below reports simulation results. Real-world accuracy of the robot, gripper, cameras, mounts, and optical models requires calibration and physical validation.

## Physical xArm 7 bring-up

The physical xArm 7 passed seven individual joint round trips and an unloaded Gripper G2 cycle. Ubuntu Ethernet setup, SDK commands, and the acceptance record are in the [hardware guide](hardware/README.md).

## Latest demonstration: assembly and FAT alignment

Starting with an empty experiment area, the robot places five components and picks up a FAT based on the new STEP model. Its camera uses an ArUco marker to align with the mirror mount. The tool then adjusts two axes through contact, guided by simulated detector feedback, before returning to its staging area for a final measurement.

https://github.com/user-attachments/assets/9f2d193b-f4ac-44a1-89d8-9c4fe5396701

The 53-second video runs at 5× speed and shows the overview, global camera, wrist RGB-D, FAT side view, and recorded detector measurements. [Open the backup MP4](docs/media/fat-cad-michelson-replay.mp4?raw=true).

### Simulation results

| Acceptance item | Result |
| --- | --- |
| Component placement | 5/5 completed; maximum planar error 1.345 mm |
| FAT optical feedback | 12 adjustments; 0 unintended contacts |
| FAT return error | 1.347 mm |
| Interference visibility after tool return | 99.9959% on a 512 grid; 99.9958% on a 1024 grid |

The video comes from local acceptance run `fat-cad-full-20261118-02`, a 250.21-second PhysX trajectory. Alignment uses an Isaac controller and a scalar optical model; the mirror mount is a simulated design matched to the new FAT CAD. **This run is stored locally and is not included in a repository clone.**

## Setup

The Isaac workflow below has run on Ubuntu 22.04 with Isaac Sim 5.1 and Python 3.11. From the repository root, create the Python environment and install the optical dependencies:

```bash
python3.11 -m venv .venv
./.venv/bin/python -m pip install -r requirements-optics.txt
```

Isaac Sim uses its own `python.sh`. The script looks by default for `~/Documents/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh`. If Isaac is installed elsewhere, set:

```bash
export ISAAC_SIM_PYTHON=/path/to/isaac-sim/python.sh
```

For environment and migration background, see the [Isaac Sim handoff](docs/isaac-sim-5.1-handoff.md). Its branch names and migration plans are historical; use the current entry points on this page.

## Run and replay the latest workflow

Run all commands from the repository root. A new experiment requires an **output directory that does not already exist**, preserving any previous evidence:

```bash
# Start with an empty experiment area: assemble, align, return the tool, and validate.
# Choose a different directory if runs/fat-cad-full-new already exists.
env -u PYTHONPATH .venv/bin/python scripts/fat_closed_loop.py run --assembly \
  --folder runs/fat-cad-full-new
```

Check progress and validate the completed evidence:

```bash
# Show the current phase and log location.
env -u PYTHONPATH .venv/bin/python scripts/fat_closed_loop.py status \
  --folder runs/fat-cad-full-new

# Check the evidence manifest and acceptance results after the run completes.
env -u PYTHONPATH .venv/bin/python scripts/fat_closed_loop.py verify \
  --folder runs/fat-cad-full-new
```

Open the Isaac 3D replay with ArUco detection and a wrist depth view:

```bash
env -u PYTHONPATH .venv/bin/python scripts/fat_closed_loop.py replay \
  --folder runs/fat-cad-full-new --camera overview --speed 4
```

Replace `overview` with `global`, `wrist`, or `fat` to choose the starting camera; cameras can also be changed in the replay window. Add `--wait` if the experiment is still running to wait for the run and evidence sealing to finish. Replay reads a saved trajectory and does not rerun the physics.

If the local acceptance run is available, replace the directory above with `runs/fat-cad-full-20261118-02`. Trajectories, images, measurements, audit files, and `evidence-manifest.json` form the evidence package. Large run directories are not distributed with the repository.

## Other experiments

| Experiment | Description and entry point |
| --- | --- |
| Earlier Isaac workflow | Five-component assembly, FAT contact adjustment, and optical validation. The repository includes the historical acceptance trajectory; see the [validation report](docs/isaac-michelson-validation-2026-11-03.md). |
| MuJoCo Michelson workflow | Vision-based assembly, contact adjustment, and optical feedback: `./.venv/bin/python -m simulation.michelson_experiment --output runs/michelson-live --view`. See the [workflow documentation](docs/michelson-full-process.md). |
| MuJoCo FAT contact test | Standalone alignment on a preassembled mount: `./.venv/bin/python -m simulation.fat_mechanics --michelson --output runs/michelson-demo --view`. See the [FAT contact validation](docs/fat-contact-michelson.md). |
| Single-object visual grasp | Global-camera localization and wrist RGB-D grasp refinement: `./.venv/bin/python -m simulation.vision_pick_place --seed 7 --output runs/visual-grasp-demo --view`. See [visual pick and place](docs/vision-pick-place.md). |
| Basic five-object handling | Online Cartesian control demonstration: `./.venv/bin/python simulation/line_assembly.py --seed 7`. See the [controller documentation](docs/task-space-control.md). |

These entry points cover different scenes and validation scopes. For example, the earlier geometric FAT demonstration does not include physical contact adjustment, while the standalone FAT contact test starts with a preassembled mount. Refer to each experiment's documentation when interpreting its results.

## Repository structure and documentation

| Path | Contents |
| --- | --- |
| `simulation/isaac/` | Isaac Sim physics, cameras, and replay |
| `simulation/mujoco/` | MuJoCo backend; shared optics and experiment code lives in `simulation/` |
| `scripts/` | Entry point for the latest FAT workflow, status, verification, and replay |
| `assets/`, `configs/` | Robot, FAT, and optical component models and scene configuration |
| `docs/` | Experiment, validation, camera, and interface documentation |
| `runs/` | Local run records; only selected historical acceptance data is included in the repository |

- [Isaac cameras and optical viewer](docs/isaac-camera-optics-view.md): camera views, detector measurements, and replay options.
- [Camera detection overlays](docs/camera-detection-overlay.md): ArUco boxes, corners, IDs, and coordinate axes.
- [Earlier Isaac Michelson validation](docs/isaac-michelson-validation-2026-11-03.md): trajectory, results, and acceptance scope.
- [Optical model](docs/lightpipes-integration.md) and [optical detector](docs/optical-detector.md): LightPipes propagation and measurements.
- [Project coordination](docs/project-management.md): repository responsibilities and contribution workflow.

The robot model is based on the [UFACTORY xArm7 in MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/ufactory_xarm7). The real gripper, robot parameters, and camera calibration remain to be verified when hardware is available.
