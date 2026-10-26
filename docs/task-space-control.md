# Online Cartesian control

The supported five-object demonstration now runs through `simulation/task_space_control.py`. It imports `make_scene` from `simulation/line_assembly.py`, so the xArm7 model, gripper, optical table, object shapes, random start generator, and five target positions are unchanged. Running `simulation/line_assembly.py` without a legacy flag starts the same new controller.

## Control method

Each physics step, the controller reads the actual position and orientation of `pinch_center` and computes the site Jacobian with MuJoCo's `mj_jacSite`. A damped least-squares pseudoinverse maps desired Cartesian linear and angular velocity to seven joint velocities:

```text
tool velocity = trajectory feedforward + gain × current pose error
joint velocity = Jᵀ (J Jᵀ + λ² I)⁻¹ tool velocity + null-space posture term
```

The joint velocity is speed-limited and integrated into the existing MuJoCo position-actuator commands. A small null-space term draws redundant joints toward Home. A command-lead limit prevents the target angles from getting far ahead of the measured joints. No numerical IK target is solved for each waypoint, and no full joint trajectory is calculated before the simulation starts.

The controller switches through pregrasp, approach, close, lift, transfer, lower, release, and retreat for each object. Tool pose targets follow time-scaled Cartesian motion; transitions wait for the actual tool to arrive within 5 mm and 3° after the nominal duration. The gripper command ramps during close and release. Object starting poses are MuJoCo ground truth, not camera measurements.

After the fifth placement, the arm returns to its high Home tool pose and holds the final joint target with position feedback. It does not reset the target to the measured joint angle. That distinction matters: resetting the target removed the servo's gravity-supporting torque and let the arm sag onto the table several seconds after the task appeared complete.

## Running and validation

```bash
./.venv/bin/python simulation/line_assembly.py --headless --seed 7
./.venv/bin/python simulation/line_assembly.py --seed 7
./.venv/bin/python simulation/task_space_control.py --headless --fixed
```

Full physics rollouts passed on random seeds 0–19, seed 1758901330, and the fixed layout: each run placed all five objects within the existing 1 cm XY and 5° orientation limits, showed contact on both finger pads and a sustained grip during transfer, and recorded no unwanted arm/table, arm/other-object, carried-cube/table, or carried-cube/other-cube contacts. Headless validation now continues for 12 seconds after parking and requires the tool to stay at least 20 cm above the table. A separate 30-second hold check on seed 7 kept the tool near 1.02 m and all five cubes stationary. The controller is deterministic for a fixed seed and software environment. The previous seed 18, which defeated the single-solution IK reference, also passed.

This is a task-space feedback controller with fixed pick-and-place goals. It does not plan around arbitrary new obstacles or account for camera uncertainty. Collision checks currently validate the run and stop headless runs that violate them; they do not generate an avoidance detour. The controller uses the Menagerie actuator model and should not be sent to a real xArm without hardware-specific limits, calibration, and safety integration.
