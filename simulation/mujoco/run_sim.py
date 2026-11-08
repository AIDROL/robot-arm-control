"""Minimal xArm7 Home and single-joint MuJoCo demonstration."""

from __future__ import annotations

import argparse
import math

import mujoco
import mujoco_menagerie as menagerie


def smoothstep(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def command_at(t: float, home: float, offset: float) -> float:
    if t < 2.0:
        return home + offset * smoothstep(t / 2.0)
    if t < 3.0:
        return home + offset
    if t < 5.0:
        return home + offset * (1.0 - smoothstep((t - 3.0) / 2.0))
    return home


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Run without a viewer")
    parser.add_argument("--joint", type=int, choices=range(1, 8), default=1)
    parser.add_argument("--delta-deg", type=float, default=20.0)
    args = parser.parse_args()

    model = menagerie.load("ufactory_xarm7")
    data = mujoco.MjData(model)
    keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if keyframe_id < 0:
        raise RuntimeError("The xArm7 model has no 'home' keyframe")

    joint_ids = []
    actuator_ids = []
    for number in range(1, 8):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{number}")
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act{number}")
        if joint_id < 0 or actuator_id < 0:
            raise RuntimeError(f"Missing xArm7 joint/actuator {number}")
        joint_ids.append(joint_id)
        actuator_ids.append(actuator_id)

    mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
    mujoco.mj_forward(model, data)
    home_ctrl = model.key_ctrl[keyframe_id].copy()
    selected_joint = joint_ids[args.joint - 1]
    selected_actuator = actuator_ids[args.joint - 1]
    home_angle = float(home_ctrl[selected_actuator])
    offset = math.radians(args.delta_deg)
    target = home_angle + offset
    lower, upper = model.actuator_ctrlrange[selected_actuator]
    if not math.isfinite(target) or not lower <= target <= upper:
        raise ValueError(
            f"joint{args.joint} target {target:.3f} rad is outside "
            f"[{lower:.3f}, {upper:.3f}] rad"
        )

    print(f"Loaded xArm7: {len(joint_ids)} arm joints, {model.nu} actuators")
    print(f"Home -> joint{args.joint} +{args.delta_deg:g} deg -> Home")

    duration = 6.0
    peak_excursion = [0.0]

    def control_callback(_model: mujoco.MjModel, state: mujoco.MjData) -> None:
        state.ctrl[:] = home_ctrl
        state.ctrl[selected_actuator] = command_at(state.time, home_angle, offset)
        actual_angle = float(state.qpos[model.jnt_qposadr[selected_joint]])
        peak_excursion[0] = max(peak_excursion[0], abs(actual_angle - home_angle))

    mujoco.set_mjcb_control(control_callback)
    try:
        if args.headless:
            while data.time < duration:
                mujoco.mj_step(model, data)
        else:
            from mujoco import viewer as mj_viewer

            print("The arm returns Home after 5 s; close the viewer when finished.")
            mj_viewer.launch(model, data)
    finally:
        mujoco.set_mjcb_control(None)

    arm_qpos = [float(data.qpos[model.jnt_qposadr[j]]) for j in joint_ids]
    home_qpos = [float(model.key_qpos[keyframe_id, model.jnt_qposadr[j]]) for j in joint_ids]
    max_error = max(abs(actual - expected) for actual, expected in zip(arm_qpos, home_qpos))
    print(f"Simulated {data.time:.2f} s; peak joint excursion: {math.degrees(peak_excursion[0]):.2f} deg")
    print(f"Maximum final Home joint error: {max_error:.4f} rad")
    if args.headless and peak_excursion[0] < abs(offset) * 0.5:
        raise RuntimeError("Selected joint did not move at least half the commanded amount")


if __name__ == "__main__":
    main()
