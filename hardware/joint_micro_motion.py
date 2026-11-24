"""Preview or test xArm 7 joints and the optional Gripper G2 in sequence."""

import argparse
from ipaddress import IPv4Address
import math
import sys
import time

from read_xarm import read, read_vector


# Signed degrees from the current pose. Set an entry to 0 to skip that joint.
JOINT_DELTAS_DEG = {
    1: +10.0,  # J1
    2: -10.0,  # J2
    3: +5.0,  # J3
    4: +10.0,  # J4
    5: +10.0,  # J5
    6: -10.0,  # J6
    7: +15.0,  # J7
}
MAX_DELTA_DEG = 20.0
SPEED_DEG_S = 10.0
ACCEL_DEG_S2 = 15.0
TOLERANCE_DEG = 0.10
MOVE_TIMEOUT_S = 10.0

G2_STEP_MM = 15
G2_SPEED_MM_S = 15
G2_FORCE_PERCENT = 10
G2_MAX_OPENING_MM = 84
G2_TOLERANCE_MM = 1
G2_TIMEOUT_S = 5


def angles(arm):
    return read_vector(arm.get_servo_angle, 7, is_real=True)


def require_ready(arm):
    state = read(arm.get_state)
    faults = read(arm.get_err_warn_code)
    pending = read(arm.get_cmdnum)
    if state != 2 or list(faults) != [0, 0] or pending != 0:
        raise RuntimeError(f"Controller not idle and ready: state={state}, faults={faults}, queued={pending}")
    if arm.mode != 0 or arm.is_simulation_robot or arm.motor_enable_states[:7] != [1] * 7:
        raise RuntimeError("Confirm real robot, position control (mode 0), and all seven motors enabled in Studio")


def target_for(start, index, target_angle):
    target = start.copy()
    target[index] = target_angle
    return target


def check_limit(arm, target):
    if read(arm.is_joint_limit, joint=target) is not False:
        raise RuntimeError(f"Controller joint-limit check failed for {target}")


def check_feedback(target, actual, tolerance):
    errors = [abs(wanted - measured) for wanted, measured in zip(target, actual)]
    if max(errors) > tolerance:
        raise RuntimeError(f"Joint error exceeds {tolerance:.3f}°: {[round(error, 4) for error in errors]}")


def move(arm, target, label, tolerance):
    require_ready(arm)
    check_limit(arm, target)
    print(f"{label} target [deg]: {target}", flush=True)
    try:
        code = arm.set_servo_angle(
            angle=target,
            speed=SPEED_DEG_S,
            mvacc=ACCEL_DEG_S2,
            wait=True,
            timeout=MOVE_TIMEOUT_S,
        )
        if code != 0:
            raise RuntimeError(f"{label}: SDK motion code={code}")
        require_ready(arm)
        measured = angles(arm)
        print(f"{label} actual [deg]: {measured}", flush=True)
        check_feedback(target, measured, tolerance)
        return measured
    except (Exception, KeyboardInterrupt):
        if arm.connected:
            try:
                print(f"Software stop request returned: {arm.set_state(4)}", file=sys.stderr)
            except Exception as exc:
                print(f"Software stop request failed: {exc}", file=sys.stderr)
        raise


def gripper_opening(arm):
    opening = read(arm.get_gripper_g2_position, check_baud=False)
    if opening is None or not 0 <= opening <= G2_MAX_OPENING_MM:
        raise RuntimeError(f"G2 invalid opening: {opening!r}")
    return opening


def gripper_outbound_target(start):
    target = start + G2_STEP_MM
    if target > G2_MAX_OPENING_MM:
        raise RuntimeError(f"G2 needs another {G2_STEP_MM} mm of opening from {start} mm")
    return target


def test_gripper(arm):
    require_ready(arm)
    arm_pose = angles(arm)
    start = gripper_opening(arm)
    for label, target in (("open", gripper_outbound_target(start)), ("return", start)):
        require_ready(arm)
        check_feedback(arm_pose, angles(arm), TOLERANCE_DEG)
        print(f"G2 {label} target [mm]: {target}", flush=True)
        code = arm.set_gripper_g2_position(
            target,
            speed=G2_SPEED_MM_S,
            force=G2_FORCE_PERCENT,
            wait=True,
            timeout=G2_TIMEOUT_S,
            check_baud=False,
        )
        if code != 0:
            raise RuntimeError(f"G2 {label}: SDK motion code={code}")
        actual = gripper_opening(arm)
        print(f"G2 {label} actual [mm]: {actual}", flush=True)
        if abs(actual - target) > G2_TOLERANCE_MM:
            raise RuntimeError(f"G2 {label} exceeds ±{G2_TOLERANCE_MM} mm from the target")
        require_ready(arm)
        check_feedback(arm_pose, angles(arm), TOLERANCE_DEG)
    print("G2 PASS", flush=True)


def run(arm, joints, execute, with_gripper=False):
    if arm.axis != 7:
        raise RuntimeError(f"Expected a 7-axis controller, found {arm.axis} axes")
    baseline = angles(arm)
    print(f"Start pose [deg]: {baseline}", flush=True)
    check_limit(arm, baseline)
    for joint in joints:
        index = joint - 1
        delta = JOINT_DELTAS_DEG[joint]
        target = target_for(baseline, index, baseline[index] + delta)
        check_limit(arm, target)
        print(f"J{joint} plan: {baseline[index]:.4f}° → {target[index]:.4f}° → {baseline[index]:.4f}°", flush=True)

    if with_gripper:
        opening = gripper_opening(arm)
        outbound = gripper_outbound_target(opening)
        print(f"G2 plan [mm]: {opening} → {outbound} → {opening}", flush=True)

    if not execute:
        print("Preview complete. Review clearance and use --execute for live motion.")
        return

    # The values are -1 until the SDK has received the first motor report.
    deadline = time.monotonic() + 3.0
    while -1 in arm.motor_enable_states[:7] and arm.connected and time.monotonic() < deadline:
        time.sleep(0.05)

    for joint in joints:
        require_ready(arm)
        index = joint - 1
        delta = JOINT_DELTAS_DEG[joint]
        start = angles(arm)
        check_feedback(baseline, start, TOLERANCE_DEG)
        tolerance = min(TOLERANCE_DEG, abs(delta) / 3.0)

        outbound = target_for(start, index, start[index] + delta)
        measured = move(arm, outbound, f"J{joint} outbound", tolerance)
        back = target_for(measured, index, start[index])
        returned = move(arm, back, f"J{joint} return", tolerance)
        check_feedback(baseline, returned, TOLERANCE_DEG)
        print(f"J{joint} PASS, measured move {measured[index] - start[index]:+.4f}°", flush=True)

    if with_gripper:
        test_gripper(arm)
    suffix = " and Gripper G2" if with_gripper else ""
    print(f"PASS: {len(joints)}/{len(joints)} selected joints{suffix}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", type=IPv4Address, default="192.168.1.212")
    parser.add_argument("--joints", nargs="+", type=int, choices=range(1, 8), default=range(1, 8))
    parser.add_argument("--gripper", action="store_true", help="Include G2 after the selected joints")
    parser.add_argument("--execute", action="store_true", help="Send motion commands after preview and checks")
    args = parser.parse_args()
    joints = []
    for joint in dict.fromkeys(args.joints):
        delta = JOINT_DELTAS_DEG[joint]
        if not math.isfinite(delta) or abs(delta) > MAX_DELTA_DEG:
            parser.error(f"J{joint} offset must be finite and within ±{MAX_DELTA_DEG:g}° for this micro-motion test")
        if delta != 0:
            joints.append(joint)
    if not joints:
        parser.error("All selected joints have zero offset")

    try:
        from xarm.wrapper import XArmAPI
    except (ImportError, OSError) as exc:
        print(f"Cannot load xArm SDK: {exc}", file=sys.stderr)
        return 1

    arm = XArmAPI(str(args.ip), is_radian=False, do_not_open=True)
    try:
        arm.connect()
        if not arm.connected:
            raise RuntimeError(f"Cannot connect to {args.ip}")
        run(arm, joints, args.execute, with_gripper=args.gripper)
    except KeyboardInterrupt:
        print("Test interrupted. Check that the robot has stopped before resuming.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Test halted: {exc}. Confirm the robot is stopped and check Studio.", file=sys.stderr)
        return 1
    finally:
        arm.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
