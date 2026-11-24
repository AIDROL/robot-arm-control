"""Read firmware and telemetry from the xArm controller."""

import argparse
from datetime import datetime
from importlib.metadata import version
from ipaddress import IPv4Address
import math
import sys
import time


def read(query, **kwargs):
    code, value = query(**kwargs)
    if code != 0:
        raise RuntimeError(f"{query.__name__} failed (SDK code {code})")
    return value


def read_vector(query, size, **kwargs):
    values = read(query, **kwargs)
    if len(values) != size or not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"{query.__name__} returned invalid data: {values!r}")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", type=IPv4Address, default="192.168.1.212", help="Control box IPv4 address")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples at 2 Hz")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be at least 1")

    try:
        from xarm.wrapper import XArmAPI
    except ImportError as exc:
        print(f"Cannot import the xArm SDK. Install hardware/requirements.txt. {exc}", file=sys.stderr)
        return 1

    arm = XArmAPI(str(args.ip), is_radian=False, do_not_open=True)
    try:
        try:
            arm.connect()
        except Exception as exc:
            raise RuntimeError(f"Cannot reach controller at {args.ip}: {exc}") from exc
        if not arm.connected:
            raise RuntimeError(f"Cannot reach controller at {args.ip}")

        print(f"Controller: {args.ip}", flush=True)
        print(f"SDK: {version('xarm-python-sdk')}", flush=True)
        print(f"Firmware: {read(arm.get_version)}", flush=True)
        print(f"J1–J7 [deg]: {read_vector(arm.get_servo_angle, 7, is_real=True)}", flush=True)
        print(f"TCP XYZ [mm], RPY [deg]: {read_vector(arm.get_position, 6)}", flush=True)

        for index in range(args.samples):
            if index:
                time.sleep(0.5)
            state = read(arm.get_state)
            error, warning = read(arm.get_err_warn_code)
            timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
            print(f"{timestamp} state={state} error={error} warning={warning}", flush=True)
    except (OSError, RuntimeError) as exc:
        print(f"xArm check failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    finally:
        arm.disconnect()

    return 0


if __name__ == "__main__":
    sys.exit(main())
