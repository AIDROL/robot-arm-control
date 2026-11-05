"""Atomic native-state transport and bounded display interpolation."""
from collections import deque
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def write_state(folder, names, poses, simulation_time, phase):
    folder = Path(folder)
    temporary = folder / "state.npz.tmp"
    with temporary.open("wb") as stream:
        np.savez(stream, body_names=np.array(names), body_poses=poses,
                 time_s=simulation_time, wall_time_s=time.monotonic(), phase=phase)
    temporary.replace(folder / "state.npz")


class DisplayBuffer:
    """Interpolate recent observed poses; clamp rather than predict future motion."""
    def __init__(self, delay=.10):
        self.delay = delay
        self.samples = deque(maxlen=32)

    def push(self, state):
        wall = float(state["wall_time_s"])
        if self.samples and wall <= self.samples[-1][0]:
            return False
        poses = np.array(state["body_poses"])
        self.samples.append((wall, float(state["time_s"]), str(state["phase"]),
                             poses[:, :3, 3], Rotation.from_matrix(poses[:, :3, :3]).as_quat()))
        return True

    def at(self, wall):
        if not self.samples:
            raise ValueError("No native state yet")
        target = wall - self.delay
        a = b = self.samples[-1]
        if target <= self.samples[0][0]:
            a = b = self.samples[0]
        else:
            for left, right in zip(self.samples, list(self.samples)[1:]):
                if left[0] <= target <= right[0]:
                    a, b = left, right
                    break
        alpha = 0. if a is b else (target - a[0]) / (b[0] - a[0])
        q0, q1 = a[4], b[4].copy()
        q1[np.sum(q0 * q1, axis=1) < 0] *= -1
        q = (1 - alpha) * q0 + alpha * q1
        q /= np.linalg.norm(q, axis=1)[:, None]
        position = (1 - alpha) * a[3] + alpha * b[3]
        simulation_time = (1 - alpha) * a[1] + alpha * b[1]
        oldest, newest = self.samples[0], self.samples[-1]
        rate = (newest[1] - oldest[1]) / max(newest[0] - oldest[0], 1e-6)
        return position, q, simulation_time, b[2], rate
