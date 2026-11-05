from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from simulation.isaac_michelson_control import Experiment


class SocketMonitorTests(unittest.TestCase):
    def test_depth_uses_native_tip_and_is_invariant_to_world_frame(self):
        poses = {}
        for index, label in enumerate(("lower", "upper")):
            knob = np.eye(4)
            knob[:3, 3] = [index * .024, 0., index * .024]
            driver = knob.copy()
            driver[:3, 3] += [.0003, -.012, .0004]
            poses[label + "_knob"] = knob
            poses[label + "_driver"] = driver
        experiment = Experiment.__new__(Experiment)
        experiment.m = SimpleNamespace(body=lambda name: SimpleNamespace(id=name))
        experiment.sim = SimpleNamespace(body_poses=poses, config={"driver_half_length_m": .020})
        expected = [.008, .0005, .008, .0005]
        np.testing.assert_allclose(experiment.socket_seating(), expected, atol=1e-12)
        world = np.eye(4)
        world[:3, :3] = Rotation.from_euler("xyz", [.7, -.3, 1.2]).as_matrix()
        world[:3, 3] = [4., -2., 3.]
        experiment.sim.body_poses = {name: world @ T for name, T in poses.items()}
        np.testing.assert_allclose(experiment.socket_seating(), expected, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
