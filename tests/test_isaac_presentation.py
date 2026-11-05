import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from simulation.isaac.presentation_data import RecordedExperiment


class PresentationTimelineTests(unittest.TestCase):
    def fixture(self, folder):
        folder = Path(folder)
        (folder / "result.json").write_text(json.dumps({"success": True, "stage": "full",
            "started_from_empty_area": True, "time_s": 2.}))
        poses = np.broadcast_to(np.eye(4), (3, 1, 4, 4)).copy()
        poses[:, 0, 0, 3] = [0., 1., 2.]
        np.savez(folder / "physx-trajectory.npz", time_s=[0., 1., 2.], body_poses=poses,
                 body_names=["arm"], phase=["assembly", "adjustment", "return"],
                 measured_qpos=[[0.], [1.], [2.]], native_joint_names=["knob"], native_joint_qpos_indices=[0])
        for label, stamp, visibility in (("initial", 1., .1), ("released-512", 2., .97), ("released-1024", 2., .98)):
            path = folder / "optics" / label
            path.mkdir(parents=True)
            (path / "measurement.json").write_text(json.dumps({"time_s": stamp, "visibility": visibility}))
        return RecordedExperiment(folder)

    def test_measurements_never_arrive_before_their_recorded_time(self):
        with TemporaryDirectory() as folder:
            replay = self.fixture(folder)
            self.assertIsNone(replay.measurement(.99))
            self.assertEqual(replay.measurement(1.5)[1], "initial")
            self.assertEqual(replay.measurement(1.5)[2]["visibility"], .1)
            self.assertEqual(len(replay.history(1.5)), 1)
            self.assertIsNone(replay.measurement(.5))  # backward seeking

    def test_final_grid_check_and_final_pose_are_both_reached(self):
        with TemporaryDirectory() as folder:
            replay = self.fixture(folder)
            self.assertEqual(replay.measurement(2.)[1], "released-1024")
            self.assertAlmostEqual(replay.sample(.5)[0][0, 0], .5)
            self.assertEqual(replay.sample(2.)[2], "return")
            self.assertEqual(replay.sample(2.)[0][0, 0], 2.)
            self.assertEqual(replay.sample(20.)[0][0, 0], 2.)


if __name__ == "__main__":
    unittest.main()
