import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
from simulation.isaac.live_stream import DisplayBuffer
from simulation.isaac.camera_optics_monitor import observations


class LiveDisplayTests(unittest.TestCase):
    def test_interpolation_uses_observed_motion_and_never_extrapolates(self):
        buffer = DisplayBuffer(delay=.1)
        for wall, position, degrees in ((10., 0., 0.), (11., 2., 90.)):
            pose = np.eye(4)[None, :, :]
            pose[0, 0, 3] = position
            pose[0, :3, :3] = Rotation.from_euler("z", degrees, degrees=True).as_matrix()
            buffer.push({"wall_time_s": wall, "time_s": wall - 10., "phase": "motion", "body_poses": pose})
        position, q, simulation_time, _, _ = buffer.at(10.6)
        np.testing.assert_allclose(position[0], [1., 0., 0.])
        self.assertAlmostEqual(Rotation.from_quat(q[0]).as_euler("xyz", degrees=True)[2], 45.)
        self.assertAlmostEqual(simulation_time, .5)
        np.testing.assert_allclose(buffer.at(100.)[0][0], [2., 0., 0.])
        np.testing.assert_allclose(buffer.at(0.)[0][0], [0., 0., 0.])

    def test_live_cameras_are_visible_before_any_control_capture(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            live = folder / "live"
            live.mkdir()
            for name in ("global", "wrist", "side"):
                (live / (name + ".jpg")).touch()
            (live / "preview.json").write_text(json.dumps({"time_s": 0.,
                "frames": {name: name + ".jpg" for name in ("global", "wrist", "side")}}))
            streams = observations(folder)
            self.assertTrue(all(streams[key][0][0] == 0. for key in ("global", "wrist", "side")))
            self.assertTrue(all(not rows for rows in observations(folder, include_live=False).values()))


if __name__ == "__main__":
    unittest.main()
