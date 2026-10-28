"""Image-only checks with independently specified camera/plane geometry."""
import unittest

import cv2
import numpy as np

from simulation.global_camera import DICTIONARY, locate


class GlobalCameraTests(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[4000., 0, 239.5], [0, 4000., 199.5], [0, 0, 1]])
        self.T = np.diag([1., -1., -1., 1.])
        self.T[2, 3] = 1  # camera one metre above z=0, looking straight down
        self.rgb = np.full((400, 480, 3), 255, dtype=np.uint8)

    def add_tag(self):
        # A 24 mm square at 1 m subtends 96 px with focal length 4000 px.
        marker = cv2.aruco.generateImageMarker(DICTIONARY, 2, 96)
        self.rgb[152:248, 192:288] = marker[:, :, None]

    def test_blank_frame_has_no_fallback_pose(self):
        estimates, _, _ = locate(self.rgb, self.K, self.T, 0.)
        self.assertEqual(estimates, [])

    def test_known_centred_tag_in_world_coordinates(self):
        self.add_tag()
        estimates, _, _ = locate(self.rgb, self.K, self.T, 0.)
        self.assertEqual(len(estimates), 1)
        self.assertEqual(estimates[0]['id'], 2)
        self.assertTrue(estimates[0]['valid'])
        pose = np.array(estimates[0]['T_world_marker'])
        np.testing.assert_allclose(pose[:3, 3], [0, 0, 0], atol=0.0003)
        np.testing.assert_allclose(pose[:3, :3], np.eye(3), atol=0.01)

    def test_plane_behind_camera_is_rejected(self):
        self.add_tag()
        estimates, _, _ = locate(self.rgb, self.K, self.T, 2.)
        self.assertEqual(estimates, [])


if __name__ == '__main__':
    unittest.main()
