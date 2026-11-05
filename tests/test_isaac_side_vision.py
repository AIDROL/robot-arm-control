from pathlib import Path
import unittest
import cv2
import numpy as np
from simulation.isaac_michelson_control import detect_side_markers, upright_marker_rotation


class SideShadowTests(unittest.TestCase):
    def test_square_orientation_uses_all_four_corners(self):
        from scipy.spatial.transform import Rotation
        square = np.array([[-.012, .012, 0.], [.012, .012, 0.], [.012, -.012, 0.], [-.012, -.012, 0.]])
        R = Rotation.from_euler("z", .7).as_matrix()
        points = square @ R.T + [.3, .6, .92635]
        np.testing.assert_allclose(upright_marker_rotation(points), R, atol=1e-12)
        points[0, 1] += .001
        error = Rotation.from_matrix(R.T @ upright_marker_rotation(points)).magnitude()
        edge = points[1] - points[0]
        single_edge_error = abs(np.arctan2(edge[1], edge[0]) - .7)
        self.assertLess(error, .6 * single_edge_error)

    def test_real_near_field_shadow_frame_remains_detectable(self):
        for name in ("fat_side_shadow.png", "fat_side_deep_shadow.png", "fat_side_insertion_shadow.png"):
            with self.subTest(name=name):
                image = cv2.imread(str(Path(__file__).with_name("fixtures") / name))
                self.assertIsNotNone(image)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                standard = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
                self.assertIsNone(standard.detectMarkers(rgb)[1])
                corners, ids, _ = detect_side_markers(rgb)
                self.assertIn(14, ids.flatten())
                self.assertEqual(corners[list(ids.flatten()).index(14)].shape, (1, 4, 2))

    def test_dark_blank_frame_cannot_supply_a_pose(self):
        for gray in (0, 12, 255):
            _, ids, _ = detect_side_markers(np.full((720, 1280, 3), gray, np.uint8))
            self.assertIsNone(ids)


if __name__ == "__main__":
    unittest.main()
