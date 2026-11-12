import unittest

import numpy as np

from simulation.isaac.display_rendering import smooth_triangle_normals


class DisplayNormalsTests(unittest.TestCase):
    def test_curved_shared_edge_is_smooth(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                           [0., -np.cos(np.pi/6), np.sin(np.pi/6)]])
        faces = np.array([[0, 1, 2], [1, 0, 3]])
        normals = smooth_triangle_normals(points, faces).reshape(2, 3, 3)
        np.testing.assert_allclose(normals[0, 0], normals[1, 1])
        self.assertGreater(normals[0, 0, 1], .1)
        np.testing.assert_allclose(np.linalg.norm(normals, axis=-1), 1.)

    def test_right_angle_keeps_separate_face_normals(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        normals = smooth_triangle_normals(points, np.array([[0, 1, 2], [1, 0, 3]]))
        np.testing.assert_allclose(normals[:3], [[0., 0., 1.]]*3)
        np.testing.assert_allclose(normals[3:], [[0., 1., 0.]]*3)


if __name__ == '__main__':
    unittest.main()
