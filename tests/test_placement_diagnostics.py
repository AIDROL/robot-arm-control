import unittest

import numpy as np

from simulation.placement_diagnostics import decompose


class PlacementDiagnosticTests(unittest.TestCase):
    def test_signed_components_and_release_motion(self):
        release = {'tcp': np.array([.003, 0, .01]), 'rotation': np.eye(3),
                   'object': np.array([.004, .002, 0])}
        result = decompose(np.array([.001, 0, -.01]), np.array([0, 0, -.01]),
                           release, np.array([.003, .004, 0]),
                           np.array([0, 0, .01]), np.eye(3), np.zeros(3))
        terms = result['terms']
        np.testing.assert_allclose(terms['tcp_translation']['world_xyz_m'], [.003, 0, 0])
        np.testing.assert_allclose(terms['grasp_registration']['world_xyz_m'], [.001, 0, 0])
        np.testing.assert_allclose(terms['held_relative_change']['world_xyz_m'], [0, .002, 0])
        np.testing.assert_allclose(terms['post_release_motion']['world_xyz_m'], [-.001, .002, 0])
        self.assertAlmostEqual(result['final_error']['xy_m'], .005)
        self.assertLess(result['closure_residual_m'], 1e-12)

    def test_tool_rotation_lever_is_separate_from_slip(self):
        # A 10 mm lever rotated 90 degrees about Z moves from (10,0) to (0,10).
        R = np.array([[0., -1, 0], [1., 0, 0], [0, 0, 1.]])
        release = {'tcp': np.array([-.01, 0, 0]), 'rotation': R,
                   'object': np.array([-.01, .01, 0])}
        result = decompose(np.array([.01, 0, 0]), np.array([.01, 0, 0]), release,
                           np.array([-.01, .01, 0]), np.array([-.01, 0, 0]), np.eye(3), np.zeros(3))
        np.testing.assert_allclose(result['terms']['tcp_rotation_lever']['world_xyz_m'], [-.01, .01, 0])
        self.assertAlmostEqual(result['terms']['held_relative_change']['norm_m'], 0)
        self.assertLess(result['closure_residual_m'], 1e-12)


if __name__ == '__main__':
    unittest.main()
