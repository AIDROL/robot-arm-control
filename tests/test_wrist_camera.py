import unittest
from unittest.mock import patch

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from simulation.global_camera import DICTIONARY
from simulation.wrist_camera import refine, E_FROM_D
from simulation.vision_pick_place import WristController
from simulation.task_space_control import CartesianController, Phase


class WristTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.full((400, 480, 3), 255, np.uint8)
        self.rgb[152:248, 192:288] = cv2.aruco.generateImageMarker(DICTIONARY, 0, 96)[:, :, None]
        self.depth = np.full((400, 480), .2)
        self.K = np.array([[800., 0, 239.5], [0, 800., 199.5], [0, 0, 1]])
        self.W = np.diag([1., -1., -1., 1.])
        self.W[2, 3] = 1.0408
        self.coarse = np.eye(4)
        self.coarse[:3, 3] = [.005, -.004, .8]

    def test_depth_corrects_position_and_not_just_xy(self):
        r, _ = refine(self.rgb, self.depth, self.K, self.W, self.coarse, 0, 1.)
        self.assertTrue(r['valid'])
        np.testing.assert_allclose(np.array(r['T_world_object'])[:3, 3], [0, 0, .8], atol=.0001)
        r2, _ = refine(self.rgb, self.depth + .001, self.K, self.W, self.coarse, 0, 1.)
        self.assertTrue(r2['valid'])
        self.assertAlmostEqual(r['T_world_object'][2][3] - r2['T_world_object'][2][3], .001)

    def test_invalid_depth_has_no_fallback(self):
        r, _ = refine(self.rgb, self.depth * np.nan, self.K, self.W, self.coarse, 0, 1.)
        self.assertFalse(r['valid'])
        self.assertEqual(r['reason'], 'INVALID_DEPTH')
        self.assertNotIn('T_world_object', r)

    def test_wrong_marker_is_rejected(self):
        r, _ = refine(self.rgb, self.depth, self.K, self.W, self.coarse, 1, 1.)
        self.assertEqual(r['reason'], 'FINE_MARKER_NOT_DETECTED')

    def test_shared_phase_arrays_are_not_corrected_twice(self):
        controller = WristController.__new__(WristController)
        shared = np.zeros(3)
        controller.phases = [Phase('x: '+label, shared, Rotation.identity(), 1., 0.)
                             for label in ['approach', 'close', 'lift', 'transfer']]
        with patch.object(CartesianController, 'advance'):
            controller.apply_refinement(None, np.array([.001, -.002, .003]))
        for p in controller.phases[:3]:
            np.testing.assert_allclose(p.position, [.001, -.002, .003])
        np.testing.assert_allclose(controller.phases[3].position, [0, 0, 0])

    def test_fk_hand_eye_matches_render_camera_axes(self):
        import mujoco
        from simulation import line_assembly as scene
        model = scene.make_scene(seed=7, include_fat=True, vision=True)
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
        mujoco.mj_forward(model, data)
        tool = data.body('xarm_gripper_base_link')
        W_E = np.eye(4)
        W_E[:3, :3], W_E[:3, 3] = tool.xmat.reshape(3, 3), tool.xpos
        expected = W_E @ E_FROM_D
        cam = model.camera('wrist_rgbd').id
        np.testing.assert_allclose(expected[:3, 3], data.cam_xpos[cam], atol=1e-10)
        np.testing.assert_allclose(expected[:3, :3],
            data.cam_xmat[cam].reshape(3, 3) @ np.diag([1, -1, -1]), atol=1e-10)


if __name__ == '__main__':
    unittest.main()
