"""Check that the native-physics pose boundary preserves the optical plant."""
import unittest
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from simulation.michelson_experiment import load_layout, make_scene
from simulation.michelson_assembly_optics import ROLES, from_state, from_poses


class OpticalPoseBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, targets = load_layout()
        cls.model = make_scene(cls.config, targets)
        cls.data = mujoco.MjData(cls.model)
        for name, T in targets.items():
            cls.data.joint(name + '_free').qpos[:3] = T[:3, 3]
            cls.data.joint(name + '_free').qpos[3:] = Rotation.from_matrix(T[:3, :3]).as_quat()[[3, 0, 1, 2]]
        mujoco.mj_kinematics(cls.model, cls.data)

    def poses(self):
        result = {}
        for role, name in ROLES.items():
            site = self.data.site(name + '_optical_axis')
            T = np.eye(4)
            T[:3, :3] = site.xmat.reshape(3, 3)
            T[:3, 3] = site.xpos
            result[role] = T
        return result

    def test_pose_boundary_preserves_pixels_and_measurements(self):
        before, image_before = from_state(self.model, self.data, n=128, config=self.config)
        after, image_after = from_poses(self.poses(), n=128, config=self.config)
        self.assertEqual(before, after)
        np.testing.assert_array_equal(image_before, image_after)

    def test_missing_or_nonrigid_pose_is_rejected(self):
        poses = self.poses()
        del poses['laser']
        with self.assertRaisesRegex(ValueError, 'ROLE_SET'):
            from_poses(poses, config=self.config)
        poses = self.poses()
        poses['laser'][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'INVALID_OPTICAL_PLANE'):
            from_poses(poses, config=self.config)

    def test_real_geometry_change_changes_the_detector(self):
        poses = self.poses()
        before, _ = from_poses(poses, n=128, config=self.config)
        poses['reference_mirror'][:3, 3] += [0., .05, 0.]
        after, _ = from_poses(poses, n=128, config=self.config)
        self.assertTrue(before['valid'])
        self.assertFalse(after['valid'])
        self.assertEqual(after['reason'], 'CHIEF_RAY_MISSED_CLEAR_APERTURE')


if __name__ == '__main__':
    unittest.main()
