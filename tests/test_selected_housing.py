"""Verify actual convex contact geometry, including voids a single hull fills."""
import unittest

import mujoco
import numpy as np

from simulation.selected_housing import add_housing, NOMINAL_MASS


class SelectedHousingTests(unittest.TestCase):
    def probe_model(self):
        spec = mujoco.MjSpec()
        body = spec.worldbody.add_body(name='housing')
        add_housing(spec, body)
        probe = spec.worldbody.add_body(name='probe')
        probe.add_freejoint()
        probe.add_geom(name='probe_ball', type=mujoco.mjtGeom.mjGEOM_SPHERE,
                       size=[.0005], mass=.001)
        model = spec.compile()
        return model, mujoco.MjData(model)

    def test_windows_are_physically_open_in_both_directions(self):
        model, data = self.probe_model()
        data.qpos[3:7] = [1, 0, 0, 0]
        for axis in [0, 1]:
            for offset in np.linspace(-.08, .08, 33):
                p = np.array([0., 0., .04])
                p[axis] = offset
                data.qpos[:3] = p
                mujoco.mj_forward(model, data)
                self.assertEqual(data.ncon, 0, (axis, offset))

    def test_recesses_stay_empty_until_actual_33_mm_surface(self):
        model, data = self.probe_model()
        data.qpos[3:7] = [1, 0, 0, 0]
        for sign in [-1, 1]:
            data.qpos[:3] = [sign*.034, 0, .0881]
            mujoco.mj_forward(model, data)
            self.assertEqual(data.ncon, 0)
            data.qpos[0] = sign*.033
            mujoco.mj_forward(model, data)
            self.assertGreater(data.ncon, 0)

    def test_free_body_falls_and_is_supported_by_floor_contacts(self):
        spec = mujoco.MjSpec()
        spec.worldbody.add_geom(name='table', type=mujoco.mjtGeom.mjGEOM_BOX,
                                pos=[0, 0, -.05], size=[.2, .2, .05], solref=[.005, 1])
        body = spec.worldbody.add_body(name='housing', pos=[0, 0, .04])
        body.add_freejoint()
        add_housing(spec, body)
        model = spec.compile()
        data = mujoco.MjData(model)
        self.assertAlmostEqual(model.body('housing').mass.item(), NOMINAL_MASS)
        self.assertEqual(model.neq, 0)
        self.assertEqual(model.body('housing').mocapid.item(), -1)
        for _ in range(1000):
            mujoco.mj_step(model, data)
        self.assertLess(data.qpos[2], .005)
        self.assertGreater(data.qpos[2], .003)
        self.assertGreater(data.ncon, 0)
        self.assertLess(np.linalg.norm(data.qvel), .01)


if __name__ == '__main__':
    unittest.main()
