"""Check real contact geometry, especially the holes convex hulls can fill."""
import json
import unittest

import mujoco
import numpy as np

from simulation.hollow_mount import HousingDimensions, make_preview, socket_centres, verify


class HollowMountTests(unittest.TestCase):
    def test_aperture_and_grasp_surface(self):
        dim = HousingDimensions()
        model = make_preview(dim).compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        report = verify(model, data, dim)
        json.dumps(report)  # CLI evidence must remain serializable.
        self.assertEqual(report['opening_clear_rays'], 49)
        np.testing.assert_allclose(report['opening_size_mm'], [40., 51.])
        np.testing.assert_allclose(model.site('mount_side_marker').pos, [0, -.0258, .024])
        marker_id = model.site('mount_side_marker').id
        np.testing.assert_allclose(data.site_xmat[marker_id].reshape(3, 3)[:, 2],
                                   [0, -1, 0], atol=1e-12)
        for label, centre in socket_centres(dim).items():
            np.testing.assert_allclose(model.site(f'mount_{label}_screw_center').pos, centre)
            self.assertEqual(model.site(f'mount_{label}_screw_center').bodyid.item(),
                             model.body('mount_kinematic_frame').id)
        self.assertTrue(report['mount_aperture_clear_without_optic'])
        self.assertEqual(report['axis_with_optic']['first_geom'], 'mount_mirror')

    def test_two_mm_probe_enters_socket_but_hits_its_edge(self):
        dim = HousingDimensions()
        spec = make_preview(dim)
        probe = spec.worldbody.add_body(name='probe')
        probe.add_freejoint(name='probe_free')
        probe.add_geom(name='probe_ball', type=mujoco.mjtGeom.mjGEOM_SPHERE,
                       size=[.002], mass=.001)
        model = spec.compile()
        data = mujoco.MjData(model)
        for centre in socket_centres(dim).values():
            for yy in np.linspace(-.030, centre[1]+.0015, 12):
                data.qpos[:3] = [centre[0], yy, centre[2]+dim.height/2]
                data.qpos[3:7] = [1, 0, 0, 0]
                mujoco.mj_forward(model, data)
                self.assertEqual(data.ncon, 0, (yy, centre, data.ncon))
            data.qpos[:3] = [centre[0]+.0025, centre[1], centre[2]+dim.height/2]
            mujoco.mj_forward(model, data)
            self.assertGreater(data.ncon, 0)
            data.qpos[:3] = [centre[0], centre[1]+dim.socket_depth-.0005,
                            centre[2]+dim.height/2]
            mujoco.mj_forward(model, data)
            self.assertGreater(data.ncon, 0, 'Tool must encounter the blind socket bottom')

    def test_empty_mount_passes_axis_but_side_wall_still_blocks(self):
        model = make_preview(optic=False).compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        report = verify(model, data, HousingDimensions())
        self.assertTrue(report['axis_with_optic']['clear'])
        geom_id = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(model, data, np.array([-.1, 0., .064]),
                                 np.array([1., 0., 0.]), np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8),
                                 True, -1, geom_id)
        self.assertGreater(distance, 0)
        self.assertEqual(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id[0])),
                         'mount_left_wall')


if __name__ == '__main__':
    unittest.main()
