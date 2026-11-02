"""Check aperture orientation as well as an unblocked line through the row."""
import unittest

import mujoco
import numpy as np

from simulation import line_assembly as scene
from simulation.selected_housing import add_housing, add_mount_targets
from simulation.beamline_geometry import inspect
from simulation.housing_profile import STL


def fixture(yaw, blocked=False):
    spec=mujoco.MjSpec()
    spec.add_numeric(name='selected_stl_profile',data=[2])
    for index,(name,color) in enumerate(scene.COMPONENTS):
        body=spec.worldbody.add_body(name=name,
                pos=[scene.FAT_TARGET_XS[index],scene.TARGET_Y,scene.TABLE_Z+STL.body_height],
                quat=[np.cos(yaw/2),0,0,np.sin(yaw/2)])
        add_housing(spec,body,name=name,marker_id=index,centered=True,color=color)
        add_mount_targets(spec,body,name)
    if blocked:
        spec.worldbody.add_geom(name='beam_block',type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[.07,scene.TARGET_Y,scene.TABLE_Z+STL.body_height+STL.screw_midpoint[2]],size=[.003,.01,.01])
    model=spec.compile();data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    return model,data


class BeamlineGeometryTests(unittest.TestCase):
    def test_rotated_mounts_share_an_open_axis(self):
        result=inspect(*fixture(np.pi/2))
        self.assertTrue(result['success'])
        for mount in result['mounts'].values():
            self.assertLess(mount['axis_error_deg'],1e-6)

    def test_previous_sideways_mounts_are_rejected(self):
        result=inspect(*fixture(0.))
        self.assertFalse(result['checks']['axes_parallel_to_row'])
        self.assertFalse(result['success'])

    def test_real_geometry_blocks_the_line(self):
        result=inspect(*fixture(np.pi/2,blocked=True))
        self.assertTrue(result['checks']['axes_parallel_to_row'])
        self.assertFalse(result['checks']['finite_line_unobstructed'])
        self.assertEqual(result['first_blocking_geom'],'beam_block')


if __name__=='__main__':unittest.main()
