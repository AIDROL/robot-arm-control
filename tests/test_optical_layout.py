import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import mujoco
import numpy as np

from simulation import line_assembly as scene
from simulation.housing_profile import STL, STL_FRAME, for_model
from simulation.optical_layout import load
from simulation.selected_housing import add_housing, add_mount_targets, add_post_base

LAYOUT = Path(__file__).resolve().parents[1]/'configs/optical_setups/three_mounts.json'


class OpticalLayoutTests(unittest.TestCase):
    def test_optical_centers_are_not_used_as_body_or_tcp_coordinates(self):
        plan = load(LAYOUT)
        np.testing.assert_allclose(plan.target_xs, [-.176, .004, .184])
        row = plan.report['components'][0]
        self.assertAlmostEqual(row['optical_center_world_m'][2], .860)
        self.assertAlmostEqual(row['body_origin_world_m'][2], .874)
        self.assertAlmostEqual(row['grasp_center_world_m'][2], .9121)
        self.assertEqual(plan.report['optical_validation'], 'not_run')

    def test_incompatible_units_height_spacing_and_axis_are_rejected(self):
        for mutation in ('unit', 'height', 'spacing', 'axis'):
            doc = json.loads(LAYOUT.read_text())
            if mutation == 'unit': doc['length_unit'] = 'mm'
            if mutation == 'height': doc['components'][0]['optical_center_m'][2] = .040
            if mutation == 'spacing': doc['components'][1]['optical_center_m'][0] = -.130
            if mutation == 'axis': doc['components'][0]['direction_xyz'] = [1, 0, 0]
            with self.subTest(mutation=mutation), TemporaryDirectory() as tmp:
                path=Path(tmp)/'layout.json';path.write_text(json.dumps(doc))
                with self.assertRaises(ValueError): load(path)

    def test_supported_layout_is_bound_to_three_component_scene(self):
        plan = load(LAYOUT)
        model = scene.make_scene(seed=7, vision=True, include_fat=True,
                                 target_xs=plan.target_xs, component_names=plan.names,
                                 layout_sha256=plan.sha256)
        self.assertEqual(tuple(n for n,_ in scene.components_for_model(model)), plan.names)
        self.assertIn(plan.sha256, scene.target_layout_id(model))
        self.assertEqual(for_model(model), STL)
        self.assertNotEqual(STL.id, STL_FRAME.id)
        data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        for name in plan.names:
            self.assertAlmostEqual(data.site(name+'_optical_axis').xpos[2]-scene.TABLE_Z, .100)
            self.assertAlmostEqual(data.site(name+'_support_bottom').xpos[2], scene.TABLE_Z)
            self.assertIn(model.geom(name+'_magnetic_base').id,scene.component_collision_geoms(model,name))

    def test_changed_layout_changes_motion_targets_and_checkpoint_identity(self):
        original=load(LAYOUT)
        doc=json.loads(LAYOUT.read_text())
        for component in doc['components']:
            component['optical_center_m'][0] += .020
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'changed.json';path.write_text(json.dumps(doc))
            changed=load(path)
        np.testing.assert_allclose(np.array(changed.target_xs)-original.target_xs,.020)
        self.assertNotEqual(changed.sha256,original.sha256)

    def test_post_and_base_are_carried_free_body_geometry_and_support_weight(self):
        spec=mujoco.MjSpec()
        spec.worldbody.add_geom(name='table',type=mujoco.mjtGeom.mjGEOM_BOX,
                                pos=[0,0,-.05],size=[.2,.2,.05],solref=[.005,1])
        body=spec.worldbody.add_body(name='housing',pos=[0,0,STL.body_height+.005])
        body.add_freejoint();add_housing(spec,body,name='housing',centered=True)
        add_post_base(body,'housing',STL);add_mount_targets(spec,body,'housing')
        model=spec.compile();data=mujoco.MjData(model)
        for _ in range(1000):mujoco.mj_step(model,data)
        mujoco.mj_forward(model,data)
        self.assertEqual(model.neq,0)
        self.assertEqual(model.nmocap,0)
        self.assertAlmostEqual(float(data.site('housing_optical_axis').xpos[2]),.100,delta=.0005)
        self.assertLess(np.linalg.norm(data.qvel),.01)
        base=model.geom('housing_magnetic_base').id
        self.assertTrue(any(base in (c.geom1,c.geom2) for c in data.contact[:data.ncon]))


if __name__=='__main__':unittest.main()
