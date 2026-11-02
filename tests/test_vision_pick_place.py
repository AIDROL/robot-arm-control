"""Checks that visual planning works without object ground-truth fields."""
import unittest
from types import SimpleNamespace

import numpy as np

from simulation import line_assembly as scene
from simulation.global_camera import select_objects
from simulation.task_space_control import phases_for_scene
from simulation.vision_pick_place import require_pose
from simulation.housing_profile import PROXY, STL


class VisualPlanningTests(unittest.TestCase):
    def observation(self):
        pose = np.eye(4)
        pose[:3, 3] = [0.2, 0.5, scene.BLOCK_Z]
        return {"valid": True, "frame_id": "world", "timestamp": 0.,
                "T_world_object": pose.tolist()}

    def test_plan_uses_estimate_without_object_state(self):
        # No body(), xpos or xmat fields exist: a truth-based planner would fail.
        model = SimpleNamespace(site=lambda _: SimpleNamespace(id=0))
        data = SimpleNamespace(site_xpos=np.array([[0., 0., 1.2]]),
                               site_xmat=np.eye(3).reshape(1, 9))
        T = require_pose(self.observation(), 0.)
        phases = phases_for_scene(model, data, scene.FAT_TARGET_XS, {"lens_1": T}, geometry=PROXY)
        grasp = next(p for p in phases if p.label == 'lens_1: approach')
        np.testing.assert_allclose(grasp.position, [0.2, 0.5, scene.BLOCK_Z + scene.GRASP_Z_OFFSET])
        self.assertEqual(len(phases), 9)
        self.assertFalse(any('lens_2' in p.label for p in phases))

    def test_stl_plan_uses_recess_height_and_x_closure_without_truth(self):
        model = SimpleNamespace(site=lambda _: SimpleNamespace(id=0))
        data = SimpleNamespace(site_xpos=np.array([[0., 0., 1.2]]), site_xmat=np.eye(3).reshape(1, 9))
        observation = self.observation()
        observation['housing_geometry'] = STL.id
        observation['T_world_object'][2][3] = scene.TABLE_Z+STL.body_height
        pose = require_pose(observation, 0.)
        plan = phases_for_scene(model, data, scene.FAT_TARGET_XS, {'lens_1': pose}, geometry=STL)
        grasp = next(p for p in plan if p.label == 'lens_1: close')
        np.testing.assert_allclose(grasp.position, [.2, .5, .9121])
        np.testing.assert_allclose(grasp.orientation.as_matrix(), STL.gripper_reference.as_matrix())
        self.assertEqual(grasp.grip, 100.)
        lifted = next(p for p in plan if p.label == 'lens_1: lift')
        carried_bottom = lifted.position[2]-STL.grasp_z-STL.body_height
        stationary_roof = scene.TABLE_Z+STL.body_height+STL.top_z
        self.assertGreater(carried_bottom-stationary_roof, .03)
        # Only final placement turns; the initial observed grasp stays unchanged.
        placed = next(p for p in plan if p.label == 'lens_1: lower')
        from scipy.spatial.transform import Rotation
        expected = Rotation.from_euler('z', np.pi/2)*STL.gripper_reference
        np.testing.assert_allclose(placed.orientation.as_matrix(), expected.as_matrix(), atol=1e-12)
        # A biased actual return-to-home orientation must not redefine calibration.
        data.site_xmat = (Rotation.from_euler('z', .04)*STL.gripper_reference).as_matrix().reshape(1,9)
        repeated = phases_for_scene(model,data,scene.FAT_TARGET_XS,{'lens_1':pose},geometry=STL)
        lower = next(p for p in repeated if p.label == 'lens_1: lower')
        np.testing.assert_allclose(lower.orientation.as_matrix(),expected.as_matrix(),atol=1e-12)
        del observation['housing_geometry']
        with self.assertRaisesRegex(ValueError, 'POSE_OUTSIDE_TABLETOP_PRIOR'):
            require_pose(observation, 0.)

    def test_detection_failure_is_rejected_before_planning(self):
        objects = select_objects({"a": {"estimates": [], "simulation_time": 0.}})
        with self.assertRaisesRegex(ValueError, 'MARKER_NOT_DETECTED'):
            require_pose(objects['lens_1'], 0.)

    def test_camera_conflict_cannot_be_used_as_control_target(self):
        a, b = np.eye(4), np.eye(4)
        b[0, 3] = 0.03
        cameras = {name: {"simulation_time": 0., "estimates": [
            {"id": 0, "valid": True, "T_world_marker": T.tolist(),
             "reprojection_error_px": 0.1}]}
            for name, T in [('a', a), ('b', b)]}
        with self.assertRaisesRegex(ValueError, 'CAMERA_CONFLICT'):
            require_pose(select_objects(cameras)['lens_1'], 0.)

    def test_wrong_frame_stale_and_nonfinite_messages_are_rejected(self):
        for key, value, reason in [('frame_id', 'camera', 'WRONG_POSE_FRAME'),
                                   ('timestamp', -1., 'STALE_OBSERVATION'),
                                   ('T_world_object', [[float('nan')] * 4] * 4, 'INVALID_TRANSFORM')]:
            with self.subTest(key=key):
                message = self.observation()
                message[key] = value
                with self.assertRaisesRegex(ValueError, reason):
                    require_pose(message, 0.)


if __name__ == '__main__':
    unittest.main()
