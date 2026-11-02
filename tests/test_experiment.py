import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import mujoco
import numpy as np

from simulation.fat_vision import locate
from simulation.global_camera import DICTIONARY
from simulation import line_assembly as scene
from simulation.vision_pick_place import run
from simulation.experiment import run as experiment_run
from simulation.housing_profile import STL, for_model
from simulation.fat_alignment import FatAlignmentController, INSERT_STANDOFF


class ExperimentTests(unittest.TestCase):
    def test_selected_scene_and_fat_targets_share_calibration(self):
        model=scene.make_scene(seed=7,include_fat=True,vision=True)
        data=mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model,data,model.key('home').id)
        mujoco.mj_forward(model,data)
        self.assertEqual(for_model(model).id,STL.id)
        poses={}
        for i,(name,_) in enumerate(scene.COMPONENTS):
            np.testing.assert_allclose(model.site(f'{name}_side_marker').pos,STL.side_center)
            np.testing.assert_allclose(model.site(f'{name}_grasp_center').pos,[0,0,STL.grasp_z])
            self.assertGreater(len(scene.component_collision_geoms(model,name)),77)
            pose=np.eye(4);pose[:3,3]=[scene.FAT_TARGET_XS[i],scene.TARGET_Y,scene.TABLE_Z+STL.body_height]
            poses[name]=pose
        controller=FatAlignmentController(model,data,object_poses=poses,assembly_complete=True,visual_alignment=True)
        controller._alignment_phases(data)
        for name,_ in scene.COMPONENTS:
            target=poses[name][:3,3]+STL.screw_midpoint-INSERT_STANDOFF*np.array(STL.insert_axis)
            np.testing.assert_allclose(controller.targets[name][0],target)

    def test_side_marker_pose_from_pixels_without_object_truth(self):
        rgb=np.full((400,480,3),255,np.uint8)
        rgb[152:248,192:288]=cv2.aruco.generateImageMarker(DICTIONARY,10,96)[:,:,None]
        K=np.array([[640.,0,239.5],[0,640.,199.5],[0,0,1.]])
        coarse=np.diag([1.,-1.,-1.,1.]);coarse[2,3]=.08
        result,_=locate(rgb,K,np.eye(4),10,coarse)
        self.assertTrue(result['valid'])
        np.testing.assert_allclose(np.array(result['T_world_marker'])[:3,3],[0,0,.08],atol=.001)

    def test_fat_missing_marker_has_no_pose_fallback(self):
        result,_=locate(np.full((400,480,3),255,np.uint8),np.eye(3),np.eye(4),10,np.eye(4))
        self.assertFalse(result['valid'])
        self.assertEqual(result['reason'],'FAT_MARKER_NOT_DETECTED')
        self.assertNotIn('T_world_marker',result)

    def test_side_tabletop_fit_uses_pixels_and_keeps_configured_orientation(self):
        rgb=np.full((400,480,3),255,np.uint8)
        rgb[152:248,192:288]=cv2.aruco.generateImageMarker(DICTIONARY,10,96)[:,:,None]
        K=np.array([[640.,0,239.5],[0,640.,199.5],[0,0,1.]])
        coarse=np.diag([1.,-1.,-1.,1.]);coarse[:3,3]=[.001,-.002,.08]
        result,_=locate(rgb,K,np.eye(4),10,coarse,tabletop=True)
        self.assertTrue(result['valid'])
        pose=np.array(result['T_world_marker'])
        np.testing.assert_allclose(pose[:3,3],[0,0,.08],atol=.0001)
        np.testing.assert_array_equal(pose[:3,:3],coarse[:3,:3])

    def test_old_proxy_checkpoint_is_rejected_for_selected_stl(self):
        with TemporaryDirectory() as tmp:
            checkpoint=Path(tmp)/'old.npz'
            np.savez(checkpoint,seed=7,state=np.zeros(1))
            with self.assertRaisesRegex(ValueError,'housing geometry mismatch'):
                experiment_run(7,Path(tmp)/'run',resume_assembly=checkpoint)

    def test_unrotated_stl_checkpoint_is_rejected_for_new_layout(self):
        with TemporaryDirectory() as tmp:
            checkpoint=Path(tmp)/'old-stl.npz'
            np.savez(checkpoint,seed=7,state=np.zeros(1),housing_geometry=STL.id)
            with self.assertRaisesRegex(ValueError,'target layout mismatch'):
                experiment_run(7,Path(tmp)/'run',resume_assembly=checkpoint)

    def test_shared_scene_is_not_reset_when_observation_fails(self):
        model=scene.make_scene(seed=7,include_fat=True,vision=True)
        data=mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model,data,model.key('home').id)
        data.time=123.4
        mujoco.mj_forward(model,data)
        before=data.qpos.copy()
        with TemporaryDirectory() as tmp, patch('simulation.global_camera.capture_observations',return_value={}):
            report=run(7,'lens_1',Path(tmp)/'run',model=model,data=data)
        self.assertFalse(report['control_started'])
        self.assertEqual(report['scene_start_time'],123.4)
        self.assertEqual(data.time,123.4)
        np.testing.assert_array_equal(data.qpos,before)


if __name__=='__main__':unittest.main()
