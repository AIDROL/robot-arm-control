import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from simulation.michelson_experiment import load_layout,make_scene,latch_base
from simulation.michelson_assembly_optics import trace,from_state,PhaseGradientSearch
from simulation.scene_recording import StateRecorder,draw_paths
from simulation.task_space_control import phases_for_scene


class FullMichelsonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config,cls.targets=load_layout();cls.model=make_scene(cls.config,cls.targets)

    def data(self):
        data=mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model,data,self.model.key('home').id)
        mujoco.mj_forward(self.model,data)
        return data

    def design_data(self):
        data=self.data()
        # A unit-test fixture, not a runtime assembly shortcut.
        for name,T in self.targets.items():
            data.joint(name+'_free').qpos[:3]=T[:3,3]
            data.joint(name+'_free').qpos[3:]=Rotation.from_matrix(T[:3,:3]).as_quat()[[3,0,1,2]]
        mujoco.mj_forward(self.model,data)
        return data

    def test_nonrow_target_controls_transfer_and_orientation(self):
        data=self.data();name='pinhole';observed=np.eye(4);observed[:3,3]=data.body(name).xpos
        phases=phases_for_scene(self.model,data,tuple(self.targets[n][0,3] for n in self.targets),
            object_poses={name:observed},target_poses={name:self.targets[name]},components=tuple((n,None) for n in self.targets))
        lower=next(p for p in phases if p.label==name+': lower')
        np.testing.assert_allclose(lower.position[:2],self.targets[name][:2,3])
        expected=Rotation.from_euler('z',-3*np.pi/4)*Rotation.from_matrix([[0,-1,0],[-1,0,0],[0,0,-1]])
        self.assertLess((lower.orientation*expected.inv()).magnitude(),1e-9)

    def test_pixel_phase_gradient_has_correct_feedback_direction(self):
        obs,_=from_state(self.model,self.design_data(),n=512,config=self.config)
        self.assertTrue(obs['valid'])
        error=obs['relative_tilt_from_pixels_rad']
        self.assertLess(error[0],0);self.assertLess(error[1],0)
        state,command=PhaseGradientSearch().decide(obs)
        self.assertEqual(state,'move');self.assertLess(command[0],0);self.assertGreater(command[1],0)

    def test_other_achieved_module_pose_changes_optics(self):
        data=self.design_data();before,_=from_state(self.model,data,n=128,config=self.config)
        T=self.targets['lens_2'];data.joint('lens_2_free').qpos[3:]=(Rotation.from_euler('z',.01)*Rotation.from_matrix(T[:3,:3])).as_quat()[[3,0,1,2]]
        mujoco.mj_forward(self.model,data);after,_=from_state(self.model,data,n=128,config=self.config)
        self.assertNotEqual(before.get('paths'),after.get('paths'))

    def test_latch_cannot_fix_an_airborne_module(self):
        data=self.data();data.joint('lens_1_free').qpos[2]+=.1;mujoco.mj_forward(self.model,data)
        before=data.qpos.copy()
        with self.assertRaisesRegex(RuntimeError,'SUPPORT'):latch_base(self.model,data,'lens_1')
        np.testing.assert_array_equal(data.qpos,before)

    def test_recording_refuses_hidden_time_reset(self):
        data=self.data();recorder=StateRecorder();recorder.capture(self.model,data,'start')
        data.time=1.;recorder.capture(self.model,data,'later');data.time=0.
        with self.assertRaisesRegex(RuntimeError,'time reset'):recorder.capture(self.model,data,'reset')

    def test_paths_can_be_appended_without_erasing_physical_scene(self):
        scene=mujoco.MjvScene(self.model,maxgeom=20)
        scene.ngeom=1
        paths=[{'points_m':[[0,0,0],[.1,0,0],[.1,.1,0]]}]
        draw_paths(scene,paths,clear=False)
        self.assertEqual(scene.ngeom,3)



class ReplayDetectorTests(unittest.TestCase):
    def test_full_layout_sensor_fields_are_mapped_and_raw_counts_preserved(self):
        import cv2
        from simulation.scene_recording import publish_recorded_detector
        config,_=load_layout()
        with TemporaryDirectory() as temporary:
            root=Path(temporary);counts=np.zeros((32,32),np.uint16);counts[12:20,12:20]=1000
            cv2.imwrite(str(root/'phase-00-detector16.png'),counts)
            obs=publish_recorded_detector(root/'display',config,root,{'visibility':.98},200.)
            actual=cv2.imread(str(root/'display/detector16.png'),cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(actual,counts)
            self.assertEqual(obs['window_m'],config['detector_window_m'])
            self.assertEqual(obs['frame_id'],'michelson_detector_xz')

if __name__=='__main__':unittest.main()
