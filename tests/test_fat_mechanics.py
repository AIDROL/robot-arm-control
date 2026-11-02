import unittest
import mujoco
import numpy as np
from simulation.fat_mechanics import make_scene, RATIO
from simulation.michelson_optics import evaluate, FringeSearch


class MechanicsTests(unittest.TestCase):
    def test_repeated_bench_construction_keeps_mesh_references(self):
        for index in range(8):
            model=make_scene(michelson=bool(index%2))
            self.assertEqual(model.nu,10)
            self.assertEqual(model.body('fat').mocapid[0],-1)
            self.assertEqual(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'mirror_free'),-1)

    def test_disengaged_motor_cannot_move_mirror(self):
        model=make_scene();data=mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model,data,model.key('home').id)
        for name in ('lower','upper'):
            data.ctrl[model.actuator(name+'_motor').id]=.5
        for _ in range(500):mujoco.mj_step(model,data)
        for name in ('lower','upper'):
            self.assertGreater(abs(data.joint(name+'_motor_joint').qpos[0]),.4)
            self.assertLess(abs(data.joint(name+'_knob_joint').qpos[0]),1e-8)
        self.assertEqual(model.body('fat').mocapid[0],-1)
        self.assertLess(np.linalg.norm([data.joint('mirror_pitch').qpos[0],data.joint('mirror_yaw').qpos[0]]),1e-8)

    def test_screw_lever_responds_to_applied_knob_torque(self):
        model=make_scene();data=mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model,data,model.key('home').id)
        for _ in range(150):
            data.qfrc_applied[model.joint('lower_knob_joint').dofadr[0]]=.00003
            mujoco.mj_step(model,data)
        knob=float(data.joint('lower_knob_joint').qpos[0])
        self.assertGreater(knob,.02)
        self.assertAlmostEqual(float(data.joint('mirror_pitch').qpos[0]),RATIO*knob,delta=2e-5)


class MichelsonTests(unittest.TestCase):
    def test_alignment_increases_integrated_visibility(self):
        aligned,_=evaluate([0,1,0],128)
        normal=np.array([.00035,1,.0003]);normal/=np.linalg.norm(normal)
        tilted,_=evaluate(normal,128)
        self.assertTrue(aligned['valid']);self.assertTrue(tilted['valid'])
        self.assertGreater(aligned['visibility'],.99)
        self.assertLess(tilted['visibility'],.2)
        self.assertLessEqual(aligned['visibility'],1)
        # Balanced ideal splitter: one detector's phase-mean power is half the source.
        self.assertAlmostEqual(aligned['mean_port_power_W'],.0005,delta=2e-7)

    def test_invalid_and_hidden_state_do_not_drive_policy(self):
        search=FringeSearch()
        status,commands=search.decide({'valid':False,'visibility':0})
        self.assertEqual(status,'invalid');np.testing.assert_equal(commands,[0,0])
        a=FringeSearch();b=FringeSearch()
        x=a.decide({'valid':True,'visibility':.1})
        y=b.decide({'valid':True,'visibility':.1,'actual_mirror_error':[5,7]})
        self.assertEqual(x[0],y[0]);np.testing.assert_equal(x[1],y[1])

    def test_rejects_wrong_return_direction(self):
        with self.assertRaises(ValueError):evaluate([1,0,0])


class EvidenceTests(unittest.TestCase):
    def test_phase_power_reconstructs_from_saved_pixels(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        import cv2
        from simulation.michelson_optics import SPEC, SIZE
        with TemporaryDirectory() as temporary:
            folder=Path(temporary)
            obs,_=evaluate([0,1,0],64,capture_folder=folder)
            for i,expected in enumerate(obs['phase_scan_power_W']):
                pixels=cv2.imread(str(folder/f'phase-{i:02d}-detector16.png'),cv2.IMREAD_UNCHANGED)
                self.assertEqual(pixels.dtype,np.uint16)
                reconstructed=pixels.astype(float).sum()*SPEC['detector_full_scale_W_m2']/65535*(SIZE/64)**2
                self.assertAlmostEqual(reconstructed,expected,places=14)

    def test_search_restores_better_command_before_another_probe(self):
        search=FringeSearch()
        _,first=search.decide({'valid':True,'visibility':.4})
        self.assertAlmostEqual(first[0],.1)
        _,second=search.decide({'valid':True,'visibility':.1})
        self.assertAlmostEqual(second[0],0.)
        self.assertGreater(second[1],0.)

    def test_michelson_detector_uses_configured_exposure(self):
        from tempfile import TemporaryDirectory
        from simulation.optical_sensor import publish
        with TemporaryDirectory() as temporary:
            obs=publish(temporary,np.full((16,16),2000.),{'window_m':.006,'input_power_W':1.,'full_scale_W_m2':5000.,'display_axes':'X/Z'},0.)
            self.assertNotIn('SATURATED',obs['reasons'])
            self.assertEqual(obs['full_scale_W_m2'],5000.)
            self.assertIn('centroid_xz_m',obs)
            self.assertNotIn('centroid_yz_m',obs)


if __name__=='__main__':unittest.main()
