import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from simulation.timing import SimulationPacer,positive_speed


class TimingTests(unittest.TestCase):
    def test_four_times_speed_scales_wall_deadline(self):
        with patch('simulation.timing.time.monotonic',side_effect=[10.,10.1]),patch('simulation.timing.time.sleep') as sleep:
            SimulationPacer(100.,speed=4).wait(102.)
            self.assertAlmostEqual(sleep.call_args.args[0],.4)

    def test_slow_motion_scales_wall_deadline(self):
        with patch('simulation.timing.time.monotonic',side_effect=[10.,10.1]),patch('simulation.timing.time.sleep') as sleep:
            SimulationPacer(100.,speed=.5).wait(101.)
            self.assertAlmostEqual(sleep.call_args.args[0],1.9)

    def test_late_rendering_and_headless_execution_do_not_wait(self):
        with patch('simulation.timing.time.monotonic',side_effect=[10.,11.,10.]),patch('simulation.timing.time.sleep') as sleep:
            SimulationPacer(100.,speed=4).wait(102.)
            SimulationPacer(100.,speed=4,enabled=False).wait(102.)
            sleep.assert_not_called()

    def test_rejects_nonfinite_and_nonpositive_speed(self):
        for value in (0,-1,float('nan'),float('inf')):
            with self.subTest(value=value),self.assertRaises(ValueError):positive_speed(value)

    def test_wall_speed_does_not_change_physical_trajectory(self):
        model=mujoco.MjModel.from_xml_string('''<mujoco><option timestep="0.001"/>
          <worldbody><body pos="0 0 1"><joint type="hinge" axis="0 1 0"/>
          <geom type="capsule" size=".02" fromto="0 0 0 0 0 -.3" mass=".1"/>
          </body></worldbody></mujoco>''')
        results=[]
        with patch('simulation.timing.time.monotonic',return_value=0),patch('simulation.timing.time.sleep'):
            for speed in (.5,4.):
                data=mujoco.MjData(model);data.qpos[0]=.4
                pacer=SimulationPacer(data.time,speed)
                for _ in range(100):mujoco.mj_step(model,data);pacer.wait(data.time)
                results.append((data.qpos.copy(),data.qvel.copy(),data.time))
        np.testing.assert_array_equal(results[0][0],results[1][0])
        np.testing.assert_array_equal(results[0][1],results[1][1])
        self.assertEqual(results[0][2],results[1][2])
        self.assertAlmostEqual(model.opt.timestep,.001)


if __name__=='__main__':unittest.main()
