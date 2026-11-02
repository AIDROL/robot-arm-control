import unittest
from dataclasses import replace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from simulation import line_assembly as scene
from simulation.task_space_control import CartesianController, Phase


class PrecisionControlTests(unittest.TestCase):
    def test_submillimetre_error_blocks_only_the_strict_phase(self):
        model = scene.make_scene(seed=7)
        for tolerance, expected_index in [(.005, 1), (.0003, 0)]:
            with self.subTest(tolerance=tolerance):
                data = mujoco.MjData(model)
                mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
                mujoco.mj_forward(model, data)
                controller = CartesianController(model, data, object_poses={})
                measured = data.site_xpos[controller.site].copy()
                orientation = Rotation.from_matrix(data.site_xmat[controller.site].reshape(3, 3))
                # A measured 0.8 mm residual used to pass the 5 mm threshold.
                target = measured + [.0008, 0, 0]
                phase = Phase('lens_1: approach', target, orientation, 1., 0.,
                              position_tolerance=tolerance)
                controller.phases = [phase, replace(phase, label='lens_1: close')]
                controller.phase_start_time = -2.
                controller(model, data)
                self.assertEqual(controller.phase_index, expected_index)
                self.assertIsNone(controller.failed)


if __name__ == '__main__':
    unittest.main()
