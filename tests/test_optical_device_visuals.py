import importlib.util
from pathlib import Path
import unittest

import numpy as np

from simulation.mujoco.michelson_experiment import load_layout, make_scene


class OpticalDeviceVisualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.targets = load_layout()
        cls.model = make_scene(cls.config, cls.targets)

    def test_laser_has_a_body_and_front_facing_aperture(self):
        model = self.model
        barrel = model.geom('lens_1_laser_barrel').id
        aperture = model.geom('lens_1_laser_aperture').id
        self.assertGreater(model.geom_pos[aperture, 1], model.geom_pos[barrel, 1])
        self.assertAlmostEqual(model.geom_pos[aperture, 1] + model.geom_size[aperture, 1], .004)
        self.assertAlmostEqual(model.geom_pos[aperture, 2], -.014)

    def test_screen_matches_the_calibrated_optical_window_and_plane(self):
        model = self.model
        screen = model.geom('beam_splitter_detector_screen').id
        self.assertAlmostEqual(model.geom_size[screen, 0]*2, self.config['detector_window_m'])
        self.assertAlmostEqual(model.geom_size[screen, 2]*2, self.config['detector_window_m'])
        self.assertAlmostEqual(model.geom_pos[screen, 1]+model.geom_size[screen, 1], .004)
        self.assertAlmostEqual(model.geom_pos[screen, 2], -.014)
        self.assertGreater(model.geom_rgba[screen, :3].min(), .9)

    def test_added_visuals_preserve_recorded_source_collision_and_inertia(self):
        root = Path(__file__).resolve().parents[1]
        source = root/'runs/isaac-michelson-final-20261103-006/source/michelson_experiment.py'
        spec = importlib.util.spec_from_file_location('simulation.mujoco._recorded_layout', source)
        archived = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(archived)
        old = archived.make_scene(self.config, self.targets)
        model = self.model
        self.assertEqual(model.nbody, old.nbody)
        self.assertEqual(model.njnt, old.njnt)
        for field in ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat'):
            np.testing.assert_array_equal(getattr(model, field), getattr(old, field))
        before = (old.geom_contype != 0) | (old.geom_conaffinity != 0)
        after = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        for field in ('geom_type', 'geom_bodyid', 'geom_pos', 'geom_quat', 'geom_size', 'geom_friction', 'geom_condim'):
            np.testing.assert_array_equal(getattr(model, field)[after], getattr(old, field)[before])
        for number in range(model.ngeom):
            name = model.geom(number).name
            if name and ('laser_' in name or 'detector_' in name):
                self.assertEqual(model.geom_contype[number], 0)
                self.assertEqual(model.geom_conaffinity[number], 0)


if __name__ == '__main__':
    unittest.main()
