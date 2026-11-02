import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from simulation.task_space_control import CartesianController, Phase
from simulation.vision_pick_place import PlacementController


class SettlingTests(unittest.TestCase):
    def controller(self):
        c = PlacementController.__new__(PlacementController)
        c.placement_strategy = 'settled'
        c.phases = [Phase('lens_1: settle', np.zeros(3), Rotation.identity(), .8, 255, False)]
        c.phase_index, c.phase_start_time, c.site, c.table_geom = 0, 0., 0, 1
        c.model = SimpleNamespace(geom=lambda _: SimpleNamespace(id=2))
        c.seating_samples, c.seated_since, c.seating_confirmed = [], None, False
        c.command_q, c.failed = np.zeros(7), None
        return c

    def test_stationary_tcp_without_support_does_not_release_and_times_out(self):
        c = self.controller()
        state = SimpleNamespace(contact=[], ncon=0, time=1.)
        with patch('simulation.vision_pick_place.mujoco.mj_objectVelocity'), \
                patch.object(CartesianController, 'advance') as advance:
            c.advance(state)
            self.assertFalse(c.seating_confirmed)
            advance.assert_not_called()
            state.time = 4.
            c.advance(state)
            self.assertTrue(c.failed.startswith('SEATING_TIMEOUT'))
            advance.assert_not_called()

    def test_support_must_persist_before_opening(self):
        c = self.controller()
        state = SimpleNamespace(contact=[SimpleNamespace(geom1=1, geom2=2)], ncon=1, time=1.)
        def contact_force(model, data, index, out):
            out[0] = .4
        with patch('simulation.vision_pick_place.mujoco.mj_objectVelocity'), \
                patch('simulation.vision_pick_place.mujoco.mj_contactForce', side_effect=contact_force), \
                patch.object(CartesianController, 'advance') as advance:
            c.advance(state)
            advance.assert_not_called()
            state.time = 1.35
            c.advance(state)
            self.assertTrue(c.seating_confirmed)
            advance.assert_called_once()


if __name__ == '__main__':
    unittest.main()
