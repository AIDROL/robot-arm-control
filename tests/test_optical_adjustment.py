import unittest
from simulation.optical_adjustment import PowerSearch


def observation(fraction,stamp=1.,valid=True):
    return {'valid':valid,'simulation_time_s':stamp,'transmitted_fraction':fraction}


class PowerSearchTests(unittest.TestCase):
    def test_power_drop_reverses_and_halves_probe(self):
        p=PowerSearch()
        first=p.decide(observation(.1),1.)
        second=p.decide(observation(.05,2.),2.)
        self.assertEqual(first['shift_world_m'],[0,.0002,0])
        self.assertEqual(second['shift_world_m'],[0,-.0001,0])

    def test_accepted_only_after_new_valid_observation(self):
        p=PowerSearch()
        self.assertEqual(p.decide(observation(.9),2.)['action'],'stop')
        self.assertEqual(p.decide(observation(.9,valid=False),1.)['action'],'stop')
        self.assertEqual(p.decide(observation(.9),1.)['action'],'accept')

    def test_attempts_and_total_travel_are_bounded(self):
        p=PowerSearch(max_actions=2)
        self.assertEqual(p.decide(observation(.1),1.)['action'],'move')
        self.assertEqual(p.decide(observation(.2),1.)['action'],'move')
        self.assertEqual(p.decide(observation(.3),1.)['reason'],'ACTION_LIMIT')
        p=PowerSearch(max_travel_m=.0002)
        p.decide(observation(.1),1.)
        self.assertEqual(p.decide(observation(.2),1.)['reason'],'TRAVEL_LIMIT')

    def test_hidden_truth_fields_cannot_choose_direction(self):
        a=observation(.1);b={**a,'actual_position_error_m':[1,2,3],'optimal_position_m':[0,0,0]}
        self.assertEqual(PowerSearch().decide(a,1.),PowerSearch().decide(b,1.))


if __name__=='__main__':unittest.main()
