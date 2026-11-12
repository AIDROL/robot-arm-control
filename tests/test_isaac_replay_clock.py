import unittest

from simulation.isaac.replay_clock import ReplayClock


class ReplayClockTests(unittest.TestCase):
    def test_speed_change_keeps_source_time_continuous(self):
        clock = ReplayClock(0., 240.8, speed=4., wall=100.)
        self.assertEqual(clock.advance(102.), 8.)
        clock.set_speed(8., wall=103.)
        self.assertEqual(clock.advance(103.), 12.)
        self.assertEqual(clock.advance(104.), 20.)

    def test_pause_does_not_accumulate_wall_time(self):
        clock = ReplayClock(0., 240.8, speed=4., wall=0.)
        clock.set_paused(True, wall=2.)
        self.assertEqual(clock.advance(100.), 8.)
        clock.set_paused(False, wall=101.)
        self.assertEqual(clock.advance(102.), 12.)

    def test_seek_and_restart_do_not_jump_on_next_frame(self):
        clock = ReplayClock(10., 240.8, speed=4., wall=0.)
        clock.seek(170., wall=3.)
        self.assertEqual(clock.advance(4.), 174.)
        clock.restart(wall=5.)
        self.assertEqual(clock.advance(6.), 14.)
        self.assertFalse(clock.paused)

    def test_four_times_finishes_full_source_in_one_minute(self):
        clock = ReplayClock(0., 240.8, speed=4., wall=0.)
        self.assertAlmostEqual(clock.advance(60.2), 240.8)
        self.assertEqual(clock.advance(100.), 240.8)
        clock.seek(-20., wall=101.)
        self.assertEqual(clock.source_time, 0.)

    def test_invalid_speed_is_rejected(self):
        clock = ReplayClock(0., 240.8, wall=0.)
        for speed in (0., -1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                clock.set_speed(speed, wall=0.)


if __name__ == '__main__':
    unittest.main()
