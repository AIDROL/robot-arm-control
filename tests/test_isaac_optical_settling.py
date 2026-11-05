from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from simulation.isaac_michelson_control import Experiment


class OpticalSettlingTests(unittest.TestCase):
    def test_transient_is_waited_out_before_measurement(self):
        with TemporaryDirectory() as folder:
            experiment = Experiment.__new__(Experiment)
            experiment.sim = SimpleNamespace(time=0., output=Path(folder))
            samples = iter([False, False, True, False, True, True, True])
            def check():
                if not next(samples):
                    raise RuntimeError("UNSETTLED_SCREW_LEVER_FOR_MEASUREMENT")
                return {"pitch": 1e-7, "yaw": 1e-7}
            experiment.check_transmission = check
            experiment.hold = lambda duration: setattr(experiment.sim, "time", experiment.sim.time + duration)
            experiment.await_optical_settling("test")
            self.assertAlmostEqual(experiment.sim.time, .12)
            self.assertTrue((Path(folder) / "test-settling.json").is_file())

    def test_persistent_error_never_passes_the_gate(self):
        experiment = Experiment.__new__(Experiment)
        experiment.sim = SimpleNamespace(time=0.)
        def check():
            raise RuntimeError("UNSETTLED_SCREW_LEVER_FOR_MEASUREMENT")
        experiment.check_transmission = check
        experiment.hold = lambda duration: setattr(experiment.sim, "time", experiment.sim.time + duration)
        with self.assertRaisesRegex(RuntimeError, "OPTICAL_TRANSMISSION_SETTLING_TIMEOUT"):
            experiment.await_optical_settling("test")
        self.assertAlmostEqual(experiment.sim.time, 2.)


if __name__ == "__main__":
    unittest.main()
