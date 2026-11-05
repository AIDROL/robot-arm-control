import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from simulation.isaac_michelson_audit import measure_scan


class DetectorAuditTests(unittest.TestCase):
    def test_quantization_overshoot_is_retained_separately_from_physical_metric(self):
        config = {"reference_phase_samples": 8, "detector_full_scale_W_m2": 100.,
                  "detector_window_m": .01, "input_power_W": .001}
        with tempfile.TemporaryDirectory() as directory:
            for i in range(8):
                counts = round(6553.5 * (1 + np.cos(2 * np.pi * i / 8)))
                cv2.imwrite(str(Path(directory) / f"phase-{i:02d}-detector16.png"),
                            np.full((32, 32), counts, np.uint16))
            report = measure_scan(directory, config)
            self.assertGreater(report["raw_visibility_fit"], 1.)
            self.assertEqual(report["visibility"], 1.)
            self.assertTrue(report["valid"])

    def test_recovers_known_modulation_from_quantized_phase_frames(self):
        config = {"reference_phase_samples": 8, "detector_full_scale_W_m2": 100.,
                  "detector_window_m": .01, "input_power_W": .001}
        with tempfile.TemporaryDirectory() as directory:
            for i in range(8):
                intensity = 10. * (1 + .97 * np.cos(2 * np.pi * i / 8 + .37))
                frame = np.full((32, 32), round(intensity / 100 * 65535), np.uint16)
                cv2.imwrite(str(Path(directory) / f"phase-{i:02d}-detector16.png"), frame)
            report = measure_scan(directory, config)
            self.assertTrue(report["valid"])
            self.assertAlmostEqual(report["visibility"], .97, delta=1e-4)
            self.assertAlmostEqual(report["mean_port_power_W"], .001, delta=1e-7)

    def test_missing_or_eight_bit_frames_cannot_pass(self):
        config = {"reference_phase_samples": 8}
        with tempfile.TemporaryDirectory() as directory:
            cv2.imwrite(str(Path(directory) / "phase-00-detector16.png"), np.ones((32, 32), np.uint8))
            with self.assertRaisesRegex(ValueError, "INVALID_RAW_16BIT"):
                measure_scan(directory, config)


if __name__ == "__main__":
    unittest.main()
