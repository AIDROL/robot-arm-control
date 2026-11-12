from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from simulation.isaac.check_display_stability import check


class DisplayStabilityTests(unittest.TestCase):
    def captures(self, folder, jitter):
        for number in range(24):
            image = np.full((200, 320, 3), 220, np.uint8)
            shift = jitter if number % 2 else 0
            cv2.rectangle(image, (70+shift, 45), (200+shift, 150), (60, 60, 60), -1)
            cv2.imwrite(str(Path(folder)/f"viewport-{number:03d}.png"), image)

    def test_fixed_scene_passes_but_outline_flicker_fails(self):
        with TemporaryDirectory() as folder:
            self.captures(folder, 0)
            self.assertTrue(check(folder)["success"])
            self.captures(folder, 3)
            self.assertFalse(check(folder)["success"])

    def test_blank_frames_cannot_masquerade_as_a_stable_scene(self):
        with TemporaryDirectory() as folder:
            for number in range(20):
                cv2.imwrite(str(Path(folder)/f"viewport-{number:03d}.png"), np.zeros((100, 100, 3), np.uint8))
            with self.assertRaisesRegex(ValueError, "no scene edges"):
                check(folder)


if __name__ == "__main__":
    unittest.main()
