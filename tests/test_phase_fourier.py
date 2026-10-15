import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mwdriver.analysis.phase_fourier import detect_fourier_carrier


class FourierCarrierTests(unittest.TestCase):
    def test_detects_synthetic_lower_half_carrier_and_mask_radius(self):
        height = width = 256
        yy, xx = np.indices((height, width))
        frame = np.cos(2.0 * np.pi * (50 * yy / height + 20 * xx / width))
        carrier = detect_fourier_carrier(frame)

        self.assertEqual((carrier.center_y, carrier.center_x), (128, 128))
        self.assertEqual((carrier.peak_y, carrier.peak_x), (178, 148))
        self.assertAlmostEqual(carrier.peak_distance, np.hypot(50, 20))
        self.assertEqual(
            carrier.mask_radius,
            int(np.clip(0.35 * np.hypot(50, 20), 6, 30)),
        )
        self.assertEqual(carrier.spectrum.shape, frame.shape)

    def test_detector_rejects_non_image_input(self):
        with self.assertRaisesRegex(ValueError, "expects a 2D frame"):
            detect_fourier_carrier(np.zeros((2, 3, 4)))


if __name__ == "__main__":
    unittest.main()
