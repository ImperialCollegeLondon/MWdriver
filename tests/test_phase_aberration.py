import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mwdriver.analysis.phase_aberration import ZERNIKE_MODES, analyze_aberration


class PhaseAberrationTests(unittest.TestCase):
    def make_pupil(self):
        height = width = 64
        yy, xx = np.indices((height, width))
        center_y = center_x = 31.5
        rho = np.hypot(yy - center_y, xx - center_x) / 25.0
        valid = rho <= 1.0
        phase = np.ma.masked_array(
            0.8 * (yy - center_y) / 25.0
            + 0.3 * (xx - center_x) / 25.0
            + 0.15 * rho ** 2,
            mask=~valid,
        )
        return phase, valid

    def test_analysis_returns_zernike_tilt_and_surface_metrics(self):
        phase, valid = self.make_pupil()
        result = analyze_aberration(phase, valid, wavelength_nm=632.8)

        self.assertEqual(result.fit_mask.shape, phase.shape)
        self.assertEqual(result.zernike_coefficients_before_waves.shape, (15,))
        self.assertEqual(result.zernike_coefficients_after_waves.shape, (15,))
        self.assertEqual(result.aberration_component_waves.shape, (6,))
        self.assertEqual(result.tilt_removed_phase.shape, phase.shape)
        self.assertEqual(result.residual_surface_nm.shape, phase.shape)
        self.assertEqual(result.wavefront_aberration_nm.shape, phase.shape)
        np.testing.assert_array_equal(result.fit_mask, valid)
        self.assertTrue(np.isfinite(result.aberration_rms_nm))
        self.assertTrue(np.isfinite(result.aberration_pv_nm))
        self.assertGreaterEqual(result.aberration_pv_nm, 0.0)
        self.assertEqual(len(ZERNIKE_MODES), 15)
        self.assertAlmostEqual(result.pupil_center_x, 31.5)
        self.assertAlmostEqual(result.pupil_center_y, 31.5)

    def test_analysis_rejects_nonmatching_phase_and_mask_shapes(self):
        with self.assertRaisesRegex(ValueError, "matching 2D"):
            analyze_aberration(
                np.zeros((8, 8)),
                np.ones((7, 8), dtype=bool),
                wavelength_nm=632.8,
            )

    def test_analysis_rejects_too_few_valid_pixels(self):
        phase = np.ma.masked_array(np.zeros((8, 8)), mask=True)
        valid = np.zeros((8, 8), dtype=bool)
        with self.assertRaisesRegex(ValueError, "Not enough valid ROI pixels"):
            analyze_aberration(phase, valid, wavelength_nm=632.8)


if __name__ == "__main__":
    unittest.main()
