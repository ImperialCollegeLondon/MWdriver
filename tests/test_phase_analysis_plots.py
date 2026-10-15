import sys
import unittest
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mwdriver.analysis.phase_aberration import ZERNIKE_MODES
from mwdriver.analysis.phase_analysis_plots import render_aberration_plots


class PhaseAnalysisPlotTests(unittest.TestCase):
    def test_render_aberration_plots_returns_labelled_rgb_images(self):
        coefficients_before = np.linspace(-0.2, 0.2, len(ZERNIKE_MODES))
        coefficients_after = coefficients_before * 0.4
        component_waves = np.asarray([0.03, 0.02, 0.01, 0.04, 0.005, 0.008])

        result = render_aberration_plots(
            ZERNIKE_MODES,
            coefficients_before,
            coefficients_after,
            component_waves,
        )

        self.assertEqual(len(result.zernike_bar_labels), len(ZERNIKE_MODES))
        self.assertEqual(result.zernike_bar_labels[0], "J01\nPiston")
        for image in (
            result.zernike_before_rgb,
            result.zernike_after_rgb,
            result.composition_rgb,
        ):
            self.assertEqual(image.ndim, 3)
            self.assertEqual(image.shape[-1], 3)
            self.assertEqual(image.dtype, np.uint8)
            self.assertGreater(np.unique(image.reshape(-1, 3), axis=0).shape[0], 1)


if __name__ == "__main__":
    unittest.main()