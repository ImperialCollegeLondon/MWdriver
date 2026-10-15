import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import phase_analysis_viewer as legacy_analysis_viewer
from matplotlib.widgets import Button
from mwdriver.analysis.phase_analysis_viewer import show_analysis_result_viewer


class PhaseAnalysisViewerTests(unittest.TestCase):
    def test_legacy_module_reexports_packaged_viewer(self):
        self.assertIs(
            legacy_analysis_viewer.show_analysis_result_viewer,
            show_analysis_result_viewer,
        )

    def test_viewer_buttons_render_image_bar_and_fourier_results(self):
        results = [
            {
                "button": "Image",
                "image": np.arange(64).reshape(8, 8),
                "title": "Image result",
                "cmap": "gray",
                "colorbar": "Intensity",
            },
            {
                "button": "Bar",
                "plot_type": "bar",
                "labels": ["J1", "J2"],
                "values": [0.2, -0.1],
                "title": "Bar result",
                "ylabel": "Waves",
            },
            {
                "button": "Fourier",
                "plot_type": "fourier",
                "spectrum": np.ones((8, 8)),
                "peak_x": 5,
                "peak_y": 6,
                "mask_radius": 2,
                "dc_x": 4,
                "dc_y": 4,
                "title": "Fourier result",
            },
        ]
        callbacks = []

        def record_callback(button, callback):
            callbacks.append(callback)

        try:
            with (
                patch.object(Button, "on_clicked", record_callback),
                patch("mwdriver.analysis.phase_analysis_viewer.plt.show") as show,
            ):
                show_analysis_result_viewer(results)
                self.assertEqual(len(callbacks), len(results))
                for callback in callbacks:
                    callback(None)
                show.assert_called_once_with()
        finally:
            matplotlib.pyplot.close("all")


if __name__ == "__main__":
    unittest.main()