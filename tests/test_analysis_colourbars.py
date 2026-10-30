import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from napari.components import ViewerModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mwdriver.ui.camera_view_controller import CameraViewController
from mwdriver.ui.phase_analysis_controls import analysis_display_data


class AnalysisColourbarTests(unittest.TestCase):
    def test_display_units_convert_radians_without_changing_stored_data(self):
        phase = np.asarray([[0, np.pi, 2 * np.pi]])
        results = {"wrapped": phase.copy(), "unwrapped": phase.copy(),
                   "tilt_removed": phase.copy(), "wavelength_nm": 600,
                   "residual_nm": np.asarray([[0, 10, 20]])}
        for key in ("wrapped", "unwrapped"):
            values, units = analysis_display_data(results, key)
            np.testing.assert_allclose(values, [[0, 0.5, 1]])
            self.assertEqual(units, "waves")
        values, units = analysis_display_data(results, "tilt_removed")
        np.testing.assert_allclose(values, [[0, 300, 600]])
        self.assertEqual(units, "nm OPD")
        np.testing.assert_array_equal(results["tilt_removed"], phase)

    def test_phase_and_surface_colourbars_follow_analysis_visibility(self):
        viewer = ViewerModel()
        preview = viewer.add_image(np.zeros((8, 8)), name="Preview")
        phase = viewer.add_image(np.arange(64).reshape(8, 8), colormap="twilight")
        surface = viewer.add_surface((
            np.asarray([[0, 0, 0], [1, 0, 1], [2, 1, 0]], dtype=float),
            np.asarray([[0, 1, 2]]), np.asarray([0, 10, 20]),
        ), colormap="viridis", contrast_limits=(0, 20))
        chart = viewer.add_image(np.zeros((8, 8, 3)), rgb=True)
        phase.metadata["colourbar_units"] = "waves"
        surface.metadata["colourbar_units"] = "nm OPD"
        controller = SimpleNamespace(
            viewer=viewer, layer=preview, roi_layer=None,
            phase_analysis_layer_records=[{"layer": layer} for layer in (phase, surface, chart)],
            phase_capture_time_series_layers=[],
        )
        for selected in (phase, surface, chart):
            CameraViewController.apply_layer_visibility_plan(controller, "analysis", selected_layer=selected)
            for layer in (phase, surface, chart):
                self.assertEqual(layer.colorbar.visible, layer is selected and layer is not chart)
            self.assertEqual(viewer.text_overlay.visible, selected is not chart)
            if selected is not chart:
                self.assertEqual(viewer.text_overlay.text, f"Colourbar: {selected.metadata['colourbar_units']}")
        CameraViewController.apply_layer_visibility_plan(controller, "preview")
        self.assertFalse(phase.colorbar.visible)
        self.assertFalse(surface.colorbar.visible)
        self.assertFalse(preview.colorbar.visible)
        self.assertFalse(viewer.text_overlay.visible)


if __name__ == "__main__":
    unittest.main()