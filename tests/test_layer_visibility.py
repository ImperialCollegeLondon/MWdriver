import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mwdriver.domain.layer_visibility import build_layer_visibility_plan


class LayerVisibilityTests(unittest.TestCase):
    def setUp(self):
        self.analysis_layers = [SimpleNamespace(), SimpleNamespace()]
        self.time_series_layers = [SimpleNamespace(), SimpleNamespace()]

    def test_preview_and_roi_modes_show_camera_and_roi_only(self):
        for mode in ("preview", "roi"):
            with self.subTest(mode=mode):
                plan = build_layer_visibility_plan(
                    mode,
                    self.analysis_layers,
                    self.time_series_layers,
                )
                self.assertTrue(plan.camera_visible)
                self.assertTrue(plan.roi_visible)
                self.assertEqual(plan.analysis_layers_visible, (False, False))
                self.assertEqual(
                    plan.time_series_layers_visible,
                    (False, False),
                )

    def test_analysis_mode_shows_only_selected_analysis_layer(self):
        plan = build_layer_visibility_plan(
            "analysis",
            self.analysis_layers,
            self.time_series_layers,
            selected_layer=self.analysis_layers[1],
        )

        self.assertFalse(plan.camera_visible)
        self.assertFalse(plan.roi_visible)
        self.assertEqual(plan.analysis_layers_visible, (False, True))
        self.assertEqual(plan.time_series_layers_visible, (False, False))

    def test_time_series_mode_shows_only_selected_series_and_roi(self):
        plan = build_layer_visibility_plan(
            "time_series",
            self.analysis_layers,
            self.time_series_layers,
            selected_layer=self.time_series_layers[0],
        )

        self.assertFalse(plan.camera_visible)
        self.assertTrue(plan.roi_visible)
        self.assertEqual(plan.analysis_layers_visible, (False, False))
        self.assertEqual(plan.time_series_layers_visible, (True, False))

    def test_unknown_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown layer visibility mode"):
            build_layer_visibility_plan(
                "unknown",
                self.analysis_layers,
                self.time_series_layers,
            )


if __name__ == "__main__":
    unittest.main()
