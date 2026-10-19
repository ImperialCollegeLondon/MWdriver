import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from napari.components import ViewerModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mwdriver.ui.camera_roi_controller import CameraRoiController


class CameraRoiControllerTests(unittest.TestCase):
    def test_reload_restores_deleted_or_missing_roi_and_reconnects_events(self):
        for deleted in (True, False):
            with self.subTest(deleted=deleted):
                controller = CameraRoiController()
                controller.viewer = ViewerModel()
                old_layer = controller.viewer.add_shapes(name="ROI")
                controller.viewer.layers.remove(old_layer)
                controller.roi_layer = old_layer if deleted else None
                controller.height = controller.width = 20
                controller.display_downsample = 1
                controller.status_label = MagicMock()
                controller.on_roi_data_changed = MagicMock()
                controller.refresh_roi_mask = MagicMock()
                vertices = np.array([[2, 3], [2, 15], [16, 15], [16, 3]])
                with patch(
                    "mwdriver.ui.camera_roi_controller.load_roi_files",
                    return_value=({}, np.ones((20, 20), dtype=bool), vertices, "ellipse"),
                ):
                    controller.reload_roi()
                self.assertIsNot(controller.roi_layer, old_layer)
                self.assertIn(controller.roi_layer, controller.viewer.layers)
                np.testing.assert_array_equal(controller.roi_layer.data[0], vertices)
                self.assertIs(controller.viewer.layers.selection.active, controller.roi_layer)
                self.assertEqual(controller.roi_layer.mode, "select")
                controller.refresh_roi_mask.assert_called_once()
                controller.on_roi_data_changed.reset_mock()
                controller.roi_layer.events.data()
                controller.on_roi_data_changed.assert_called_once()
                controller.status_label.setText.assert_called_with("Status: Saved ROI reloaded")

    def test_reload_reuses_existing_roi_without_duplicate_connection(self):
        controller = CameraRoiController()
        controller.viewer = ViewerModel()
        existing = controller.viewer.add_shapes(name="ROI")
        controller.roi_layer = existing
        controller.height = controller.width = 20
        controller.display_downsample = 1
        controller.status_label = MagicMock()
        controller.refresh_roi_mask = MagicMock()
        controller.on_roi_data_changed = MagicMock()
        vertices = np.array([[2, 3], [2, 15], [16, 15], [16, 3]])
        with patch(
            "mwdriver.ui.camera_roi_controller.load_roi_files",
            return_value=({}, np.ones((20, 20), dtype=bool), vertices, "ellipse"),
        ):
            controller.reload_roi()
            controller.reload_roi()
        self.assertIs(controller.roi_layer, existing)
        self.assertEqual(len(controller.viewer.layers), 1)