import sys
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from napari.components import ViewerModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mwdriver.ui.camera_phase_analysis_controller as analysis_controller
from mwdriver.ui.camera_phase_analysis_controller import CameraPhaseAnalysisController
from mwdriver.capture.phase_capture_storage import PhaseSaveResult
from mwdriver.ui.camera_roi_controller import CameraRoiController
from mwdriver.domain.roi_geometry import enclosing_circle_vertices


class CameraPhaseAnalysisControllerTests(unittest.TestCase):
    def test_automatic_circle_populates_roi_and_is_used_by_subsequent_analysis(self):
        class Controller(CameraPhaseAnalysisController, CameraRoiController):
            pass

        controller = Controller()
        controller.viewer = ViewerModel()
        controller.roi_layer = None
        controller.height, controller.width = 60, 80
        controller.display_downsample = 2
        controller.on_roi_data_changed = MagicMock()
        valid = np.zeros((60, 80), dtype=bool)
        valid[20:40, 25:55] = True
        vertices = enclosing_circle_vertices(valid)
        controller.phase_analysis_results = {
            "automatic_roi_vertices": vertices,
            "roi_mask": valid,
        }
        self.assertTrue(controller.apply_automatic_analysis_roi())
        np.testing.assert_allclose(controller.roi_layer.data[0], vertices / 2)
        self.assertEqual(controller.roi_layer.shape_type, ["ellipse"])
        self.assertEqual(controller.roi_layer.mode, "select")
        mask, _, info = controller.get_current_inside_mask()
        self.assertTrue(np.all(mask[valid]))
        np.testing.assert_allclose(info["raw_coordinates"]["vertices"], vertices)
        self.assertTrue(controller.display_mask_enabled)
        self.assertFalse(controller.apply_automatic_analysis_roi())
        self.assertEqual(len(controller.roi_layer.data), 1)

    def test_automatic_roi_does_not_replace_user_roi(self):
        existing = np.asarray([[1, 1], [1, 9], [9, 9], [9, 1]])
        viewer = SimpleNamespace(
            phase_analysis_results={"automatic_roi_vertices": existing * 2},
            roi_layer=SimpleNamespace(data=[existing]),
            get_current_inside_mask=lambda: (np.ones((10, 10), dtype=bool), "ellipse", {}),
        )
        self.assertFalse(CameraPhaseAnalysisController.apply_automatic_analysis_roi(viewer))
        np.testing.assert_array_equal(viewer.roi_layer.data[0], existing)

    def test_latest_successful_save_is_preferred_when_its_stack_exists(self):
        with tempfile.TemporaryDirectory() as save_root:
            session_dir = Path(save_root) / "phase_capture_latest"
            session_dir.mkdir()
            raw_stack_path = session_dir / "phase_capture_latest_raw_stack.npy"
            raw_stack_path.touch()
            viewer = SimpleNamespace(
                latest_phase_save_result=PhaseSaveResult(
                    save_success=True,
                    session_dir=str(session_dir),
                    raw_stack_path=str(raw_stack_path),
                    metadata_path=None,
                )
            )

            with patch.object(analysis_controller, "SAVE_DIR", save_root):
                selected = CameraPhaseAnalysisController.find_latest_phase_analysis_session(
                    viewer
                )

        self.assertEqual(selected, str(session_dir.resolve()))

    def test_fallback_selects_newest_persisted_session_with_raw_stack(self):
        with tempfile.TemporaryDirectory() as save_root:
            older = Path(save_root) / "phase_capture_older"
            newer = Path(save_root) / "phase_capture_newer"
            incomplete = Path(save_root) / "phase_capture_incomplete"
            older.mkdir()
            newer.mkdir()
            incomplete.mkdir()
            older_stack = older / "phase_capture_older_raw_stack.npy"
            newer_stack = newer / "phase_capture_newer_raw_stack.npy"
            older_stack.touch()
            newer_stack.touch()
            (incomplete / "metadata.json").touch()
            older_time = 1_000_000_000
            newer_time = 1_000_000_100
            os.utime(older_stack, (older_time, older_time))
            os.utime(newer_stack, (newer_time, newer_time))
            viewer = SimpleNamespace(latest_phase_save_result=None)

            with patch.object(analysis_controller, "SAVE_DIR", save_root):
                selected = CameraPhaseAnalysisController.find_latest_phase_analysis_session(
                    viewer
                )

        self.assertEqual(selected, str(newer.resolve()))


if __name__ == "__main__":
    unittest.main()