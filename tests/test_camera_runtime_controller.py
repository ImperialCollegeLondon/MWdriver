import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mwdriver.ui.camera_runtime_controller import CameraRuntimeController, CameraWorker


class CameraRuntimeControllerTests(unittest.TestCase):
    def test_live_preview_has_only_one_unacknowledged_frame(self):
        worker = CameraWorker(None)
        frames = []
        worker.frame_ready.connect(lambda raw, display, meta: frames.append(raw))
        frame = np.ones((4, 5))
        self.assertTrue(worker.publish_frame(frame, None))
        for _ in range(100):
            self.assertFalse(worker.publish_frame(frame, None))
        self.assertEqual(len(frames), 1)
        worker.acknowledge_preview_frame()
        self.assertTrue(worker.publish_frame(frame, None))

    def test_histogram_preview_rate_is_limited_but_capture_is_not(self):
        worker = CameraWorker(None, histogram_enabled_getter=lambda: True)
        frame = np.ones((4, 5))
        with patch("mwdriver.ui.camera_runtime_controller.time.monotonic", side_effect=[0, 0.1, 0.21]):
            self.assertTrue(worker.publish_frame(frame, None))
            worker.acknowledge_preview_frame()
            self.assertFalse(worker.publish_frame(frame, None))
            self.assertTrue(worker.publish_frame(frame, None))
        for index in range(6):
            self.assertTrue(worker.publish_frame(frame, {"slot": index}, external_capture=True))

    def test_preview_acknowledgement_runs_even_when_display_fails(self):
        worker = MagicMock()
        viewer = SimpleNamespace(worker=worker)
        with patch.object(CameraRuntimeController, "_apply_frame", side_effect=RuntimeError("display failed")):
            with self.assertRaises(RuntimeError):
                CameraRuntimeController.update_frame(viewer, None, None)
        worker.acknowledge_preview_frame.assert_called_once()

    def test_external_trigger_frames_skip_preview_updates(self):
        raw_frame = object()
        display_frame = object()
        assignments = []
        viewer = SimpleNamespace(
            latest_raw_frame=None,
            latest_display_frame=None,
            phase_capture_active=True,
            current_camera_mode="external",
            recalibration_active=False,
            handle_triggered_phase_frame=lambda frame, frame_meta=None: assignments.append(
                (frame, frame_meta)
            ),
            apply_display_mask=lambda _frame: self.fail("display mask must not run"),
            layer=None,
        )
        frame_meta = {"slot": 2}

        CameraRuntimeController.update_frame(
            viewer,
            raw_frame,
            display_frame,
            frame_meta,
        )

        self.assertIs(viewer.latest_raw_frame, raw_frame)
        self.assertIs(viewer.latest_display_frame, display_frame)
        self.assertEqual(assignments, [(raw_frame, frame_meta)])

    def test_calibration_previews_each_frame_after_capture_handling(self):
        assignments = []
        layer = SimpleNamespace(data=None)
        viewer = SimpleNamespace(
            latest_raw_frame=None,
            latest_display_frame=None,
            phase_capture_active=True,
            current_camera_mode="external",
            recalibration_active=True,
            layer=layer,
        )

        def handle_frame(raw_frame, frame_meta=None):
            self.assertIs(layer.data, previous_display_frame)
            assignments.append((raw_frame, frame_meta))
            if len(assignments) == 6:
                viewer.phase_capture_active = False
                viewer.current_camera_mode = "live"
                viewer.recalibration_active = False

        viewer.handle_triggered_phase_frame = handle_frame
        previous_display_frame = None
        for phase_index in range(6):
            with self.subTest(phase_index=phase_index):
                raw_frame = object()
                display_frame = object()
                frame_meta = {"slot": phase_index}

                CameraRuntimeController.update_frame(
                    viewer, raw_frame, display_frame, frame_meta
                )

                self.assertEqual(assignments[-1], (raw_frame, frame_meta))
                self.assertIs(viewer.latest_raw_frame, raw_frame)
                self.assertIs(viewer.latest_display_frame, display_frame)
                self.assertIs(layer.data, display_frame)
                previous_display_frame = display_frame

        self.assertEqual(len(assignments), 6)

    def test_live_preview_updates_layer_without_applying_roi_mask(self):
        raw_frame = object()
        display_frame = object()
        viewer = SimpleNamespace(
            latest_raw_frame=None,
            latest_display_frame=None,
            phase_capture_active=False,
            current_camera_mode="live",
            apply_display_mask=lambda _frame: self.fail(
                "live preview must not apply the ROI mask"
            ),
            layer=SimpleNamespace(data=None),
        )

        CameraRuntimeController.update_frame(viewer, raw_frame, display_frame)

        self.assertIs(viewer.latest_raw_frame, raw_frame)
        self.assertIs(viewer.latest_display_frame, display_frame)
        self.assertIs(viewer.layer.data, display_frame)


if __name__ == "__main__":
    unittest.main()