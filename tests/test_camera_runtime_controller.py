import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mwdriver.ui.camera_runtime_controller import CameraRuntimeController


class CameraRuntimeControllerTests(unittest.TestCase):
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