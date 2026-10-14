import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import camera_manual_capture_controller as manual_controller
from camera_manual_capture_controller import CameraManualCaptureController


class StatusLabel:
    def __init__(self):
        self.text = None

    def setText(self, text):
        self.text = text


class ManualCaptureViewer(CameraManualCaptureController):
    def __init__(self, frame):
        self.initialize_manual_capture_state()
        self.latest_raw_frame = frame
        self.is_streaming = True
        self.status_label = StatusLabel()
        self.saved_frames = None
        self.save_thread = None

    def start_save_manual_frames_worker(self, frames):
        self.saved_frames = frames


class CameraManualCaptureControllerTests(unittest.TestCase):
    def test_request_collects_six_copied_frames_at_configured_interval(self):
        frame = np.zeros((2, 2), dtype=np.uint16)
        viewer = ManualCaptureViewer(frame)
        scheduled = []

        with patch.object(
            manual_controller.QTimer,
            "singleShot",
            side_effect=lambda delay, callback: scheduled.append((delay, callback)),
        ):
            viewer.request_manual_capture()
            self.assertEqual(viewer.manual_capture_index, 1)
            self.assertEqual(len(scheduled), 1)

            for frame_value in range(2, 7):
                frame.fill(frame_value)
                delay, callback = scheduled.pop(0)
                self.assertEqual(delay, manual_controller.MANUAL_CAPTURE_INTERVAL_MS)
                callback()
                if frame_value < manual_controller.MANUAL_CAPTURE_COUNT:
                    self.assertEqual(len(scheduled), 1)

        self.assertEqual(len(viewer.saved_frames), manual_controller.MANUAL_CAPTURE_COUNT)
        expected_values = [0, 2, 3, 4, 5, 6]
        for expected_value, captured_frame in zip(expected_values, viewer.saved_frames):
            np.testing.assert_array_equal(
                captured_frame,
                np.full((2, 2), expected_value, dtype=np.uint16),
            )
        self.assertFalse(viewer.manual_capture_in_progress)
        self.assertEqual(viewer.manual_capture_index, 0)
        self.assertEqual(viewer.manual_capture_frames, [])

    def test_request_without_a_frame_does_not_start_capture(self):
        viewer = ManualCaptureViewer(frame=None)
        viewer.manual_capture_in_progress = False

        viewer.request_manual_capture()

        self.assertFalse(viewer.manual_capture_in_progress)
        self.assertEqual(viewer.manual_capture_frames, [])
        self.assertEqual(viewer.manual_capture_index, 0)
        self.assertEqual(viewer.status_label.text, "Status: No frame available for manual capture")


if __name__ == "__main__":
    unittest.main()
