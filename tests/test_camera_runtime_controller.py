import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from camera_runtime_controller import CameraRuntimeController


class CameraRuntimeControllerTests(unittest.TestCase):
    def test_external_trigger_frames_skip_display_and_profile_updates(self):
        raw_frame = object()
        display_frame = object()
        assignments = []
        viewer = SimpleNamespace(
            latest_raw_frame=None,
            latest_display_frame=None,
            phase_capture_active=True,
            current_camera_mode="external",
            handle_triggered_phase_frame=lambda frame, frame_meta=None: assignments.append(
                (frame, frame_meta)
            ),
            apply_display_mask=lambda _frame: self.fail("display mask must not run"),
            layer=None,
            update_intensity_profile=lambda: self.fail("profile update must not run"),
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

    def test_live_preview_updates_masked_layer_and_profile(self):
        raw_frame = object()
        display_frame = object()
        profile_updates = []
        viewer = SimpleNamespace(
            latest_raw_frame=None,
            latest_display_frame=None,
            phase_capture_active=False,
            current_camera_mode="live",
            apply_display_mask=lambda frame: ("masked", frame),
            layer=SimpleNamespace(data=None),
            update_intensity_profile=lambda: profile_updates.append(True),
        )

        CameraRuntimeController.update_frame(viewer, raw_frame, display_frame)

        self.assertIs(viewer.latest_raw_frame, raw_frame)
        self.assertIs(viewer.latest_display_frame, display_frame)
        self.assertEqual(viewer.layer.data, ("masked", display_frame))
        self.assertEqual(profile_updates, [True])


if __name__ == "__main__":
    unittest.main()