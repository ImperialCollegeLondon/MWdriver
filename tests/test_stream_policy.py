import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stream_policy import get_camera_stream_mode


class CameraStreamPolicyTests(unittest.TestCase):
    def get_mode(self, **overrides):
        state = {
            "streaming_allowed": True,
            "preview_layer_visible": True,
            "camera_mode": "live",
            "phase_capture_active": False,
            "recalibration_active": False,
        }
        state.update(overrides)
        return get_camera_stream_mode(**state)

    def test_preview_requires_permission_visible_layer_live_mode_and_idle_state(self):
        self.assertEqual(self.get_mode(), "preview")
        blocked_states = (
            {"streaming_allowed": False},
            {"preview_layer_visible": False},
            {"camera_mode": "external"},
            {"phase_capture_active": True},
            {"recalibration_active": True},
        )
        for overrides in blocked_states:
            with self.subTest(overrides=overrides):
                self.assertEqual(self.get_mode(**overrides), "stopped")

    def test_external_capture_mode_takes_priority_over_preview_gates(self):
        self.assertEqual(
            self.get_mode(
                streaming_allowed=False,
                preview_layer_visible=False,
                camera_mode="external",
                phase_capture_active=True,
                recalibration_active=True,
            ),
            "capture",
        )

    def test_external_mode_without_capture_stays_stopped(self):
        self.assertEqual(
            self.get_mode(camera_mode="external"),
            "stopped",
        )


if __name__ == "__main__":
    unittest.main()
