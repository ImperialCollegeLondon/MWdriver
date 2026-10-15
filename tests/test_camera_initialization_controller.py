import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mwdriver.ui.camera_initialization_controller as initialization_controller
from mwdriver.ui.camera_initialization_controller import CameraInitializationController


class FakeCamera:
    def __init__(self, calls, fail_gain=False):
        self.calls = calls
        self.fail_gain = fail_gain

    def set_exposure(self, exposure):
        self.calls.append(("exposure", exposure))

    def set_gain(self, gain):
        self.calls.append(("gain", gain))
        if self.fail_gain:
            raise RuntimeError("gain unsupported")

    def get_detector_size(self):
        self.calls.append(("detector_size",))
        return 640, 480

    def get_device_info(self):
        self.calls.append(("device_info",))
        return "fake camera"

    def set_trigger_mode(self, mode):
        self.calls.append(("trigger_mode", mode))

    def start_acquisition(self):
        self.calls.append(("start_acquisition",))


class CameraInitializationControllerTests(unittest.TestCase):
    def test_camera_setup_preserves_configuration_and_start_order(self):
        calls = []
        viewer = SimpleNamespace(
            cam=None,
            exposure_s=0.025,
            gain_value=3.0,
            current_camera_mode="external",
        )

        with patch.object(
            initialization_controller,
            "NativeThorlabsCamera",
            lambda: FakeCamera(calls),
        ):
            CameraInitializationController.init_camera(viewer)

        self.assertEqual(
            calls,
            [
                ("exposure", 0.025),
                ("gain", 3.0),
                ("detector_size",),
                ("device_info",),
                ("trigger_mode", "int"),
                ("start_acquisition",),
            ],
        )
        self.assertEqual((viewer.width, viewer.height), (640, 480))
        self.assertEqual(viewer.current_camera_mode, "live")

    def test_unsupported_gain_does_not_prevent_acquisition(self):
        calls = []
        viewer = SimpleNamespace(
            cam=None,
            exposure_s=0.01,
            gain_value=0.0,
            current_camera_mode="external",
        )

        with patch.object(
            initialization_controller,
            "NativeThorlabsCamera",
            lambda: FakeCamera(calls, fail_gain=True),
        ):
            CameraInitializationController.init_camera(viewer)

        self.assertIn(("start_acquisition",), calls)
        self.assertEqual(viewer.current_camera_mode, "live")


if __name__ == "__main__":
    unittest.main()
