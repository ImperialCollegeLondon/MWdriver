import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mwdriver.hardware.thorlabs_native_camera import NativeThorlabsCamera
from mwdriver.capture.phase_capture_timestamps import build_phase_timestamp_payload


ENUMS = SimpleNamespace(
    OPERATION_MODE=SimpleNamespace(SOFTWARE_TRIGGERED=0, HARDWARE_TRIGGERED=1),
    TRIGGER_POLARITY=SimpleNamespace(ACTIVE_HIGH=0, ACTIVE_LOW=1),
)


class FakeCamera:
    def __init__(self):
        self.events = []
        self.frames = []
        self.buffer = np.zeros((2, 3), dtype=np.uint16)
        self.sensor_width_pixels = 3
        self.sensor_height_pixels = 2
        self.model = "fake"
        self.serial_number = "123"
        self.gain_range = SimpleNamespace(min=0, max=48)

    def convert_decibels_to_gain(self, gain):
        return int(round(gain * 2))

    def arm(self, count):
        self.events.append(("arm", count))

    def disarm(self):
        self.events.append("disarm")

    def issue_software_trigger(self):
        self.events.append("trigger")

    def dispose(self):
        self.events.append("dispose")

    def get_pending_frame_or_null(self):
        if not self.frames:
            self.buffer.fill(999)
            return None
        count = self.frames.pop(0)
        self.buffer.fill(count)
        return SimpleNamespace(
            image_buffer=self.buffer,
            frame_count=count,
            time_stamp_relative_ns_or_null=count * 1_000_000,
        )


class FakeSDK:
    def __init__(self):
        self.camera = FakeCamera()
        self.disposed = False

    def discover_available_cameras(self):
        return ["123"]

    def open_camera(self, serial):
        return self.camera

    def dispose(self):
        self.disposed = True


class NativeCameraTests(unittest.TestCase):
    def setUp(self):
        self.sdk = FakeSDK()
        self.camera = NativeThorlabsCamera(sdk_factory=lambda: self.sdk, enums=ENUMS)
        self.addCleanup(self.camera.close)

    def test_live_view_arms_and_issues_one_continuous_trigger(self):
        self.camera.set_trigger_mode("int")
        self.camera.start_acquisition()
        self.assertEqual(self.sdk.camera.operation_mode, 0)
        self.assertEqual(self.sdk.camera.frames_per_trigger_zero_for_unlimited, 0)
        self.assertEqual(self.sdk.camera.events, [("arm", 64), "trigger"])

    def test_external_capture_preserves_six_frames_and_copies_buffers(self):
        self.camera.set_trigger_mode("ext")
        self.camera.setup_ext_trigger("rise")
        self.camera.start_acquisition(1)
        self.assertEqual(self.sdk.camera.frames_per_trigger_zero_for_unlimited, 1)
        self.assertEqual(self.sdk.camera.trigger_polarity, 0)
        self.assertNotIn("trigger", self.sdk.camera.events)
        self.sdk.camera.frames = list(range(6))
        image_range = self.camera.get_new_images_range()
        images, metadata = self.camera.read_multiple_images(image_range, return_info=True)
        self.assertEqual(len(images), 6)
        for index, image in enumerate(images):
            np.testing.assert_array_equal(image, np.full((2, 3), index))
            self.assertEqual(metadata[index]["framestamp"], index)
            self.assertEqual(metadata[index]["acquisition_index"], index)
            self.assertEqual(metadata[index]["trigger_mode"], "ext")
            self.assertEqual(metadata[index]["camera_timestamp"], index * 1_000_000)
            self.assertEqual(metadata[index]["acquisition_id"], metadata[0]["acquisition_id"])
        payload = build_phase_timestamp_payload(metadata[5], host_time_s=1.0, phase_index=5)
        self.assertEqual(payload["camera_timestamp"] / self.camera.get_timestamp_clock_frequency(), 0.005)
        self.assertIsNone(self.camera.get_new_images_range())

    def test_live_view_returns_newest_frame(self):
        self.camera.start_acquisition()
        self.sdk.camera.frames = [1, 2, 3]
        image, metadata = self.camera.read_newest_image(return_info=True)
        np.testing.assert_array_equal(image, np.full((2, 3), 3))
        self.assertEqual(metadata["framestamp"], 3)
        self.assertIsNone(self.camera.read_newest_image())

    def test_settings_use_native_units(self):
        self.camera.set_exposure(0.0125)
        self.camera.set_gain(7.5)
        self.assertEqual(self.sdk.camera.exposure_time_us, 12500)
        self.assertEqual(self.sdk.camera.gain, 15)
        self.assertEqual(self.camera.get_detector_size(), (3, 2))

    def test_gain_is_clamped_to_camera_range(self):
        self.camera.set_gain(30)
        self.assertEqual(self.sdk.camera.gain, 48)
        self.camera.set_gain(-1)
        self.assertEqual(self.sdk.camera.gain, 0)

    def test_mode_switch_clears_pending_frames_and_resumes_live_view(self):
        self.camera.set_trigger_mode("ext")
        self.camera.start_acquisition(1)
        self.sdk.camera.frames = [1]
        self.camera.get_new_images_range()
        self.camera.set_trigger_mode("int")
        self.camera.start_acquisition()
        self.assertIsNone(self.camera.read_newest_image())
        self.assertIn("disarm", self.sdk.camera.events)
        self.assertEqual(self.sdk.camera.events[-1], "trigger")

    def test_close_disposes_camera_before_sdk_and_is_repeatable(self):
        self.camera.start_acquisition()
        self.camera.close()
        self.camera.close()
        self.assertEqual(self.sdk.camera.events[-2:], ["disarm", "dispose"])
        self.assertTrue(self.sdk.disposed)

    def test_open_failure_disposes_sdk(self):
        self.sdk.discover_available_cameras = lambda: []
        with self.assertRaisesRegex(RuntimeError, "No Thorlabs camera"):
            NativeThorlabsCamera(sdk_factory=lambda: self.sdk, enums=ENUMS)
        self.assertTrue(self.sdk.disposed)


if __name__ == "__main__":
    unittest.main()