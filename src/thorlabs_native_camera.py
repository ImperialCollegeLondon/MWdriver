"""Native Thorlabs TSI backend for the desktop capture controller."""

import os
import math
from pathlib import Path
from threading import RLock

import numpy as np


_dll_directory_handles = []


def _configure_dll_path():
    if os.name != "nt":
        return
    configured = os.environ.get("THORLABS_TSI_DLL_DIR")
    if configured:
        directories = [Path(configured)]
    else:
        directories = [
            Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
            / "Thorlabs" / "Scientific Imaging" / "ThorCam",
            Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
            / "Thorlabs" / "ThorImageCAM" / "Bin",
        ]
    for directory in directories:
        if (directory / "thorlabs_tsi_camera_sdk.dll").is_file():
            os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
            _dll_directory_handles.append(os.add_dll_directory(str(directory)))
            return
    raise RuntimeError(
        "Thorlabs native camera DLLs were not found. Install ThorCam or set "
        "THORLABS_TSI_DLL_DIR to the SDK's Native_64_lib directory "
        "(Native_32_lib for 32-bit Python)."
    )


class NativeThorlabsCamera:
    """Own the native SDK, camera and copied pending frames."""

    def __init__(self, serial_number=None, *, sdk_factory=None, enums=None):
        if sdk_factory is None:
            _configure_dll_path()
            try:
                from thorlabs_tsi_sdk.tl_camera import TLCameraSDK
                from thorlabs_tsi_sdk import tl_camera_enums
            except ImportError as error:
                raise RuntimeError(
                    "Install the official Thorlabs Python SDK package: "
                    "python -m pip install <path-to-thorlabs_tsi_camera_python_sdk_package.zip>"
                ) from error
            sdk_factory = TLCameraSDK
            enums = tl_camera_enums
        self._enums = enums
        self._lock = RLock()
        self._pending = []
        self._next_index = 0
        self._acquisition_id = 0
        self._armed = False
        self._mode = "int"
        self._camera = None
        self._sdk = sdk_factory()
        try:
            serials = self._sdk.discover_available_cameras()
            print("Detected cameras:", serials)
            if not serials:
                raise RuntimeError(
                    "No Thorlabs camera detected. Check USB, camera drivers and "
                    "whether ThorCam or another application has the camera open."
                )
            self._camera = self._sdk.open_camera(serial_number or serials[0])
            self._camera.image_poll_timeout_ms = 0
        except Exception:
            self.close()
            raise

    def set_exposure(self, exposure_s):
        with self._lock:
            self._camera.exposure_time_us = int(round(exposure_s * 1_000_000))

    def set_gain(self, gain):
        with self._lock:
            gain_index = self._camera.convert_decibels_to_gain(gain)
            gain_range = self._camera.gain_range
            self._camera.gain = max(gain_range.min, min(gain_index, gain_range.max))

    def get_detector_size(self):
        with self._lock:
            return self._camera.sensor_width_pixels, self._camera.sensor_height_pixels

    def get_device_info(self):
        with self._lock:
            return {"model": self._camera.model, "serial_number": self._camera.serial_number}

    def set_trigger_mode(self, mode):
        if mode not in ("int", "ext"):
            raise ValueError(f"Unsupported trigger mode: {mode}")
        with self._lock:
            self.stop_acquisition()
            self._camera.operation_mode = (
                self._enums.OPERATION_MODE.SOFTWARE_TRIGGERED
                if mode == "int" else self._enums.OPERATION_MODE.HARDWARE_TRIGGERED
            )
            self._mode = mode

    def setup_ext_trigger(self, edge):
        polarities = {
            "rise": self._enums.TRIGGER_POLARITY.ACTIVE_HIGH,
            "fall": self._enums.TRIGGER_POLARITY.ACTIVE_LOW,
        }
        if edge not in polarities:
            raise ValueError(f"Unsupported trigger edge: {edge}")
        with self._lock:
            self._camera.trigger_polarity = polarities[edge]

    def start_acquisition(self, frames_per_trigger=None):
        with self._lock:
            self.stop_acquisition()
            self._camera.frames_per_trigger_zero_for_unlimited = (
                0 if self._mode == "int" else (frames_per_trigger or 1)
            )
            self._camera.arm(64)
            self._armed = True
            self._acquisition_id += 1
            self._pending.clear()
            self._next_index = 0
            try:
                if self._mode == "int":
                    self._camera.issue_software_trigger()
            except Exception:
                self.stop_acquisition()
                raise

    def stop_acquisition(self):
        with self._lock:
            if self._armed:
                self._camera.disarm()
                self._armed = False
            self._pending.clear()

    def _poll_frames(self):
        if not self._armed:
            return
        for _ in range(64):
            frame = self._camera.get_pending_frame_or_null()
            if frame is None:
                break
            image = np.array(frame.image_buffer, copy=True)
            metadata = {
                "acquisition_id": self._acquisition_id,
                "trigger_mode": self._mode,
                "acquisition_index": self._next_index,
                "framestamp": frame.frame_count,
                "camera_timestamp": frame.time_stamp_relative_ns_or_null,
            }
            self._pending.append((self._next_index, image, metadata))
            self._next_index += 1

    def get_new_images_range(self):
        with self._lock:
            if not self._pending:
                self._poll_frames()
            if not self._pending:
                return None
            return self._pending[0][0], self._pending[-1][0] + 1

    def read_multiple_images(self, image_range, return_info=False):
        with self._lock:
            start, stop = image_range
            selected = [item for item in self._pending if start <= item[0] < stop]
            self._pending = [item for item in self._pending if item[0] >= stop]
            images = [item[1] for item in selected]
            if return_info:
                return images, [item[2] for item in selected]
            return images

    def read_newest_image(self, return_info=False):
        with self._lock:
            if not self._pending:
                self._poll_frames()
            if not self._pending:
                return None
            _, image, metadata = self._pending[-1]
            self._pending.clear()
            return (image, metadata) if return_info else image

    def get_timestamp_clock_frequency(self):
        return 1_000_000_000

    def get_recommended_trigger_spacing_ms(self, exposure_s=None):
        with self._lock:
            exposure_ms = (
                self._camera.exposure_time_us / 1000.0
                if exposure_s is None else exposure_s * 1000.0
            )
            readout_ms = self._camera.sensor_readout_time_ns / 1_000_000.0
            return math.ceil(exposure_ms + readout_ms + 5.0)

    @property
    def acquisition_id(self):
        with self._lock:
            return self._acquisition_id

    def close(self):
        with self._lock:
            try:
                self.stop_acquisition()
            finally:
                try:
                    if self._camera is not None:
                        self._camera.dispose()
                        self._camera = None
                finally:
                    if self._sdk is not None:
                        self._sdk.dispose()
                        self._sdk = None