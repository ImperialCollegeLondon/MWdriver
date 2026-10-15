"""Compatibility re-exports for the packaged manual-capture controller."""

from mwdriver.ui.camera_manual_capture_controller import (
    CameraManualCaptureController,
    FrameSaveWorker,
    MANUAL_CAPTURE_COUNT,
    MANUAL_CAPTURE_INTERVAL_MS,
    SAVE_DIR,
    SCRIPT_DIR,
)

__all__ = [
    "CameraManualCaptureController",
    "FrameSaveWorker",
    "MANUAL_CAPTURE_COUNT",
    "MANUAL_CAPTURE_INTERVAL_MS",
    "SAVE_DIR",
    "SCRIPT_DIR",
]
