import time
import os
import json

os.environ["QT_API"] = "pyside6"

from qtpy.QtCore import QObject

import numpy as np


import napari
from application_bootstrap import run_application
from phase_capture_controller import (
    PHASE_CAPTURE_COUNT,
    PhaseCaptureController,
)
from camera_runtime_controller import CameraRuntimeController, CameraWorker
from camera_recalibration_controller import CameraRecalibrationController
from camera_controls_controller import (
    CameraControlsController,
    STANDARD_PHASE_EXPOSURE_MS,
)
from camera_roi_controller import CameraRoiController
from camera_view_controller import CameraViewController
from camera_profile_controller import CameraProfileController
from camera_dock_controller import CameraDockController
from camera_phase_analysis_controller import (
    CameraPhaseAnalysisController,
    PHASE_ANALYSIS_SCRIPT,
)
from camera_manual_capture_controller import (
    CameraManualCaptureController,
    FrameSaveWorker,
    MANUAL_CAPTURE_COUNT,
    MANUAL_CAPTURE_INTERVAL_MS,
    SAVE_DIR,
    SCRIPT_DIR,
)
from camera_initialization_controller import CameraInitializationController
from camera_phase_command_controller import CameraPhaseCommandController
from thorlabs_native_camera import NativeThorlabsCamera
from pico_serial_worker import PicoSerialWorker
from phase_capture_worker import PhaseSaveWorker
from phase_capture_storage import PhaseSaveResult
from mwdriver.domain.recalibration import (
    DEFAULT_RECALIBRATION_CONFIG,
)
from mwdriver.domain.phase_psi import (
    DEFAULT_ALGORITHM as DEFAULT_PSI_ALGORITHM,
    LEGACY_90_DEG,
    get_algorithm as get_psi_algorithm,
)


# ============================================================
# Pico / Phase Capture å‚æ•°
# ============================================================

PICO_PORT = "COM4"
PICO_BAUDRATE = 115200

# Automatic phase-point checker.
# After every six-frame phase capture is fully saved, camera.py launches
# check_measured_vs_expected_phase.py for that exact session folder.
PHASE_CHECK_SCRIPT = os.path.join(
    SCRIPT_DIR,
    "check_measured_vs_expected_phase.py"
)
PHASE_CHECK_TIMEOUT_S = 180

print("Automatic phase checker:", PHASE_CHECK_SCRIPT)

print("Default save directory:", SAVE_DIR)
print("Phase analysis viewer:", PHASE_ANALYSIS_SCRIPT)

# ============================================================
# Camera trigger mode
# ============================================================
# Mixed mode:
#   normal state: internal trigger / live view
#   phase capture: temporarily switch to external trigger
#   after 6 frames: switch back to internal trigger / live view

# Recalibration parameters.
RECALIBRATION_CONFIG = DEFAULT_RECALIBRATION_CONFIG
RECALIBRATION_TRIGGER_JSON_PATH = os.path.join(
    SCRIPT_DIR,
    "phase_trigger_positions.json"
)
RECALIBRATION_TRIGGER_UPDATE_ACK_TIMEOUT_S = 1.5

# Pico drives GP15 with PWM at logic-level amplitude; the falling edge duty
# ramps linearly from MAX_DUTY down to 0 across the configured falling edge,
# mirroring build_falling_duty_table() in main.py on the Pico.


class ThorlabsCameraViewer(
    PhaseCaptureController,
    CameraViewController,
    CameraRoiController,
    CameraProfileController,
    CameraControlsController,
    CameraDockController,
    CameraRuntimeController,
    CameraPhaseAnalysisController,
    CameraRecalibrationController,
    CameraManualCaptureController,
    CameraInitializationController,
    CameraPhaseCommandController,
    QObject,
):
    def __init__(self):
        super().__init__()

        self.cam = None
        self.viewer = None
        self.layer = None

        self.pico_port = PICO_PORT
        self.pico_baudrate = PICO_BAUDRATE

        # Camera runs in live view normally.
        # It is switched to external trigger only during phase capture.
        self.current_camera_mode = "live"
        self.pending_phase_indices = []
        # Stores triggered frames that arrive before the serial CAPTURE_PHASE_x marker.
        # This protects phase capture when trigger intervals are short and serial/USB timing jitter occurs.
        self.unassigned_triggered_frames = []
        self.switching_camera_mode = False

        self.pico_worker = None
        self.pico_thread = None

        self.phase_capture_active = False
        self.phase_capture_frames = [None] * PHASE_CAPTURE_COUNT
        self.phase_capture_times = [None] * PHASE_CAPTURE_COUNT
        self.phase_capture_frame_timestamps = [None] * PHASE_CAPTURE_COUNT
        self.phase_capture_timestamp_clock_hz = None
        self.phase_capture_count = 0
        self.phase_capture_start_time = None
        self.phase_capture_mode_enter_time_s = None

        self.initialize_manual_capture_state()

        self.save_worker = None
        self.save_thread = None

        # When recalibration needs to queue another automatic capture,
        # defer it until the current save worker is fully cleaned up.
        self.pending_recalibration_trigger_positions_ms = None
        self.pending_recalibration_status_text = None

        self.roi_layer = None
        self.roi_params = None
        self.roi_mask = None

        self.last_roi_count = 0
        self.waiting_for_roi_finish = False

        self.mask_mode = "outside"
        self.display_mask_enabled = False
        self.current_display_mask = None

        self.initialize_profile_controller_state()
        self.latest_raw_frame = None
        self.latest_display_frame = None
        self.camera_adjust_widget = None
        self.camera_status_widget = None
        self.phase_analysis_controls_widget = None
        self.phase_analysis_title_label = None
        self.initialize_camera_dock_state()
        self.phase_analysis_selector = None
        self.view_phase_analysis_button = None

        self.is_streaming = False
        self.streaming_allowed = True

        self.worker = None
        self.worker_thread = None

        self.exposure_s = STANDARD_PHASE_EXPOSURE_MS[
            get_psi_algorithm(DEFAULT_PSI_ALGORITHM)["phase_step_deg"]
        ] / 1000.0
        self.gain_value = 0.0

        self.display_downsample = 1

        self.recalibration_active = False
        self.recalibration_stage = None
        self.recalibration_coarse_positions_ms = None
        self.recalibration_target_positions_ms = None
        self.recalibration_current_positions_ms = None
        self.recalibration_refinement_round = 0
        self.recalibration_incomplete_verification_retries = 0
        self.recalibration_verification_capture_count = 0
        self.recalibration_pending_pass_candidate = None
        self.recalibration_coarse_session_dir = None
        self.recalibration_candidates = []
        self.recalibration_auto_restart_count = 0
        self.latest_phase_save_result = None
        self.phase_analysis_process = None
        self.phase_analysis_output_path = None
        self.phase_analysis_source_stack_name = None
        self.phase_analysis_results = {}
        self.phase_analysis_layer = None
        self.phase_analysis_layer_records = []
        self.phase_analysis_view_active = False
        self.phase_capture_time_series_layers = []
        self.pending_phase_capture_layer = None
        self.last_trigger_update_ack_positions_ms = None
        self.last_trigger_update_error_text = None
        self.active_trigger_positions_ms = None
        self.selected_psi_algorithm_id = DEFAULT_PSI_ALGORITHM
        self.last_selected_psi_algorithm_id = DEFAULT_PSI_ALGORITHM
        self.last_capture_algorithm_id = DEFAULT_PSI_ALGORITHM
        # Trigger positions actually sent to Pico for the in-progress capture,
        # used to plot measured phase vs. waveform voltage after saving.
        self.last_capture_trigger_positions_ms = None

        self.init_camera()
        self.init_viewer()
        self.init_controls()
        self.init_roi_controls()
        self.init_profile_controls()
        self.init_mwdriver_controls_dock()

        self.load_active_trigger_positions_from_json()

        self.sync_stream_with_active_layer()
        self.start_pico_listener()

    def load_active_trigger_positions_from_json(self):
        if not os.path.exists(RECALIBRATION_TRIGGER_JSON_PATH):
            self.active_trigger_positions_ms = None
            return None

        try:
            with open(
                RECALIBRATION_TRIGGER_JSON_PATH,
                "r",
                encoding="utf-8",
            ) as file:
                payload = json.load(file)

            calibrations = payload.get("calibrations", {})
            calibration = calibrations.get(self.selected_psi_algorithm_id)
            if calibration is None and "trigger_positions_ms" in payload:
                if self.selected_psi_algorithm_id != LEGACY_90_DEG:
                    self.active_trigger_positions_ms = None
                    return None
                calibration = payload

            if calibration is None:
                self.active_trigger_positions_ms = None
                return None

            positions = calibration.get("trigger_positions_ms")
            positions = self.validate_trigger_positions_ms(positions)
            self.active_trigger_positions_ms = positions
            print(
                "Loaded active trigger positions for",
                self.selected_psi_algorithm_id,
                ":",
                positions,
            )
            return positions

        except Exception as error:
            print(
                "Failed to load active trigger positions from JSON:",
                error,
            )
            self.active_trigger_positions_ms = None
            return None

    def get_recalibration_target_phase_deg(self):
        return np.asarray(
            get_psi_algorithm(self.selected_psi_algorithm_id)["expected_phase_deg"],
            dtype=np.float64,
        )


    def cleanup_save_worker(self):
        self.save_worker = None
        self.save_thread = None
        print("Save worker cleaned up.")

        pending_positions = self.pending_recalibration_trigger_positions_ms
        if (
            pending_positions is None
            or not self.recalibration_active
        ):
            return

        try:
            status_text = self.pending_recalibration_status_text
            if status_text:
                self.status_label.setText(status_text)

            self.pending_recalibration_trigger_positions_ms = None
            self.pending_recalibration_status_text = None

            print(
                "Save worker cleaned up; starting deferred recalibration capture.",
                pending_positions,
            )

            self.queue_phase_capture_with_trigger_positions(
                pending_positions
            )

        except Exception as error:
            self.abort_recalibration(
                f"Status: Recalibration restart after save failed: {error}"
            )

    def close(self):
        print("Closing camera...")

        try:
            self.stop_pico_listener()
        except Exception as e:
            print("Error while stopping Pico listener:", e)

        try:
            self.clear_intensity_profile()
        except Exception as e:
            print("Error while clearing intensity profile:", e)

        try:
            self.stop_stream()
        except Exception as e:
            print("Error while stopping stream:", e)

        try:
            if self.save_thread is not None:
                self.save_thread.quit()
                self.save_thread.wait(2000)
        except Exception as e:
            print("Error while stopping save thread:", e)

        try:
            if (
                self.phase_analysis_process is not None
                and self.phase_analysis_process.poll() is None
            ):
                self.phase_analysis_process.terminate()
                self.phase_analysis_process.wait(timeout=2)
        except Exception as e:
            print("Error while stopping phase analysis:", e)

        try:
            if self.cam is not None:
                self.cam.stop_acquisition()
                self.cam.close()
        except Exception as e:
            print("Error while closing camera:", e)

        print("Camera closed.")


def main():
    return run_application(ThorlabsCameraViewer, napari.run)


if __name__ == "__main__":
    main()