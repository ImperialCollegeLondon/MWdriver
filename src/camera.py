import sys
import time
import os
import json
import copy
import subprocess
import ast
import shutil

os.environ["QT_API"] = "pyside6"

from qtpy.QtCore import QObject, QThread, Signal, QTimer, Qt, QEvent, QCoreApplication
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QFormLayout,
    QGridLayout,
    QAbstractButton,
    QScrollArea,
)

import numpy as np

from thorlabs_native_camera import NativeThorlabsCamera

import napari
from application_bootstrap import run_application
from phase_capture_controller import (
    EXTERNAL_TRIGGER_EDGE,
    PHASE_CAPTURE_COUNT,
    TRIANGLE_WAVE_SUPPLY_VOLTAGE,
    PhaseCaptureController,
    triangle_wave_falling_edge_voltage,
)
from stream_policy import get_camera_stream_mode
from camera_frames import prepare_display_frame
from worker_lifecycle import stop_worker_thread
from camera_controls_controller import (
    CameraControlsController,
    STANDARD_PHASE_EXPOSURE_MS,
)
from camera_roi_controller import CameraRoiController
from camera_view_controller import CameraViewController
from camera_profile_controller import CameraProfileController

from phase_capture_timestamps import (
    build_phase_timestamp_payload,
    normalize_frame_framestamp,
)
from phase_capture_frame_filter import (
    decide_framestamp_phase_slot,
    decide_native_phase_frame,
    decide_phase_frame_assignment,
    decide_triangle_done_action,
    select_queued_phase_slots,
    should_accept_phase_frame,
)
from phase_capture_storage import (
    allocate_capture_session_dir,
    PhaseSaveResult,
    save_phase_capture_metadata,
    save_phase_pngs,
    save_raw_phase_stack,
    save_roi_snapshots,
)
from phase_capture_process import run_phase_check
from pico_serial_worker import PicoSerialWorker
from manual_capture_worker import FrameSaveWorker
from phase_capture_worker import PhaseSaveWorker
from recalibration import (
    DEFAULT_RECALIBRATION_CONFIG,
    build_even_trigger_positions_ms,
    compute_coarse_refined_positions_ms,
    compute_iterative_refined_positions_ms as compute_iterative_refined_positions,
    decide_incomplete_capture_action,
    decide_verification_capture_action,
    enforce_minimum_trigger_spacing_ms,
    is_strict_recalibration_pass,
    score_recalibration_candidate,
    select_best_recalibration_candidate,
    should_auto_restart_recalibration as should_auto_restart_candidate,
    validate_coarse_measurement,
    validate_trigger_positions_ms as validate_recalibration_trigger_positions_ms,
)
from phase_psi import (
    DEFAULT_ALGORITHM as DEFAULT_PSI_ALGORITHM,
    LEGACY_90_DEG,
    get_algorithm as get_psi_algorithm,
)


# ============================================================
# Pico / Phase Capture å‚æ•°
# ============================================================

PICO_PORT = "COM4"
PICO_BAUDRATE = 115200

MANUAL_CAPTURE_COUNT = 6
MANUAL_CAPTURE_INTERVAL_MS = 150

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(SCRIPT_DIR, "captured_frames")

# Automatic phase-point checker.
# After every six-frame phase capture is fully saved, camera.py launches
# check_measured_vs_expected_phase.py for that exact session folder.
PHASE_CHECK_SCRIPT = os.path.join(
    SCRIPT_DIR,
    "check_measured_vs_expected_phase.py"
)
PHASE_CHECK_TIMEOUT_S = 180

print("Automatic phase checker:", PHASE_CHECK_SCRIPT)

PHASE_ANALYSIS_SCRIPT = os.path.join(
    SCRIPT_DIR,
    "phase_analysis_new.py"
)

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



class CameraWorker(QObject):
    frame_ready = Signal(object, object, object)
    fps_ready = Signal(float)
    status_ready = Signal(str)

    def __init__(self, cam, display_downsample=1, phase_mode_getter=None):
        super().__init__()

        self.cam = cam
        self.display_downsample = display_downsample
        self.phase_mode_getter = phase_mode_getter

        self.running = False

        self.n_frames = 0
        self.t_fps = time.monotonic()
        self.last_fps_update = time.monotonic()
        self.fps_update_interval = 1.0

    def run(self):
        self.running = True

        self.n_frames = 0
        self.t_fps = time.monotonic()
        self.last_fps_update = time.monotonic()

        self.status_ready.emit("Status: Running...")

        while self.running:
            try:
                image_range = self.cam.get_new_images_range()

                if image_range is None:
                    QThread.msleep(1)
                    continue

                # During phase capture, GP12 triggers can arrive very close together.
                # Do not use read_newest_image() in that mode, because it can skip
                # older triggered frames if several frames are already in the camera buffer.
                # Instead, read all currently available frames and emit them one by one.
                in_phase_external_mode = False
                try:
                    if callable(self.phase_mode_getter):
                        in_phase_external_mode = bool(self.phase_mode_getter())
                except Exception:
                    in_phase_external_mode = False

                raw_frames = []
                raw_frame_infos = []

                if in_phase_external_mode:
                    try:
                        frames_result = self.cam.read_multiple_images(
                            image_range,
                            return_info=True,
                        )

                        if frames_result is None:
                            raw_frames = []
                            raw_frame_infos = []
                        elif isinstance(frames_result, tuple):
                            frames = frames_result[0]
                            infos = frames_result[1] if len(frames_result) > 1 else None
                        else:
                            frames = frames_result
                            infos = None

                        if isinstance(frames, np.ndarray):
                            if frames.ndim == 2:
                                raw_frames = [frames]
                            elif frames.ndim >= 3:
                                raw_frames = [frames[i] for i in range(frames.shape[0])]
                        elif isinstance(frames, (list, tuple)):
                            raw_frames = [np.asarray(frame) for frame in frames]

                        if isinstance(infos, (list, tuple)):
                            raw_frame_infos = list(infos)
                        elif infos is not None:
                            raw_frame_infos = [infos] * max(1, len(raw_frames))
                        else:
                            raw_frame_infos = [None] * len(raw_frames)

                    except Exception as e:
                        self.status_ready.emit(
                            f"Status: read_multiple_images warning: {e}; using newest frame fallback"
                        )
                        raw_frame = self.cam.read_newest_image()
                        if raw_frame is not None:
                            raw_frames = [raw_frame]
                            raw_frame_infos = [None]
                else:
                    try:
                        raw_frame = self.cam.read_newest_image(return_info=True)
                        if raw_frame is not None:
                            if isinstance(raw_frame, tuple) and len(raw_frame) == 2:
                                raw_frame, frame_info = raw_frame
                                raw_frames = [raw_frame]
                                raw_frame_infos = [frame_info]
                            else:
                                raw_frames = [raw_frame]
                                raw_frame_infos = [None]
                        else:
                            raw_frames = []
                            raw_frame_infos = []
                    except Exception as e:
                        self.status_ready.emit(
                            f"Status: read_newest_image warning: {e}; using fallback"
                        )
                        raw_frame = self.cam.read_newest_image()
                        if raw_frame is not None:
                            raw_frames = [raw_frame]
                            raw_frame_infos = [None]

                if len(raw_frames) == 0:
                    QThread.msleep(1)
                    continue

                for index, raw_frame in enumerate(raw_frames):
                    if raw_frame is None:
                        continue

                    frame_meta = raw_frame_infos[index] if index < len(raw_frame_infos) else None

                    display_frame = prepare_display_frame(
                        raw_frame,
                        self.display_downsample,
                    )

                    self.frame_ready.emit(raw_frame, display_frame, frame_meta)

                    self.n_frames += 1

                now = time.monotonic()

                if now - self.last_fps_update >= self.fps_update_interval:
                    fps = self.n_frames / (now - self.t_fps)
                    self.fps_ready.emit(fps)
                    self.last_fps_update = now

            except Exception as e:
                self.status_ready.emit(f"Status: Worker error: {e}")
                QThread.msleep(10)

        self.status_ready.emit("Status: Stopped")

    def stop(self):
        self.running = False


class ThorlabsCameraViewer(
    PhaseCaptureController,
    CameraViewController,
    CameraRoiController,
    CameraProfileController,
    CameraControlsController,
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

        self.manual_capture_in_progress = False
        self.manual_capture_frames = []
        self.manual_capture_index = 0

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
        self.mwdriver_controls_scroll = None
        self.mwdriver_controls_content = None
        self.mwdriver_controls_layout = None
        self.phase_analysis_selector = None
        self.view_phase_analysis_button = None
        self.controls_container = None
        self.controls_resize_filter_widgets = []
        self.controls_resize_timer = None

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


    def init_camera(self):
        print("Searching for Thorlabs camera...")
        self.cam = NativeThorlabsCamera()

        self.cam.set_exposure(self.exposure_s)

        try:
            self.cam.set_gain(self.gain_value)
            print(f"Gain: {self.gain_value}")
        except Exception as e:
            print("This camera may not support software gain control:", e)

        self.width, self.height = self.cam.get_detector_size()

        print("Camera info:", self.cam.get_device_info())
        print("Detector size:", self.width, self.height)
        print(f"Exposure: {self.exposure_s * 1000:.1f} ms")

        # Start in internal trigger mode for normal live view.
        try:
            self.cam.set_trigger_mode("int")
            self.current_camera_mode = "live"
            print("Camera trigger mode: int / live view")
        except Exception as e:
            print("Warning: set_trigger_mode('int') failed:", e)
            print("Continuing with camera default acquisition mode.")

        self.cam.start_acquisition()
        print("Camera acquisition started in live view mode.")

    def init_viewer(self):
        display_height = self.height // self.display_downsample
        display_width = self.width // self.display_downsample

        dummy = np.zeros((display_height, display_width), dtype=np.uint16)

        self.viewer = napari.Viewer()

        self.viewer.window._qt_window.resize(1600, 900)
        self.viewer.window._qt_window.showMaximized()

        self.layer = self.viewer.add_image(
            dummy,
            name="Camera Preview",
            colormap="gray",
            contrast_limits=(0, 1023)
        )
        self.layer.events.visible.connect(
            self.on_camera_preview_visibility_changed
        )

        self.roi_layer = self.viewer.add_shapes(
            name="ROI",
            ndim=2,
            shape_type="rectangle",
            edge_width=3,
            edge_color="red",
            face_color=[1, 0, 0, 0.15]
        )

        self.roi_layer.events.data.connect(self.on_roi_data_changed)

        self.viewer.layers.selection.active = self.layer
        self.viewer.reset_view()

        print("Napari viewer ready.")
        print(
            f"Preview size: {display_width} x {display_height} "
            f"(downsample x{self.display_downsample})"
        )

        QTimer.singleShot(500, self.simplify_layer_controls)
        QTimer.singleShot(600, self.simplify_napari_interface)
        QTimer.singleShot(700, self.install_controls_resize_filter)
        QTimer.singleShot(1000, self.simplify_napari_interface)

        try:
            self.viewer.layers.events.removed.connect(
                self.on_phase_analysis_layer_removed
            )
            self.viewer.layers.selection.events.active.connect(
                self.on_active_layer_changed
            )
        except Exception as e:
            print("Layer controls active-change hook warning:", e)

        try:
            if self.on_profile_mouse_drag in self.viewer.mouse_drag_callbacks:
                self.viewer.mouse_drag_callbacks.remove(self.on_profile_mouse_drag)

            self.viewer.mouse_drag_callbacks.append(self.on_profile_mouse_drag)
            print("Viewer-level profile mouse drag callback connected.")
        except Exception as e:
            print("Viewer-level profile mouse drag callback warning:", e)









    def simplify_napari_interface(self):
        """
        Hide unnecessary napari built-in buttons around the layer list.

        This keeps the real layers and functions unchanged:
        - ROI layer is still used for drawing / moving / resizing ROI.
        - Camera Preview layer is still used for live camera display.
        - Other program functions are not removed.

        It only hides napari's own UI buttons such as add layer, delete layer,
        console, grid, home, 2D/3D and other default viewer buttons.
        """

        try:
            qt_viewer = self.viewer.window._qt_viewer
        except Exception as e:
            print("Cannot access napari qt_viewer:", e)
            return

        # Hide the top layer-button bar: points / shapes / labels / delete etc.
        possible_layer_button_attrs = [
            "layerButtons",
            "layer_buttons",
            "buttons",
        ]

        for attr in possible_layer_button_attrs:
            try:
                obj = getattr(qt_viewer, attr, None)
                if obj is not None:
                    obj.setVisible(False)
            except Exception:
                pass

        # Hide the bottom viewer-button bar: console / grid / home / 2D-3D etc.
        possible_viewer_button_attrs = [
            "viewerButtons",
            "viewer_buttons",
            "viewerButtonsContainer",
        ]

        for attr in possible_viewer_button_attrs:
            try:
                obj = getattr(qt_viewer, attr, None)
                if obj is not None:
                    obj.setVisible(False)
            except Exception:
                pass

        # Extra compatibility for different napari versions.
        # Hide buttons by tooltip / objectName / text, but keep layer eye buttons.
        keywords_to_hide = [
            "new points layer",
            "new shapes layer",
            "new labels layer",
            "new image layer",
            "add points",
            "add shapes",
            "add labels",
            "delete selected layers",
            "delete layer",
            "console",
            "roll dimensions",
            "transpose dimensions",
            "grid mode",
            "home",
            "reset view",
            "toggle ndisplay",
            "2d/3d",
            "activity",
        ]

        try:
            buttons = qt_viewer.findChildren(QAbstractButton)

            for button in buttons:
                tooltip = button.toolTip().strip().lower()
                object_name = button.objectName().strip().lower()
                text = button.text().strip().lower()
                combined = f"{tooltip} {object_name} {text}"

                # Do not hide layer visibility buttons, otherwise the ROI / preview
                # rows may look broken in some napari versions.
                if "visibility" in combined or "visible" in combined:
                    continue

                if any(keyword in combined for keyword in keywords_to_hide):
                    button.setVisible(False)

        except Exception as e:
            print("Hide napari buttons warning:", e)

        print("Napari interface simplified: ROI and Camera Preview remain visible.")

    def simplify_layer_controls(self):
        allowed_labels = {
            "contrast limits",
            "gamma",
            "colormap",
        }

        image_control_labels = {
            "opacity",
            "blending",
            "contrast limits",
            "auto-contrast",
            "gamma",
            "colormap",
            "projection mode",
            "interpolation",
            "rendering",
            "depiction",
            "attenuation",
            "iso threshold",
            "plane",
            "shading",
        }

        try:
            qt_viewer = self.viewer.window._qt_viewer
            controls_container = qt_viewer.controls
        except Exception as e:
            print("Cannot access napari layer controls:", e)
            return

        labels = controls_container.findChildren(QLabel)

        for label in labels:
            text = label.text().strip().lower().replace(":", "")

            if text not in image_control_labels:
                continue

            should_show = text in allowed_labels
            self.set_control_row_visible(label, should_show)

        self.place_profile_controls_in_layer_controls()
        self.place_camera_adjust_controls_in_layer_controls()
        self.place_camera_status_controls_in_layer_controls()
        self.place_phase_analysis_controls_in_layer_controls()

    def set_control_row_visible(self, label, visible):
        parent = label.parentWidget()
        checked_parents = []

        for _ in range(5):
            if parent is None:
                break
            checked_parents.append(parent)
            parent = parent.parentWidget()

        for parent in checked_parents:
            layout = parent.layout()
            if layout is None:
                continue

            if isinstance(layout, QFormLayout):
                try:
                    row, role = layout.getWidgetPosition(label)
                except Exception:
                    row, role = -1, None

                if row >= 0:
                    for form_role in (
                        QFormLayout.LabelRole,
                        QFormLayout.FieldRole,
                        QFormLayout.SpanningRole,
                    ):
                        item = layout.itemAt(row, form_role)
                        if item is not None and item.widget() is not None:
                            item.widget().setVisible(visible)
                    return

            if isinstance(layout, QGridLayout):
                for i in range(layout.count()):
                    item = layout.itemAt(i)
                    if item is not None and item.widget() is label:
                        row, col, row_span, col_span = layout.getItemPosition(i)

                        for j in range(layout.count()):
                            item2 = layout.itemAt(j)
                            if item2 is None or item2.widget() is None:
                                continue

                            row2, col2, row_span2, col_span2 = layout.getItemPosition(j)
                            if row2 == row:
                                item2.widget().setVisible(visible)
                        return

            try:
                for i in range(layout.count()):
                    item = layout.itemAt(i)
                    if item is not None and item.widget() is label:
                        if parent is not None and parent is not self.viewer.window._qt_viewer.controls:
                            parent.setVisible(visible)
                        else:
                            label.setVisible(visible)
                        return
            except Exception:
                pass

        label.setVisible(visible)




    def place_phase_analysis_controls_in_layer_controls(self):
        self.add_widget_to_mwdriver_dock(self.phase_analysis_controls_widget)

    def init_mwdriver_controls_dock(self):
        self.mwdriver_controls_scroll = QScrollArea()
        self.mwdriver_controls_scroll.setObjectName("mwdriver_controls_scroll")
        self.mwdriver_controls_scroll.setWidgetResizable(True)
        self.mwdriver_controls_scroll.setMinimumWidth(300)

        self.mwdriver_controls_content = QWidget()
        self.mwdriver_controls_content.setObjectName("mwdriver_controls_content")
        self.mwdriver_controls_layout = QVBoxLayout()
        self.mwdriver_controls_layout.setContentsMargins(10, 10, 10, 10)
        self.mwdriver_controls_layout.setSpacing(12)
        self.mwdriver_controls_content.setLayout(self.mwdriver_controls_layout)
        self.mwdriver_controls_scroll.setWidget(self.mwdriver_controls_content)

        self.viewer.window.add_dock_widget(
            self.mwdriver_controls_scroll,
            name="MWdriver Controls",
            area="right",
        )

        self.add_widget_to_mwdriver_dock(self.camera_adjust_widget)
        self.add_widget_to_mwdriver_dock(self.profile_controls_widget)
        self.add_widget_to_mwdriver_dock(self.camera_status_widget)

    def add_widget_to_mwdriver_dock(self, widget):
        if widget is None or self.mwdriver_controls_layout is None:
            return
        if self.mwdriver_controls_layout.indexOf(widget) < 0:
            self.mwdriver_controls_layout.addWidget(
                widget,
                0,
                Qt.AlignLeft | Qt.AlignTop,
            )
        widget.setVisible(True)
        self.mwdriver_controls_content.adjustSize()

    def install_controls_resize_filter(self):
        """
        Install resize filters on napari layer-controls widget and its parent widgets.

        Dragging the splitter usually resizes an outer dock widget, not only
        qt_viewer.controls. Therefore we listen to qt_viewer.controls, its
        parent widgets, and the main window.
        """

        try:
            qt_viewer = self.viewer.window._qt_viewer
            self.controls_container = qt_viewer.controls
            self.controls_resize_filter_widgets = []

            widget = self.controls_container

            # Listen to qt_viewer.controls and several parent widgets.
            for _ in range(8):
                if widget is None:
                    break

                try:
                    widget.installEventFilter(self)
                    self.controls_resize_filter_widgets.append(widget)
                except Exception:
                    pass

                widget = widget.parentWidget()

            # Also listen to the main napari window.
            try:
                self.viewer.window._qt_window.installEventFilter(self)
                self.controls_resize_filter_widgets.append(self.viewer.window._qt_window)
            except Exception:
                pass

            print(
                "Controls resize filter installed on",
                len(self.controls_resize_filter_widgets),
                "widgets."
            )

            self.controls_resize_timer = QTimer(self)
            self.controls_resize_timer.setSingleShot(True)
            self.controls_resize_timer.timeout.connect(
                self.refresh_left_embedded_controls
            )
            self.controls_resize_timer.start(100)

        except Exception as e:
            print("Install controls resize filter warning:", e)

    def eventFilter(self, obj, event):
        """
        Update embedded widget sizes when the left dock area or main window changes size.
        """

        try:
            if (
                obj in self.controls_resize_filter_widgets
                and event.type() == QEvent.Resize
            ):
                if self.controls_resize_timer is not None:
                    self.controls_resize_timer.start(80)

        except Exception as e:
            print("Controls resize event warning:", e)

        return False

    def get_left_controls_available_width(self):
        """
        Get the real available width of the left controls area.

        qt_viewer.controls may not change width when the outer dock is resized,
        so this checks parent widgets and uses the largest reasonable width.
        """

        widths = []

        try:
            qt_viewer = self.viewer.window._qt_viewer
            widget = qt_viewer.controls

            for _ in range(8):
                if widget is None:
                    break

                try:
                    width = widget.width()
                    if width > 0:
                        widths.append(width)
                except Exception:
                    pass

                widget = widget.parentWidget()

        except Exception:
            pass

        if len(widths) == 0:
            return 320

        # Avoid using the whole main-window width.
        reasonable_widths = [w for w in widths if 150 <= w <= 800]

        if len(reasonable_widths) == 0:
            return max(widths)

        return max(reasonable_widths)

    def refresh_left_embedded_controls(self):
        """
        Recalculate all embedded controls after dragging the left/right splitter.
        """

        try:
            self.place_profile_controls_in_layer_controls()
            self.place_camera_adjust_controls_in_layer_controls()
            self.place_camera_status_controls_in_layer_controls()
            self.place_phase_analysis_controls_in_layer_controls()
        except Exception as e:
            print("Refresh left embedded controls warning:", e)

































    def start_stream(self):
        if self.is_streaming:
            return

        stream_mode = get_camera_stream_mode(
            streaming_allowed=self.streaming_allowed,
            preview_layer_visible=(
                self.layer is not None and self.layer.visible
            ),
            camera_mode=self.current_camera_mode,
            phase_capture_active=self.phase_capture_active,
            recalibration_active=self.recalibration_active,
        )
        if stream_mode == "stopped":
            return

        self.is_streaming = True

        self.worker_thread = QThread()
        self.worker = CameraWorker(
            cam=self.cam,
            display_downsample=self.display_downsample,
            phase_mode_getter=lambda: (
                self.phase_capture_active
                and self.current_camera_mode == "external"
            )
        )

        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)

        self.worker.frame_ready.connect(self.update_frame)
        self.worker.fps_ready.connect(self.update_fps)
        self.worker.status_ready.connect(self.update_status)

        self.worker_thread.start()

        self.update_stream_menu_actions()

        self.status_label.setText("Status: Starting...")

    def stop_stream(self):
        if not self.is_streaming:
            return

        self.is_streaming = False

        stop_worker_thread(self.worker, self.worker_thread)

        self.worker = None
        self.worker_thread = None

        self.update_stream_menu_actions()

        self.status_label.setText("Status: Stopped")
        self.fps_label.setText("FPS: --")

    def update_frame(self, raw_frame, display_frame, frame_meta=None):
        self.latest_raw_frame = raw_frame
        self.latest_display_frame = display_frame

        # In mixed mode, only save phase images from frames that arrive
        # while the camera is temporarily in external trigger mode.
        if (
            self.phase_capture_active
            and self.current_camera_mode == "external"
        ):
            self.handle_triggered_phase_frame(raw_frame, frame_meta=frame_meta)

            # Critical stability improvement:
            # During the 6 hardware-triggered phase frames, do NOT update napari display.
            # Napari contrast-limits redraws / layer updates / profile plotting can block the
            # GUI thread long enough for close-spaced trigger frames to be missed.
            # The raw frame has already been assigned above, so return immediately.
            return

        display_frame = self.apply_display_mask(display_frame)

        self.layer.data = display_frame

        self.update_intensity_profile()

    def update_fps(self, fps):
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def update_status(self, text):
        self.status_label.setText(text)

    def start_pico_listener(self):
        if self.pico_thread is not None:
            return

        self.pico_thread = QThread()
        self.pico_worker = PicoSerialWorker(
            port=self.pico_port,
            baudrate=self.pico_baudrate,
            reboot_pico=True
        )

        self.pico_worker.moveToThread(self.pico_thread)

        self.pico_thread.started.connect(self.pico_worker.run)

        self.pico_worker.phase_session_started.connect(self.start_phase_capture_session)
        self.pico_worker.phase_requested.connect(self.capture_phase_frame)
        self.pico_worker.phase_session_done.connect(self.on_phase_session_done)

        self.pico_worker.capture_requested.connect(self.request_manual_capture)

        self.pico_worker.status_ready.connect(self.update_pico_status)
        self.pico_worker.message_ready.connect(self.print_pico_message)


        self.pico_thread.start()

        print("Pico listener thread started.")

    def stop_pico_listener(self):
        try:
            if self.pico_worker is not None:
                self.pico_worker.stop()
        except Exception as e:
            print("Error while stopping Pico worker:", e)

        try:
            if self.pico_thread is not None:
                self.pico_thread.quit()
                self.pico_thread.wait(2000)
        except Exception as e:
            print("Error while stopping Pico thread:", e)

        self.pico_worker = None
        self.pico_thread = None

        print("Pico listener stopped.")

    def update_pico_status(self, text):
        print(text)

        try:
            self.pico_label.setText(text)
        except Exception:
            pass

    def print_pico_message(self, text):
        print(text)

        if "TRIGGER_POSITIONS_UPDATED_MS:" in text:
            try:
                payload = text.split("TRIGGER_POSITIONS_UPDATED_MS:", 1)[1].strip()
                parsed = ast.literal_eval(payload)
                if isinstance(parsed, (list, tuple)):
                    self.last_trigger_update_ack_positions_ms = [
                        int(round(value)) for value in parsed
                    ]
                    self.last_trigger_update_error_text = None
                    print(
                        "Parsed Pico trigger-update ACK:",
                        self.last_trigger_update_ack_positions_ms,
                    )
            except Exception as error:
                print("Failed to parse trigger-update ACK:", error)

        if "TRIGGER_POSITIONS_UPDATE_ERROR:" in text:
            self.last_trigger_update_error_text = text














    def find_latest_phase_analysis_session(self):
        latest_result = self.latest_phase_save_result
        if isinstance(latest_result, PhaseSaveResult) and latest_result.save_success:
            session_dir = latest_result.session_dir
            if (
                session_dir
                and os.path.basename(session_dir).startswith("phase_capture_")
                and os.path.isdir(session_dir)
                and any(
                    filename.endswith("_raw_stack.npy")
                    for filename in os.listdir(session_dir)
                )
            ):
                return os.path.abspath(session_dir)

        candidates = []
        if os.path.isdir(SAVE_DIR):
            for entry in os.scandir(SAVE_DIR):
                if not entry.is_dir() or not entry.name.startswith("phase_capture_"):
                    continue

                raw_stack_paths = [
                    os.path.join(entry.path, filename)
                    for filename in os.listdir(entry.path)
                    if filename.endswith("_raw_stack.npy")
                ]
                if raw_stack_paths:
                    candidates.append((
                        max(os.path.getmtime(path) for path in raw_stack_paths),
                        entry.path,
                    ))

        if not candidates:
            return None

        return os.path.abspath(max(candidates)[1])

    def run_latest_phase_analysis(self):
        session_dir = self.find_latest_phase_analysis_session()
        if session_dir is None:
            self.status_label.setText(
                "Status: No completed phase capture is available for analysis"
            )
            return

        self.run_phase_analysis_session(session_dir, "latest phase capture")

    def run_selected_phase_analysis(self):
        selected_layer = self.viewer.layers.selection.active
        if selected_layer not in self.phase_capture_time_series_layers:
            self.status_label.setText(
                "Status: Select a captured six-frame stack in the layer list first"
            )
            return

        session_dir = selected_layer.metadata.get("session_dir")
        raw_stack_path = selected_layer.metadata.get("raw_stack_path")
        if (
            not session_dir
            or not os.path.isdir(session_dir)
            or not raw_stack_path
            or not os.path.isfile(raw_stack_path)
        ):
            self.status_label.setText(
                "Status: Selected stack is still saving or has no saved session"
            )
            return

        self.run_phase_analysis_session(
            session_dir,
            f"selected stack {selected_layer.name}",
        )

    def run_phase_analysis_session(self, session_dir, description):
        if self.save_thread is not None:
            self.status_label.setText(
                "Status: Phase capture is still being saved; please wait"
            )
            return

        if not os.path.isfile(PHASE_ANALYSIS_SCRIPT):
            self.status_label.setText(
                f"Status: Phase analysis script not found: {PHASE_ANALYSIS_SCRIPT}"
            )
            return

        if (
            self.phase_analysis_process is not None
            and self.phase_analysis_process.poll() is None
        ):
            self.status_label.setText(
                "Status: Phase analysis viewer is already open"
            )
            return

        try:
            raw_stack_paths = sorted(
                os.path.join(session_dir, filename)
                for filename in os.listdir(session_dir)
                if filename.endswith("_raw_stack.npy")
            )
            if not raw_stack_paths:
                self.status_label.setText(
                    f"Status: No raw image stack found in {session_dir}"
                )
                return
            self.phase_analysis_source_stack_name = os.path.basename(
                raw_stack_paths[0]
            )

            self.phase_analysis_output_path = os.path.join(
                session_dir,
                "phase_analysis_results.npz",
            )
            if os.path.exists(self.phase_analysis_output_path):
                os.remove(self.phase_analysis_output_path)

            if self.view_phase_analysis_button is not None:
                self.view_phase_analysis_button.setEnabled(False)

            self.phase_analysis_process = subprocess.Popen(
                [
                    sys.executable,
                    PHASE_ANALYSIS_SCRIPT,
                    session_dir,
                    "--no-viewer",
                    "--export",
                    self.phase_analysis_output_path,
                ],
                cwd=SCRIPT_DIR,
            )
            self.status_label.setText(
                f"Status: Analyzing {description} in background: "
                f"{os.path.basename(session_dir)}"
            )
            print(f"Started phase analysis for {description}:", session_dir)
            QTimer.singleShot(200, self.check_phase_analysis_process)
        except Exception as error:
            self.phase_analysis_process = None
            self.status_label.setText(
                f"Status: Failed to start phase analysis: {error}"
            )

    def check_phase_analysis_process(self):
        process = self.phase_analysis_process
        if process is None:
            return

        return_code = process.poll()
        if return_code is None:
            QTimer.singleShot(200, self.check_phase_analysis_process)
            return

        self.phase_analysis_process = None
        if return_code != 0:
            self.status_label.setText(
                f"Status: Phase analysis failed with code {return_code}"
            )
            if self.view_phase_analysis_button is not None:
                self.view_phase_analysis_button.setEnabled(True)
            return

        output_path = self.phase_analysis_output_path
        if not output_path or not os.path.exists(output_path):
            self.status_label.setText(
                "Status: Phase analysis finished but produced no result file"
            )
            return

        try:
            with np.load(output_path) as result_file:
                self.phase_analysis_results = {
                    key: np.asarray(result_file[key])
                    for key in result_file.files
                }

            self.phase_analysis_view_active = True
            self.create_phase_analysis_controls(selected_key="wrapped")
            self.show_selected_phase_analysis_result(create_new_layer=True)
            self.view_phase_analysis_button.setText("Return to Camera Preview")
            self.status_label.setText(
                "Status: Phase analysis ready; use the controls below"
            )
            print("Phase analysis results loaded:", output_path)
        except Exception as error:
            self.status_label.setText(
                f"Status: Failed to load phase analysis results: {error}"
            )






    def request_pico_phase_capture_from_button(self):
        """
        Software button version of pressing GP11.
        It sends a command to Pico. Pico will then output:
        - TRIANGLE_START
        - GP15 triangle wave
        - GP12 hardware triggers
        - CAPTURE_PHASE_0 ... CAPTURE_PHASE_5
        - TRIANGLE_DONE
        """
        print("====================================")
        print("Start Phase Capture button clicked")
        print("pico_worker is None:", self.pico_worker is None)
        print("recalibration_active:", self.recalibration_active)
        print("====================================")

        if self.recalibration_active:
            self.status_label.setText(
                "Status: Recalibration is running; wait for it to finish"
            )
            return

        if self.save_thread is not None:
            self.status_label.setText(
                "Status: Previous save/check is still running; please wait"
            )
            self.phase_label.setText(
                "Phase Capture: Waiting for previous save/check to finish"
            )
            return

        if self.pico_worker is None:
            self.status_label.setText("Status: Pico not connected")
            self.pico_label.setText("Pico: Not connected")
            return

        positions = self.active_trigger_positions_ms
        if positions is None:
            positions = self.load_active_trigger_positions_from_json()

        if positions is not None:
            try:
                positions = self.build_formal_capture_positions_ms(positions)
                self.status_label.setText(
                    "Status: Sending Start Phase Capture with active calibrated triggers"
                )
                self.phase_label.setText(
                    "Phase Capture: Software button pressed"
                )
                self.queue_phase_capture_with_trigger_positions(positions)
                return
            except Exception as error:
                self.status_label.setText(
                    f"Status: Phase capture not started: {error}"
                )
                print("Start Phase Capture trigger-update warning:", error)
                return

        if self.selected_psi_algorithm_id != LEGACY_90_DEG:
            self.status_label.setText(
                "Status: Calibrate this PSI mode before capturing"
            )
            self.phase_label.setText(
                "Phase Capture: No calibration saved for selected mode"
            )
            return

        self.status_label.setText(
            "Status: Sending Start Phase Capture command to Pico"
        )
        self.phase_label.setText(
            "Phase Capture: Software button pressed"
        )
        # Positions for this capture are unknown (no active/loaded triggers),
        # so skip the phase-vs-voltage plot rather than showing wrong voltages.
        self.last_capture_trigger_positions_ms = None
        self.pico_worker.request_start_phase_from_gui()

    def build_formal_capture_positions_ms(self, trigger_positions_ms):
        return self.validate_trigger_positions_ms(trigger_positions_ms)

    def build_even_falling_edge_positions_ms(self):
        """
        Build six trigger times across the falling edge with guard margins.

        The first/last few milliseconds are intentionally avoided because
        camera arm latency and waveform boundary transitions can increase
        missing-frame risk.
        """
        positions = build_even_trigger_positions_ms(
            RECALIBRATION_CONFIG,
        )
        return self.validate_trigger_positions_ms(positions)

    def validate_trigger_positions_ms(self, trigger_positions_ms):
        return validate_recalibration_trigger_positions_ms(
            trigger_positions_ms,
            RECALIBRATION_CONFIG,
        )

    def enforce_minimum_trigger_spacing(
        self,
        trigger_positions_ms,
        min_spacing_ms=None,
    ):
        """
        Spread six trigger positions apart so no adjacent pair is closer than
        min_spacing_ms, while keeping them inside the falling-edge guard bounds.

        Interpolation-derived positions (from measured phase curvature) can
        otherwise collapse two points close together in steep regions of the
        phase curve, which is a deterministic hardware timing failure rather
        than random jitter, so it repeats even on identical retries.
        """
        if min_spacing_ms is None:
            min_spacing_ms = RECALIBRATION_CONFIG.safe_trigger_spacing_ms
        return enforce_minimum_trigger_spacing_ms(
            trigger_positions_ms,
            config=RECALIBRATION_CONFIG,
            camera_required_spacing_ms=self.cam.get_recommended_trigger_spacing_ms(
                exposure_s=self.exposure_s
            ),
            requested_spacing_ms=min_spacing_ms,
        )

    def compute_recalibrated_trigger_positions_ms(
        self,
        coarse_positions_ms,
        measured_phase_deg,
    ):
        return compute_coarse_refined_positions_ms(
            coarse_positions_ms,
            measured_phase_deg,
            self.get_recalibration_target_phase_deg(),
            config=RECALIBRATION_CONFIG,
            camera_required_spacing_ms=self.cam.get_recommended_trigger_spacing_ms(
                exposure_s=self.exposure_s
            ),
        )

    def validate_coarse_recalibration_measurement(self, measured_phase_deg):
        return validate_coarse_measurement(
            measured_phase_deg,
            RECALIBRATION_CONFIG,
        )

    def save_recalibrated_trigger_positions(
        self,
        coarse_positions_ms,
        measured_phase_deg,
        refined_positions_ms,
        coarse_session_dir,
    ):
        refined_positions = self.validate_trigger_positions_ms(refined_positions_ms)
        algorithm = get_psi_algorithm(self.selected_psi_algorithm_id)
        calibration = {
            "calibration_type": "six_point_recalibration",
            "algorithm_id": self.selected_psi_algorithm_id,
            "edge": "falling",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "coarse_session_dir": coarse_session_dir,
            "coarse_trigger_positions_ms": [
                int(value) for value in coarse_positions_ms
            ],
            "coarse_measured_phase_deg": [
                float(value) for value in measured_phase_deg
            ],
            "expected_phase_deg": [
                float(value) for value in self.get_recalibration_target_phase_deg()
            ],
            "trigger_positions_ms": [
                int(value) for value in refined_positions
            ],
            "note": (
                "Generated from one coarse six-point capture over the full "
                "falling edge, then inverted onto "
                f"{', '.join(str(int(value)) for value in algorithm['expected_phase_deg'])} deg."
            ),
        }

        calibrations = {}
        if os.path.exists(RECALIBRATION_TRIGGER_JSON_PATH):
            try:
                with open(
                    RECALIBRATION_TRIGGER_JSON_PATH,
                    "r",
                    encoding="utf-8",
                ) as file:
                    previous_payload = json.load(file)
                calibrations.update(previous_payload.get("calibrations", {}))
                if (
                    not calibrations
                    and "trigger_positions_ms" in previous_payload
                ):
                    calibrations[LEGACY_90_DEG] = previous_payload
            except (OSError, json.JSONDecodeError):
                pass
        calibrations[self.selected_psi_algorithm_id] = calibration
        payload = {"calibrations": calibrations}

        with open(
            RECALIBRATION_TRIGGER_JSON_PATH,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(payload, file, indent=4, ensure_ascii=False)

        print(
            "Saved recalibrated trigger positions:",
            RECALIBRATION_TRIGGER_JSON_PATH,
        )

        self.active_trigger_positions_ms = list(refined_positions)

    def queue_phase_capture_with_trigger_positions(self, trigger_positions_ms):
        if self.pico_worker is None:
            raise RuntimeError("Pico worker is not available.")

        positions = self.validate_trigger_positions_ms(trigger_positions_ms)
        self.last_capture_algorithm_id = self.selected_psi_algorithm_id

        required_spacing_ms = self.cam.get_recommended_trigger_spacing_ms(
            exposure_s=self.exposure_s,
        )
        actual_spacing_ms = min(np.diff(positions))
        print(
            f"Camera trigger spacing: minimum {actual_spacing_ms:g} ms; "
            f"recommended {required_spacing_ms} ms at "
            f"{self.exposure_s * 1000:g} ms exposure"
        )
        if actual_spacing_ms < required_spacing_ms:
            raise ValueError(
                f"Trigger gap {actual_spacing_ms:g} ms is below the camera-safe "
                f"{required_spacing_ms} ms spacing. Recalibrate or reduce exposure."
            )

        self.last_trigger_update_ack_positions_ms = None
        self.last_trigger_update_error_text = None

        self.pico_worker.request_set_trigger_positions_ms(positions)
        self.wait_for_trigger_update_ack(positions)
        self.pico_worker.request_start_phase_from_gui()

        self.last_capture_trigger_positions_ms = positions

        return positions

    def queue_or_defer_recalibration_phase_capture(
        self,
        trigger_positions_ms,
        status_text,
    ):
        positions = self.validate_trigger_positions_ms(trigger_positions_ms)

        if self.save_thread is not None:
            self.pending_recalibration_trigger_positions_ms = positions
            self.pending_recalibration_status_text = status_text

            self.status_label.setText(
                "Status: Waiting for previous save/check before next recalibration capture"
            )
            self.phase_label.setText(
                "Phase Capture: Recalibration queued, waiting for save worker"
            )

            print(
                "Deferred recalibration capture until save worker cleanup.",
                positions,
            )
            return "deferred"

        self.pending_recalibration_trigger_positions_ms = None
        self.pending_recalibration_status_text = None

        self.status_label.setText(status_text)
        self.queue_phase_capture_with_trigger_positions(positions)
        return "queued"

    def wait_for_trigger_update_ack(self, expected_positions_ms):
        deadline = time.monotonic() + RECALIBRATION_TRIGGER_UPDATE_ACK_TIMEOUT_S

        while time.monotonic() < deadline:
            QCoreApplication.processEvents()

            if self.last_trigger_update_error_text is not None:
                raise RuntimeError(
                    "Pico rejected trigger update: "
                    f"{self.last_trigger_update_error_text}"
                )

            ack_positions = self.last_trigger_update_ack_positions_ms
            if ack_positions is not None:
                expected = [int(round(value)) for value in expected_positions_ms]
                if ack_positions != expected:
                    raise RuntimeError(
                        "Pico trigger-update ACK mismatch: "
                        f"expected {expected}, got {ack_positions}"
                    )
                return

            time.sleep(0.01)

        raise TimeoutError(
            "Timed out waiting for Pico trigger-update ACK "
            f"({RECALIBRATION_TRIGGER_UPDATE_ACK_TIMEOUT_S:.1f} s)."
        )

    def compute_iterative_refined_positions_ms(
        self,
        current_positions_ms,
        measured_phase_deg,
    ):
        return compute_iterative_refined_positions(
            current_positions_ms,
            measured_phase_deg,
            self.get_recalibration_target_phase_deg(),
            config=RECALIBRATION_CONFIG,
            camera_required_spacing_ms=self.cam.get_recommended_trigger_spacing_ms(
                exposure_s=self.exposure_s
            ),
        )

    def abort_recalibration(self, status_text):
        self.recalibration_active = False
        self.sync_stream_with_active_layer()
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

        self.status_label.setText(status_text)
        self.phase_label.setText("Phase Capture: Recalibration aborted")

    def should_auto_restart_recalibration(self, best_candidate):
        return should_auto_restart_candidate(
            best_candidate,
            RECALIBRATION_CONFIG,
        )

    def purge_current_recalibration_session_artifacts(self):
        candidate_dirs = []
        for candidate in self.recalibration_candidates:
            if not isinstance(candidate, dict):
                continue
            session_dir = candidate.get("session_dir")
            if isinstance(session_dir, str) and session_dir:
                candidate_dirs.append(os.path.abspath(session_dir))

        if self.recalibration_coarse_session_dir:
            candidate_dirs.append(os.path.abspath(self.recalibration_coarse_session_dir))

        for session_dir in sorted(set(candidate_dirs)):
            if not os.path.isdir(session_dir):
                continue
            try:
                shutil.rmtree(session_dir)
                print(
                    "Removed first-round recalibration artifact before auto-restart:",
                    session_dir,
                )
            except Exception as error:
                print(
                    "Could not remove first-round recalibration artifact:",
                    session_dir,
                    error,
                )

    def restart_recalibration_after_failed_verification(self, best_candidate):
        if self.recalibration_auto_restart_count >= RECALIBRATION_CONFIG.max_auto_restarts:
            return False

        try:
            coarse_positions = self.build_even_falling_edge_positions_ms()
            coarse_positions = self.validate_trigger_positions_ms(coarse_positions)
        except Exception as error:
            self.abort_recalibration(
                f"Status: Auto-recalibration restart failed building coarse map: {error}"
            )
            return True

        self.purge_current_recalibration_session_artifacts()

        self.recalibration_auto_restart_count += 1
        self.recalibration_stage = "coarse_capture"
        self.recalibration_coarse_positions_ms = coarse_positions
        self.recalibration_target_positions_ms = None
        self.recalibration_current_positions_ms = coarse_positions
        self.recalibration_refinement_round = 0
        self.recalibration_incomplete_verification_retries = 0
        self.recalibration_verification_capture_count = 0
        self.recalibration_pending_pass_candidate = None
        self.recalibration_coarse_session_dir = None
        self.recalibration_candidates = []

        self.phase_label.setText("Phase Capture: Recalibration auto-restart")
        self.status_label.setText(
            "Status: First-round calibration artifacts removed; "
            "auto-restarting recalibration"
        )

        try:
            self.queue_phase_capture_with_trigger_positions(coarse_positions)
        except Exception as error:
            self.abort_recalibration(
                f"Status: Auto-recalibration restart failed to queue capture: {error}"
            )

        return True

    def retry_incomplete_verification_capture(self):
        if (
            not self.recalibration_active
            or self.recalibration_stage != "verification_capture"
            or self.recalibration_current_positions_ms is None
        ):
            return

        try:
            # An incomplete capture with these exact positions can be a
            # deterministic hardware timing collision (two triggers too close
            # together) rather than random jitter, so retrying unchanged
            # positions would repeat the same failure. Widen the spacing a
            # bit more on each retry before trying again.
            escalated_spacing_ms = RECALIBRATION_CONFIG.safe_trigger_spacing_ms + (
                self.recalibration_incomplete_verification_retries
                * RECALIBRATION_CONFIG.safe_trigger_spacing_ms
            )
            retry_positions = self.enforce_minimum_trigger_spacing(
                self.recalibration_current_positions_ms,
                min_spacing_ms=escalated_spacing_ms,
            )
            self.recalibration_current_positions_ms = retry_positions

            self.queue_or_defer_recalibration_phase_capture(
                retry_positions,
                "Status: Retrying incomplete verification capture "
                f"{self.recalibration_incomplete_verification_retries}/"
                f"{RECALIBRATION_CONFIG.max_incomplete_verification_retries}",
            )
        except Exception as error:
            self.abort_recalibration(
                f"Status: Recalibration verification retry failed: {error}"
            )
    def build_recalibration_candidate(self, session_dir, measured_phase_deg):
        return score_recalibration_candidate(
            session_dir=session_dir,
            measured_phase_deg=measured_phase_deg,
            target_phase_deg=self.get_recalibration_target_phase_deg(),
            trigger_positions_ms=self.recalibration_current_positions_ms,
            config=RECALIBRATION_CONFIG,
        )

    def keep_only_recalibration_candidate(self, selected_candidate):
        selected_session_dir = os.path.abspath(selected_candidate["session_dir"])

        for candidate in self.recalibration_candidates:
            candidate_session_dir = os.path.abspath(candidate["session_dir"])
            if candidate_session_dir == selected_session_dir:
                continue

            if not os.path.isdir(candidate_session_dir):
                continue

            try:
                shutil.rmtree(candidate_session_dir)
                print("Removed rejected recalibration session:", candidate_session_dir)
            except Exception as error:
                print(
                    "Could not remove rejected recalibration session:",
                    candidate_session_dir,
                    error,
                )

        self.recalibration_candidates = [selected_candidate]

    def finalize_recalibration_candidate(self, selected_candidate, fallback_used):
        self.keep_only_recalibration_candidate(selected_candidate)

        selected_positions = self.validate_trigger_positions_ms(
            selected_candidate["trigger_positions_ms"]
        )

        self.save_recalibrated_trigger_positions(
            coarse_positions_ms=self.recalibration_coarse_positions_ms,
            measured_phase_deg=selected_candidate["measured_phase_deg"],
            refined_positions_ms=selected_positions,
            coarse_session_dir=self.recalibration_coarse_session_dir,
        )

        try:
            self.last_trigger_update_ack_positions_ms = None
            self.last_trigger_update_error_text = None
            self.pico_worker.request_set_trigger_positions_ms(selected_positions)
            self.wait_for_trigger_update_ack(selected_positions)
        except Exception as error:
            print(
                "Best-session trigger positions were saved but could not be "
                f"applied to Pico: {error}"
            )

        self.recalibration_active = False
        self.sync_stream_with_active_layer()
        self.recalibration_stage = None
        self.recalibration_current_positions_ms = None
        self.recalibration_target_positions_ms = None
        self.recalibration_refinement_round = 0
        self.recalibration_incomplete_verification_retries = 0
        self.recalibration_pending_pass_candidate = None
        self.recalibration_verification_capture_count = 0
        self.recalibration_auto_restart_count = 0

    def start_recalibration_from_button(self):
        print("====================================")
        print("Recalibrate button clicked")
        print("pico_worker is None:", self.pico_worker is None)
        print("phase_capture_active:", self.phase_capture_active)
        print("save_thread is None:", self.save_thread is None)
        print("====================================")

        if self.pico_worker is None:
            self.status_label.setText("Status: Pico not connected")
            self.phase_label.setText("Phase Capture: Recalibration aborted")
            return

        if self.phase_capture_active:
            self.status_label.setText(
                "Status: Phase capture is active; wait before recalibration"
            )
            return

        if self.save_thread is not None:
            self.status_label.setText(
                "Status: Save worker busy; wait before recalibration"
            )
            return

        if self.recalibration_active:
            self.status_label.setText(
                "Status: Recalibration already running"
            )
            return

        try:
            coarse_positions = self.build_even_falling_edge_positions_ms()
            coarse_positions = self.validate_trigger_positions_ms(coarse_positions)
        except Exception as error:
            self.status_label.setText(
                f"Status: Cannot build coarse trigger map: {error}"
            )
            return

        self.recalibration_active = True
        self.sync_stream_with_active_layer()
        self.recalibration_stage = "coarse_capture"
        self.recalibration_coarse_positions_ms = coarse_positions
        self.recalibration_target_positions_ms = None
        self.recalibration_current_positions_ms = coarse_positions
        self.recalibration_refinement_round = 0
        self.recalibration_incomplete_verification_retries = 0
        self.recalibration_verification_capture_count = 0
        self.recalibration_pending_pass_candidate = None
        self.recalibration_coarse_session_dir = None
        self.recalibration_candidates = []
        self.recalibration_auto_restart_count = 0

        self.phase_label.setText("Phase Capture: Recalibration coarse capture")
        self.status_label.setText(
            "Status: Recalibration stage 1/2 - coarse six-point capture"
        )

        try:
            self.queue_phase_capture_with_trigger_positions(coarse_positions)
        except Exception as error:
            self.recalibration_active = False
            self.sync_stream_with_active_layer()
            self.recalibration_stage = None
            self.status_label.setText(
                f"Status: Recalibration start failed: {error}"
            )
            self.phase_label.setText("Phase Capture: Recalibration failed")
            return


    def request_manual_capture(self):
        print("====================================")
        print("request_manual_capture() called")
        print("is_streaming:", self.is_streaming)
        print("latest_raw_frame is None:", self.latest_raw_frame is None)
        print("manual_capture_in_progress:", self.manual_capture_in_progress)
        print("====================================")

        if self.manual_capture_in_progress:
            print("Manual capture ignored: previous capture still running.")
            self.status_label.setText("Status: Manual capture already running")
            return

        if self.latest_raw_frame is None:
            print("No frame available for manual capture.")
            self.status_label.setText("Status: No frame available for manual capture")
            return

        self.manual_capture_in_progress = True
        self.manual_capture_frames = []
        self.manual_capture_index = 0

        self.status_label.setText("Status: Manual capture started")
        print("Manual capture started.")

        self.collect_manual_frame()

    def collect_manual_frame(self):
        if self.latest_raw_frame is None:
            print("No latest_raw_frame during manual capture.")
            self.finish_manual_capture()
            return

        frame = self.latest_raw_frame.copy()
        self.manual_capture_frames.append(frame)

        self.manual_capture_index += 1

        self.status_label.setText(
            f"Status: Manual captured {self.manual_capture_index}/{MANUAL_CAPTURE_COUNT}"
        )

        print(
            f"Manual captured frame "
            f"{self.manual_capture_index}/{MANUAL_CAPTURE_COUNT}"
        )

        if self.manual_capture_index < MANUAL_CAPTURE_COUNT:
            QTimer.singleShot(
                MANUAL_CAPTURE_INTERVAL_MS,
                self.collect_manual_frame
            )
        else:
            self.finish_manual_capture()

    def finish_manual_capture(self):
        frames = self.manual_capture_frames.copy()

        self.manual_capture_frames = []
        self.manual_capture_index = 0
        self.manual_capture_in_progress = False

        if len(frames) == 0:
            self.status_label.setText("Status: Manual capture failed - no frames")
            return

        self.status_label.setText("Status: Saving manual frames...")
        print(f"Manual capture finished. Saving {len(frames)} frames...")

        self.start_save_manual_frames_worker(frames)

    def start_save_manual_frames_worker(self, frames):
        if self.save_thread is not None:
            print("Save worker is already running.")
            self.status_label.setText("Status: Save worker already running")
            return

        self.save_thread = QThread()
        self.save_worker = FrameSaveWorker(
            frames=frames,
            save_dir=SAVE_DIR,
            prefix="manual_capture"
        )

        self.save_worker.moveToThread(self.save_thread)

        self.save_thread.started.connect(self.save_worker.run)
        self.save_worker.status_ready.connect(self.update_status)
        self.save_worker.finished.connect(self.on_manual_save_finished)
        self.save_worker.finished.connect(self.save_thread.quit)
        self.save_thread.finished.connect(self.cleanup_save_worker)

        self.save_thread.start()

    def on_manual_save_finished(self, text):
        print(text)
        self.status_label.setText(f"Status: {text}")

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