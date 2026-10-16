"""Phase-capture lifecycle methods mixed into the desktop camera viewer."""

import copy
import json
import os
import time
from pathlib import Path

import numpy as np
from qtpy.QtCore import QThread, QTimer

from mwdriver.capture.phase_capture_frame_filter import (
    decide_framestamp_phase_slot,
    decide_native_phase_frame,
    decide_phase_frame_assignment,
    decide_triangle_done_action,
    select_queued_phase_slots,
    should_accept_phase_frame,
)
from mwdriver.capture.phase_capture_timestamps import (
    build_phase_timestamp_payload,
    normalize_frame_framestamp,
)
from mwdriver.capture.phase_capture_storage import PhaseSaveResult
from mwdriver.capture.phase_capture_worker import PhaseSaveWorker
from mwdriver.domain.phase_psi import get_algorithm as get_psi_algorithm
from mwdriver.domain.recalibration import (
    DEFAULT_RECALIBRATION_CONFIG,
    decide_incomplete_capture_action,
    decide_verification_capture_action,
    is_strict_recalibration_pass,
    select_best_recalibration_candidate,
)


PHASE_CAPTURE_COUNT = DEFAULT_RECALIBRATION_CONFIG.frame_count
RECALIBRATION_CONFIG = DEFAULT_RECALIBRATION_CONFIG
EXTERNAL_TRIGGER_EDGE = "rise"
TRIANGLE_WAVE_SUPPLY_VOLTAGE = 3.3
MODULE_DIR = str(Path(__file__).resolve().parents[2])
SAVE_DIR = os.path.join(MODULE_DIR, "captured_frames")

def triangle_wave_falling_edge_voltage(trigger_position_ms):
    """
    Approximate triangle-wave analog voltage at a falling-edge trigger time.

    Mirrors the Pico's linear falling-edge duty ramp so the measured phase
    of each captured frame can be plotted against the waveform voltage that
    was present when the frame was triggered.
    """
    fraction = 1.0 - (
        (float(trigger_position_ms) + 1.0)
        / float(RECALIBRATION_CONFIG.falling_edge_ms)
    )
    fraction = max(0.0, min(1.0, fraction))
    return TRIANGLE_WAVE_SUPPLY_VOLTAGE * fraction


class PhaseCaptureController:
    def start_phase_capture_session(self):
        self.last_capture_algorithm_id = self.selected_psi_algorithm_id
        print("====================================")
        print("Phase capture session started")
        print("Current camera mode:", self.current_camera_mode)
        print("save_thread is None:", self.save_thread is None)
        print("====================================")

        if self.save_thread is not None:
            print(
                "Phase capture ignored: save worker is still running. "
                "Starting a new capture now would make this run unsavable."
            )
            self.phase_label.setText(
                "Phase Capture: Ignored while previous save is still running"
            )
            self.status_label.setText(
                "Status: Ignored TRIANGLE_START because save worker is busy"
            )
            return

        self.phase_capture_active = True
        self.sync_stream_with_active_layer()
        self.phase_capture_frames = [None] * PHASE_CAPTURE_COUNT
        self.phase_capture_times = [None] * PHASE_CAPTURE_COUNT
        self.phase_capture_count = 0
        self.phase_capture_start_time = time.monotonic()
        self.phase_capture_mode_enter_time_s = time.monotonic()
        self.pending_phase_indices = []
        self.unassigned_triggered_frames = []

        self.phase_label.setText(
            "Phase Capture: Started, switching to external trigger"
        )
        self.status_label.setText(
            "Status: Triangle started, switching camera to external trigger"
        )

        # Important: Pico sends TRIANGLE_START before the rising edge.
        # This gives the PC time to switch the camera from live view to
        # external trigger mode before the 6 phase pulses arrive.
        self.enter_external_trigger_mode_for_phase_capture()
    def enter_external_trigger_mode_for_phase_capture(self):
        """
        Temporarily switch camera from live view/internal trigger mode
        to external trigger mode. The camera then waits for GP12 pulses.
        """
        if self.switching_camera_mode:
            print("Camera mode switch already in progress.")
            return

        self.switching_camera_mode = True

        print("====================================")
        print("Switching camera to external trigger mode...")
        print("====================================")

        was_streaming = self.is_streaming

        try:
            if was_streaming:
                self.stop_stream()
        except Exception as e:
            print("Stop stream before external trigger warning:", e)

        try:
            if self.cam is not None:
                self.cam.stop_acquisition()
        except Exception as e:
            print("stop_acquisition before external trigger warning:", e)

        try:
            self.cam.set_exposure(self.exposure_s)
            print(
                f"Phase capture exposure kept at the selected setting: "
                f"{self.exposure_s * 1000:.1f} ms"
            )
        except Exception as e:
            print("Phase exposure setup warning:", e)

        try:
            self.cam.set_trigger_mode("ext")
            print("Camera trigger mode set to: ext")
        except Exception as e:
            print("ERROR: set_trigger_mode('ext') failed:", e)
            print("The camera may not support external trigger through this API.")

        try:
            self.cam.setup_ext_trigger(EXTERNAL_TRIGGER_EDGE)
            print(f"External trigger edge set to: {EXTERNAL_TRIGGER_EDGE}")
        except Exception as e:
            print("External trigger edge setup warning:", e)

        try:
            self.cam.start_acquisition(1)
            print("Camera acquisition restarted in external trigger mode.")
        except Exception as e:
            print("ERROR: start_acquisition in external trigger mode failed:", e)

        self.current_camera_mode = "external"

        try:
            self.sync_stream_with_active_layer()
        except Exception as e:
            print("Start capture reader in external trigger mode warning:", e)

        self.switching_camera_mode = False

        self.phase_label.setText(
            "Phase Capture: External trigger ready, waiting for phase 0-5"
        )
        self.status_label.setText(
            "Status: External trigger mode ready, waiting for GP12 pulses"
        )
    def clear_camera_buffer(self):
        """
        Drain any stale frames after changing camera trigger mode.
        This reduces the chance that an old live-view frame is saved as phase 0.
        """
        try:
            for _ in range(20):
                image_range = self.cam.get_new_images_range()
                if image_range is None:
                    break
                _ = self.cam.read_newest_image()
        except Exception as e:
            print("Clear camera buffer warning:", e)
    def assign_phase_frame(self, phase_index, raw_frame, source_text="hardware-triggered frame", frame_meta=None):
        """
        Store one raw frame into phase_capture_frames[phase_index].

        This helper is used by both possible timing orders:
        1) serial marker arrives first, then camera frame arrives;
        2) camera frame arrives first, then serial marker arrives.
        """
        decision = decide_phase_frame_assignment(
            self.phase_capture_active,
            phase_index,
            self.phase_capture_frames,
            phase_count=PHASE_CAPTURE_COUNT,
        )
        if decision.status == "inactive":
            return False
        if decision.status == "invalid_index":
            print("Invalid phase index for assignment:", phase_index)
            return False
        if decision.status == "duplicate":
            print(f"Phase {phase_index} already has a frame, ignoring duplicate assignment.")
            return False

        self.phase_capture_frames[phase_index] = raw_frame.copy()

        if self.phase_capture_start_time is not None:
            host_time_s = time.monotonic()
            self.phase_capture_times[phase_index] = (
                host_time_s - self.phase_capture_start_time
            )
        else:
            host_time_s = time.monotonic()
            self.phase_capture_times[phase_index] = None

        if self.phase_capture_timestamp_clock_hz is None:
            try:
                self.phase_capture_timestamp_clock_hz = self.cam.get_timestamp_clock_frequency()
            except Exception:
                self.phase_capture_timestamp_clock_hz = None

        self.phase_capture_frame_timestamps[phase_index] = build_phase_timestamp_payload(
            frame_meta if frame_meta is not None else raw_frame,
            host_time_s=host_time_s,
            phase_index=phase_index,
        )
        self.phase_capture_frame_timestamps[phase_index]["timestamp_clock_frequency_hz"] = (
            self.phase_capture_timestamp_clock_hz
        )

        self.phase_capture_count += 1

        self.phase_label.setText(
            f"Phase Capture: {source_text} {phase_index} "
            f"({self.phase_capture_count}/6)"
        )
        self.status_label.setText(
            f"Status: Phase {phase_index} assigned from {source_text} "
            f"({self.phase_capture_count}/6)"
        )

        print("====================================")
        print(f"Frame assigned for phase {phase_index} from {source_text}")
        print("phase_capture_count:", self.phase_capture_count)
        print("pending_phase_indices:", self.pending_phase_indices)
        print("unassigned_triggered_frames:", len(self.unassigned_triggered_frames))
        print("phase_times:", self.phase_capture_times)
        print("phase_frame_timestamps:", self.phase_capture_frame_timestamps)
        print("====================================")

        if self.phase_capture_count == PHASE_CAPTURE_COUNT:
            self.finish_phase_capture_session()

        return True
    def handle_triggered_phase_frame(self, raw_frame, frame_meta=None):
        """
        Called from update_frame() whenever the camera sends a new frame while
        phase capture is active and the camera is in external trigger mode.

        If the camera supplies a valid framestamp, assign the frame to the
        corresponding phase slot instead of relying solely on arrival order.
        """
        if not self.phase_capture_active:
            return

        if self.current_camera_mode != "external":
            return

        if self.phase_capture_count >= PHASE_CAPTURE_COUNT:
            print("Extra external-trigger frame received after 6/6; ignoring.")
            return

        native_decision = decide_native_phase_frame(
            frame_meta,
            expected_acquisition_id=getattr(
                self.cam,
                "acquisition_id",
                object(),
            ),
            phase_count=PHASE_CAPTURE_COUNT,
        )
        if native_decision.status == "wrong_trigger_mode":
            print("Ignoring queued live-view frame during phase capture.")
            return
        if native_decision.status == "stale_acquisition":
            print("Ignoring frame from an earlier camera acquisition.")
            return
        if native_decision.status == "invalid_index":
            acquisition_index = frame_meta.get("acquisition_index")
            print(
                f"Unexpected native acquisition index {acquisition_index}; ignoring."
            )
            return
        if native_decision.status == "accepted":
            self.assign_phase_frame(
                phase_index=native_decision.phase_index,
                raw_frame=raw_frame,
                source_text="native external frame",
                frame_meta=frame_meta,
            )
            return

        now_s = time.monotonic()
        if not should_accept_phase_frame(
            phase_capture_count=self.phase_capture_count,
            frame_arrival_time_s=now_s,
            phase_capture_mode_enter_time_s=self.phase_capture_mode_enter_time_s,
        ):
            print(
                "Ignoring stale frame during initial external-trigger grace window: "
                f"count={self.phase_capture_count}, now_s={now_s:.3f}, "
                f"mode_enter_s={self.phase_capture_mode_enter_time_s:.3f}"
            )
            return

        phase_index = self.phase_capture_count
        framestamp = None
        try:
            frame_info = frame_meta if frame_meta is not None else raw_frame
            if isinstance(frame_info, dict):
                frame_info = frame_info.get("info", frame_info)
            else:
                frame_info = getattr(frame_info, "info", frame_info)
            framestamp = normalize_frame_framestamp(frame_info)
        except Exception:
            framestamp = None

        framestamp_decision = decide_framestamp_phase_slot(
            framestamp,
            phase_count=PHASE_CAPTURE_COUNT,
        )
        if framestamp_decision.status == "valid":
            rounded = framestamp_decision.rounded_framestamp
            target_index = framestamp_decision.phase_index

            # Assign any earlier queued frames to lower empty phase slots
            # before assigning the current valid framestamp frame.
            queued_slots = select_queued_phase_slots(
                self.phase_capture_frames,
                queued_frame_count=len(self.unassigned_triggered_frames),
                before_slot=target_index,
            )
            for slot in queued_slots:
                if not self.unassigned_triggered_frames:
                    break
                queued_frame, queued_meta = self.unassigned_triggered_frames.pop(0)
                self.assign_phase_frame(
                    phase_index=slot,
                    raw_frame=queued_frame,
                    source_text=f"queued frame assigned to phase {slot}",
                    frame_meta=queued_meta,
                )

            if self.phase_capture_frames[target_index] is None:
                phase_index = target_index
                source_text = f"framestamp external frame ({rounded})"
            else:
                print(
                    f"Frame with framestamp {rounded} already assigned; ignoring duplicate/stale frame."
                )
                return
        elif framestamp_decision.status == "out_of_range":
            print(
                f"Camera framestamp {framestamp} out of expected range; "
                "holding frame until a valid phase framestamp arrives."
            )
            self.unassigned_triggered_frames.append((raw_frame.copy(), frame_meta))
            return
        elif self.phase_capture_count == 0:
            print(
                "No phase framestamp on first external-trigger frame; "
                "holding frame until a valid phase framestamp arrives."
            )
            self.unassigned_triggered_frames.append((raw_frame.copy(), frame_meta))
            return
        else:
            source_text = "arrival-order external frame"

        self.assign_phase_frame(
            phase_index=phase_index,
            raw_frame=raw_frame,
            source_text=source_text,
            frame_meta=frame_meta
        )
    def return_to_live_view_mode(self):
        """
        Switch camera back to internal trigger / live view mode after
        the 6 externally triggered phase frames have been captured.
        """
        if self.switching_camera_mode:
            print("Camera mode switch already in progress.")
            return

        self.switching_camera_mode = True

        print("====================================")
        print("Switching camera back to live view mode...")
        print("====================================")

        was_streaming = self.is_streaming

        try:
            if was_streaming:
                self.stop_stream()
        except Exception as e:
            print("Stop stream before live view warning:", e)

        try:
            if self.cam is not None:
                self.cam.stop_acquisition()
        except Exception as e:
            print("stop_acquisition before live view warning:", e)

        try:
            self.cam.set_trigger_mode("int")
            print("Camera trigger mode set to: int")
        except Exception as e:
            print("set_trigger_mode('int') warning:", e)

        try:
            self.cam.start_acquisition()
            print("Camera acquisition restarted in live view mode.")
        except Exception as e:
            print("start_acquisition in live view mode warning:", e)

        self.current_camera_mode = "live"

        try:
            self.sync_stream_with_active_layer()
        except Exception as e:
            print("Resume permitted live preview warning:", e)

        self.switching_camera_mode = False

        self.status_label.setText("Status: Returned to live view")
        self.phase_label.setText("Phase Capture: Live view resumed")
    def capture_phase_frame(self, phase_index):
        """
        Receive CAPTURE_PHASE_x from Pico.

        In the current timing-stable Pico program, all six GP12 hardware
        triggers are generated first. CAPTURE_PHASE_0 ... CAPTURE_PHASE_5
        are printed only after the timing-critical triangle waveform has
        finished.

        Therefore:
            - these markers are logging/status information only;
            - they must never start a new phase-capture session;
            - frame assignment remains based on external-frame arrival order;
            - TRIANGLE_START is the only message allowed to start a session.
        """

        print("====================================")
        print(f"capture_phase_frame({phase_index}) marker received")
        print("phase_capture_active:", self.phase_capture_active)
        print("current_camera_mode:", self.current_camera_mode)
        print("phase_capture_count:", self.phase_capture_count)
        print("pending_phase_indices:", self.pending_phase_indices)
        print(
            "unassigned_triggered_frames:",
            len(self.unassigned_triggered_frames)
        )
        print(
            "Note: marker is logging only; "
            "frames are assigned by arrival order."
        )
        print("====================================")

        if phase_index < 0 or phase_index >= PHASE_CAPTURE_COUNT:
            print("Invalid phase marker index:", phase_index)
            self.status_label.setText(
                f"Status: Invalid phase marker {phase_index}"
            )
            return

        # The real six-frame session may already have finished before these
        # delayed marker messages arrive. Ignore them rather than incorrectly
        # entering external-trigger mode for a second empty capture.
        if not self.phase_capture_active:
            print(
                f"Late phase marker {phase_index} received after the real "
                "capture session completed; marker ignored."
            )
            return

        # If capture is still active, the marker remains informational only.
        self.phase_label.setText(
            f"Phase Capture: marker phase {phase_index} received"
        )
        self.status_label.setText(
            f"Status: Marker phase {phase_index} received; "
            "hardware-triggered frames are assigned by arrival order"
        )
    def on_phase_session_done(self):
        """
        Handle TRIANGLE_DONE from Pico.

        The sixth camera frame can be delivered before TRIANGLE_DONE reaches
        the PC. In that normal successful path, finish_phase_capture_session()
        has already:
            - copied all six frames;
            - reset phase_capture_count to zero;
            - set phase_capture_active to False;
            - started the save worker;
            - returned the camera to live view.

        Therefore phase_capture_count == 0 at TRIANGLE_DONE does not
        necessarily mean that zero frames were captured. The active-session
        flag must be checked first.
        """

        print("TRIANGLE_DONE received from Pico.")

        completion_action = decide_triangle_done_action(
            self.phase_capture_active,
            self.phase_capture_count,
            phase_count=PHASE_CAPTURE_COUNT,
        )

        # Normal successful path: the six-frame acquisition was finalized
        # before this serial message arrived.
        if completion_action == "finalized":
            print(
                "TRIANGLE_DONE arrived after the six-frame capture "
                "had already been finalized."
            )

            if self.save_thread is not None:
                self.phase_label.setText(
                    "Phase Capture: Triangle done, 6/6 captured, saving"
                )
                self.status_label.setText(
                    "Status: Triangle done, 6/6 captured, "
                    "save worker running"
                )
            else:
                self.phase_label.setText(
                    "Phase Capture: Triangle done, capture completed"
                )
                self.status_label.setText(
                    "Status: Triangle done, phase capture completed"
                )

            return

        # A real capture session is still active. Give delayed camera frames
        # time to arrive only when fewer than six frames are currently present.
        if completion_action == "await_late_frames":
            self.phase_label.setText(
                f"Phase Capture: Triangle done, only "
                f"{self.phase_capture_count}/6 captured"
            )
            self.status_label.setText(
                f"Status: Triangle done, phase capture incomplete "
                f"{self.phase_capture_count}/6"
            )

            print(
                "TRIANGLE_DONE arrived while phase capture was still active."
            )
            print(
                "Current captured frame count:",
                self.phase_capture_count
            )

            QTimer.singleShot(
                6000,
                self.check_phase_capture_after_done
            )
            return

        # Defensive branch. Normally assign_phase_frame() finalizes the
        # session immediately when the sixth frame is assigned.
        self.phase_label.setText(
            "Phase Capture: Triangle done, 6/6 captured"
        )
        self.status_label.setText(
            "Status: Triangle done, 6/6 captured"
        )
        self.finish_phase_capture_session()
    def check_phase_capture_after_done(self):
        if not self.phase_capture_active:
            return

        if self.phase_capture_count == PHASE_CAPTURE_COUNT:
            self.finish_phase_capture_session()
            return

        # Late frame salvage:
        # If some externally-triggered frames arrived without valid framestamp
        # and were queued, assign them to the remaining empty phase slots before
        # declaring the session incomplete.
        if self.unassigned_triggered_frames:
            missing_slots = select_queued_phase_slots(
                self.phase_capture_frames,
                queued_frame_count=len(self.unassigned_triggered_frames),
            )

            print(
                "Attempting late queued-frame salvage after TRIANGLE_DONE.",
                "queued=", len(self.unassigned_triggered_frames),
                "missing=", missing_slots,
            )

            for slot in missing_slots:
                if not self.unassigned_triggered_frames:
                    break

                if self.phase_capture_frames[slot] is None:
                    queued_frame, queued_meta = self.unassigned_triggered_frames.pop(0)
                    self.assign_phase_frame(
                        phase_index=slot,
                        raw_frame=queued_frame,
                        source_text=f"late queued external frame {slot}",
                        frame_meta=queued_meta,
                    )

                    if not self.phase_capture_active:
                        # assign_phase_frame() may have finished and reset the session.
                        return

            if self.phase_capture_count == PHASE_CAPTURE_COUNT:
                self.finish_phase_capture_session()
                return

        missing = [
            i for i, frame in enumerate(self.phase_capture_frames)
            if frame is None
        ]

        print("Phase capture still incomplete after triangle done.")
        print("Missing:", missing)
        print("Pending:", self.pending_phase_indices)

        self.phase_label.setText(
            f"Phase Capture: Incomplete after triangle done, missing {missing}"
        )
        self.status_label.setText(
            f"Status: Incomplete hardware-trigger capture, missing {missing}"
        )

        retry_verification_capture = False
        incomplete_capture_decision = decide_incomplete_capture_action(
            recalibration_active=self.recalibration_active,
            recalibration_stage=self.recalibration_stage,
            incomplete_retry_count=(
                self.recalibration_incomplete_verification_retries
            ),
            has_previous_candidate=bool(self.recalibration_candidates),
            config=RECALIBRATION_CONFIG,
        )
        if incomplete_capture_decision.action == "retry":
            self.recalibration_incomplete_verification_retries += 1
            retry_verification_capture = True
        elif incomplete_capture_decision.action == "keep_best":
            best_candidate = select_best_recalibration_candidate(
                self.recalibration_candidates
            )
            self.finalize_recalibration_candidate(
                selected_candidate=best_candidate,
                fallback_used=True,
            )
            self.status_label.setText(
                "Status: Verification capture failed; kept the best "
                "previously measured session"
            )
            self.phase_label.setText(
                "Phase Capture: Kept best previous verification session"
            )
        elif incomplete_capture_decision.action == "abort":
            self.abort_recalibration(
                "Status: Recalibration aborted - verification capture "
                "incomplete after retries"
            )

        # Do not stay locked in external trigger mode forever.
        # Return to live view so the system can be tested again.
        self.phase_capture_active = False
        self.pending_phase_indices = []
        self.unassigned_triggered_frames = []
        self.return_to_live_view_mode()

        if retry_verification_capture:
            QTimer.singleShot(500, self.retry_incomplete_verification_capture)
    def finish_phase_capture_session(self):
        if not self.phase_capture_active:
            return

        if any(frame is None for frame in self.phase_capture_frames):
            missing = [
                i for i, frame in enumerate(self.phase_capture_frames)
                if frame is None
            ]

            print("Phase capture incomplete. Missing:", missing)
            self.status_label.setText(f"Status: Phase capture incomplete, missing {missing}")
            self.phase_label.setText(f"Phase Capture: Incomplete, missing {missing}")
            return

        frames = [frame.copy() for frame in self.phase_capture_frames]
        phase_times = self.phase_capture_times.copy()
        phase_frame_timestamps = []

        if self.recalibration_active:
            self.pending_phase_capture_layer = None
            print("Calibration capture: skipping Napari phase-stack layer.")
        else:
            try:
                self.add_phase_capture_time_series_layer(
                    frames,
                    self.last_capture_algorithm_id,
                )
            except Exception as error:
                print("Could not add phase stack to Napari:", error)

        for idx, host_time_s in enumerate(phase_times):
            if self.phase_capture_frame_timestamps[idx] is not None:
                record = dict(self.phase_capture_frame_timestamps[idx])
                record.setdefault("host_time_relative_s", host_time_s)
                record.setdefault("timestamp_source", "camera")
                phase_frame_timestamps.append(record)
            else:
                phase_frame_timestamps.append({
                    "phase_index": idx,
                    "camera_timestamp": None,
                    "host_time_s": time.monotonic(),
                    "host_time_relative_s": host_time_s,
                    "timestamp_source": "host_fallback",
                    "timestamp_clock_frequency_hz": self.phase_capture_timestamp_clock_hz,
                })

        self.phase_capture_active = False
        self.phase_capture_frames = [None] * PHASE_CAPTURE_COUNT
        self.phase_capture_times = [None] * PHASE_CAPTURE_COUNT
        self.phase_capture_frame_timestamps = [None] * PHASE_CAPTURE_COUNT
        self.phase_capture_count = 0
        self.phase_capture_start_time = None
        self.phase_capture_mode_enter_time_s = None
        self.pending_phase_indices = []
        self.unassigned_triggered_frames = []

        self.status_label.setText("Status: Saving phase frames...")
        self.phase_label.setText("Phase Capture: Saving 6 hardware-triggered frames...")

        print("Phase capture complete. Saving 6 hardware-triggered frames...")
        print("Phase times:", phase_times)
        print("Phase frame timestamps:", phase_frame_timestamps)

        save_started = self.start_save_phase_frames_worker(
            frames,
            phase_times,
            phase_frame_timestamps,
        )

        if not save_started:
            self.status_label.setText(
                "Status: Save worker busy - this capture was not saved"
            )
            self.phase_label.setText(
                "Phase Capture: Not saved (save worker busy)"
            )
            print(
                "Phase capture warning: save worker busy, "
                "this completed capture could not be saved."
            )

        # After the 6 hardware-triggered frames are collected, return the
        # camera to normal live view mode.
        self.return_to_live_view_mode()
    def add_phase_capture_time_series_layer(self, frames, algorithm_id):
        stack = np.stack(frames, axis=0)
        capture_number = len(self.phase_capture_time_series_layers) + 1
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        algorithm = get_psi_algorithm(algorithm_id)

        time_series_layer = self.viewer.add_image(
            stack,
            name=(
                f"Phase time series {capture_number:03d} "
                f"{timestamp} ({algorithm['phase_step_deg']:g} deg)"
            ),
            axis_labels=("time", "y", "x"),
            colormap="gray",
            metadata={
                "algorithm_id": algorithm_id,
                "expected_phase_deg": list(algorithm["expected_phase_deg"]),
                "phase_times_s": list(self.phase_capture_times),
            },
        )
        self.phase_capture_time_series_layers.append(time_series_layer)
        self.pending_phase_capture_layer = time_series_layer

        self.apply_layer_visibility_plan(
            "time_series",
            selected_layer=time_series_layer,
        )
        self.viewer.layers.selection.active = time_series_layer
        self.viewer.reset_view()
        print(
            "Added capture stack to Napari:",
            time_series_layer.name,
            stack.shape,
        )
    def start_save_phase_frames_worker(self, frames, phase_times, phase_frame_timestamps=None):
        if self.save_thread is not None:
            print("Save worker is already running.")
            self.status_label.setText("Status: Save worker already running")
            return False

        # Snapshot the current layer's mask for the automatic phase check.
        if self.roi_mask is not None:
            roi_mask_snapshot = np.array(
                self.roi_mask,
                dtype=bool,
                copy=True
            )
        else:
            roi_mask_snapshot = None

        session_prefix = "phase_capture"
        if self.recalibration_active and self.recalibration_stage == "coarse_capture":
            session_prefix = "phase_calibration"

        self.save_thread = QThread()
        self.save_worker = PhaseSaveWorker(
            frames=frames,
            phase_times=phase_times,
            frame_timestamps=phase_frame_timestamps,
            roi_mask=roi_mask_snapshot,
            save_dir=SAVE_DIR,
            session_prefix=session_prefix,
            trigger_positions_ms=self.last_capture_trigger_positions_ms,
            psi_algorithm_id=self.last_capture_algorithm_id,
            voltage_mapper=triangle_wave_falling_edge_voltage,
            recalibration_config=RECALIBRATION_CONFIG,
        )

        self.save_worker.moveToThread(self.save_thread)

        self.save_thread.started.connect(self.save_worker.run)
        self.save_worker.status_ready.connect(self.update_status)
        self.save_worker.result_ready.connect(self.on_phase_save_result)
        self.save_worker.finished.connect(self.on_phase_save_finished)
        self.save_worker.finished.connect(self.save_thread.quit)
        self.save_thread.finished.connect(self.cleanup_save_worker)

        self.save_thread.start()
        return True
    def on_phase_save_finished(self, text):
        print(text)
        self.status_label.setText(f"Status: {text}")
        self.phase_label.setText(text)
    def on_phase_save_result(self, result):
        self.latest_phase_save_result = result

        if (
            isinstance(result, PhaseSaveResult)
            and result.save_success
            and self.pending_phase_capture_layer is not None
        ):
            self.pending_phase_capture_layer.metadata["session_dir"] = result.session_dir
            self.pending_phase_capture_layer.metadata["raw_stack_path"] = result.raw_stack_path
            self.pending_phase_capture_layer = None
        elif isinstance(result, PhaseSaveResult) and not result.save_success:
            self.pending_phase_capture_layer = None

        if not self.recalibration_active:
            return

        if not isinstance(result, PhaseSaveResult):
            self.abort_recalibration(
                "Status: Recalibration failed - invalid save result"
            )
            return

        if not result.save_success:
            self.abort_recalibration(
                "Status: Recalibration failed during save"
            )
            return

        session_dir = result.session_dir
        phase_check_json_path = result.phase_check_json_path

        if not phase_check_json_path and session_dir:
            phase_check_json_path = os.path.join(
                session_dir,
                "measured_phase_check.json",
            )

        if not phase_check_json_path or not os.path.exists(phase_check_json_path):
            self.abort_recalibration(
                "Status: Recalibration failed - missing measured_phase_check.json"
            )
            return

        try:
            with open(
                phase_check_json_path,
                "r",
                encoding="utf-8",
            ) as file:
                phase_check = json.load(file)

            measured_phase_deg = np.asarray(
                phase_check.get("measured_phase_deg", []),
                dtype=np.float64,
            )
        except Exception as error:
            self.abort_recalibration(
                f"Status: Recalibration failed reading phase check: {error}"
            )
            return

        if self.recalibration_stage == "coarse_capture":
            try:
                print(
                    "Recalibration coarse phase measurement:",
                    measured_phase_deg.tolist(),
                )
                refined_positions = self.compute_recalibrated_trigger_positions_ms(
                    coarse_positions_ms=self.recalibration_coarse_positions_ms,
                    measured_phase_deg=measured_phase_deg,
                )
                print(
                    "Recalibration initial refined trigger positions:",
                    refined_positions,
                )

                self.save_recalibrated_trigger_positions(
                    coarse_positions_ms=self.recalibration_coarse_positions_ms,
                    measured_phase_deg=measured_phase_deg,
                    refined_positions_ms=refined_positions,
                    coarse_session_dir=session_dir,
                )

                self.recalibration_target_positions_ms = refined_positions
                self.recalibration_current_positions_ms = refined_positions
                self.recalibration_refinement_round = 0
                self.recalibration_coarse_session_dir = session_dir
                self.recalibration_stage = "verification_capture"

                self.phase_label.setText(
                    "Phase Capture: Recalibration verification capture"
                )
                queue_result = self.queue_or_defer_recalibration_phase_capture(
                    refined_positions,
                    "Status: Recalibration stage 2/2 - verifying refined triggers",
                )
                print(
                    "Recalibration initial verification capture:",
                    queue_result,
                    refined_positions,
                )

            except Exception as error:
                print("Recalibration coarse-to-verification transition failed:", error)
                self.abort_recalibration(
                    f"Status: Recalibration solve failed: {error}"
                )

            return

        if self.recalibration_stage == "verification_capture":
            try:
                candidate = self.build_recalibration_candidate(
                    session_dir=session_dir,
                    measured_phase_deg=measured_phase_deg,
                )
            except Exception as error:
                self.abort_recalibration(
                    f"Status: Recalibration failed scoring verification: {error}"
                )
                return

            self.recalibration_candidates.append(candidate)
            self.recalibration_incomplete_verification_retries = 0
            self.recalibration_verification_capture_count += 1

            phase_error = np.asarray(candidate["phase_error_deg"])
            max_abs_error = candidate["max_abs_error"]
            count_error_over_5 = candidate["count_error_over_5"]
            count_error_over_10 = candidate["count_error_over_10"]

            print("Recalibration verification error (deg):", phase_error.tolist())
            print("Recalibration max abs error (deg):", max_abs_error)
            print("Recalibration verification score:", candidate["score"])
            print(
                "Recalibration verification counts:",
                "over_5=", count_error_over_5,
                "over_10=", count_error_over_10,
            )

            # Moderate-error candidates remain eligible for the final
            # best-candidate fallback, but must not stop timing optimisation.
            verification_passed = is_strict_recalibration_pass(
                phase_error,
                RECALIBRATION_CONFIG,
            )
            verification_action = decide_verification_capture_action(
                verification_passed=verification_passed,
                has_pending_pass_candidate=(
                    self.recalibration_pending_pass_candidate is not None
                ),
                verification_capture_count=(
                    self.recalibration_verification_capture_count
                ),
                config=RECALIBRATION_CONFIG,
            )

            if verification_action == "confirm":
                previous_pass_candidate = self.recalibration_pending_pass_candidate

                selected_candidate = select_best_recalibration_candidate(
                    [previous_pass_candidate, candidate]
                )
                self.finalize_recalibration_candidate(
                    selected_candidate=selected_candidate,
                    fallback_used=False,
                )

                self.phase_label.setText(
                    "Phase Capture: Recalibration confirmed"
                )
                self.status_label.setText(
                    "Status: Recalibration confirmed by two passing "
                    "verification captures; kept the better session"
                )
                return

            if verification_action == "keep_best":
                best_candidate = select_best_recalibration_candidate(
                    self.recalibration_candidates
                )
                self.finalize_recalibration_candidate(
                    selected_candidate=best_candidate,
                    fallback_used=True,
                )
                self.phase_label.setText(
                    "Phase Capture: Best candidate kept without confirmation"
                )
                self.status_label.setText(
                    "Status: Verification limit reached before a second "
                    "confirmation; kept the best candidate"
                )
                return

            if verification_action == "await_confirmation":
                self.recalibration_pending_pass_candidate = candidate
                self.phase_label.setText(
                    "Phase Capture: Recalibration confirmation capture"
                )
                self.queue_or_defer_recalibration_phase_capture(
                    self.recalibration_current_positions_ms,
                    "Status: First verification passed; capturing one "
                    "confirmation with the same trigger positions",
                )
                return

            self.recalibration_pending_pass_candidate = None

            if verification_action == "evaluate_best":
                best_candidate = select_best_recalibration_candidate(
                    self.recalibration_candidates
                )

                if self.should_auto_restart_recalibration(best_candidate):
                    restarted = self.restart_recalibration_after_failed_verification(
                        best_candidate
                    )
                    if restarted:
                        return

                self.finalize_recalibration_candidate(
                    selected_candidate=best_candidate,
                    fallback_used=True,
                )

                self.phase_label.setText(
                    "Phase Capture: Recalibration completed with best candidate"
                )
                self.status_label.setText(
                    "Status: Recalibration completed after "
                    f"{RECALIBRATION_CONFIG.max_verification_captures} verification captures; "
                    f"kept best session with score {best_candidate['score']}"
                )
                return

            try:
                next_positions = self.compute_iterative_refined_positions_ms(
                    current_positions_ms=self.recalibration_current_positions_ms,
                    measured_phase_deg=measured_phase_deg,
                )

                self.recalibration_refinement_round += 1
                self.recalibration_current_positions_ms = next_positions
                self.recalibration_target_positions_ms = next_positions

                self.save_recalibrated_trigger_positions(
                    coarse_positions_ms=self.recalibration_coarse_positions_ms,
                    measured_phase_deg=measured_phase_deg,
                    refined_positions_ms=next_positions,
                    coarse_session_dir=session_dir,
                )

                self.phase_label.setText(
                    "Phase Capture: Recalibration iterative refine"
                )
                self.queue_or_defer_recalibration_phase_capture(
                    next_positions,
                    "Status: Recalibration refine "
                    f"{self.recalibration_refinement_round}/"
                    f"{RECALIBRATION_CONFIG.max_refinement_rounds}, "
                    f"current max error {max_abs_error:.2f} deg",
                )

            except Exception as error:
                self.abort_recalibration(
                    f"Status: Recalibration iterative refine failed: {error}"
                )
