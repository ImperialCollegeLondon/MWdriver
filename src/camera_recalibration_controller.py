"""GUI-level recalibration scheduling and persistence orchestration."""

import json
import os
import shutil
import time

import numpy as np
from qtpy.QtCore import QCoreApplication

from mwdriver.domain.phase_psi import LEGACY_90_DEG, get_algorithm as get_psi_algorithm
from mwdriver.domain.recalibration import (
    DEFAULT_RECALIBRATION_CONFIG,
    build_even_trigger_positions_ms,
    compute_coarse_refined_positions_ms,
    compute_iterative_refined_positions_ms as compute_iterative_refined_positions,
    enforce_minimum_trigger_spacing_ms,
    score_recalibration_candidate,
    should_auto_restart_recalibration as should_auto_restart_candidate,
    validate_coarse_measurement,
    validate_trigger_positions_ms as validate_recalibration_trigger_positions_ms,
)

RECALIBRATION_CONFIG = DEFAULT_RECALIBRATION_CONFIG
RECALIBRATION_TRIGGER_JSON_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "phase_trigger_positions.json",
)
RECALIBRATION_TRIGGER_UPDATE_ACK_TIMEOUT_S = 1.5


class CameraRecalibrationController:
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
