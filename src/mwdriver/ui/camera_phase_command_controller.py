"""Software phase-capture command and Pico trigger-update ACK handling."""

import ast

from mwdriver.domain.phase_psi import LEGACY_90_DEG


class CameraPhaseCommandController:
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

    def request_pico_phase_capture_from_button(self):
        """Send the software-button equivalent of pressing Pico GP11."""
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
        self.last_capture_trigger_positions_ms = None
        self.pico_worker.request_start_phase_from_gui()
