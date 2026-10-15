import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from camera_phase_command_controller import CameraPhaseCommandController
from mwdriver.domain.phase_psi import LEGACY_90_DEG


class Label:
    def __init__(self):
        self.text = None

    def setText(self, text):
        self.text = text


class CameraPhaseCommandControllerTests(unittest.TestCase):
    def make_viewer(self, **overrides):
        state = {
            "pico_worker": None,
            "recalibration_active": False,
            "save_thread": None,
            "status_label": Label(),
            "phase_label": Label(),
            "pico_label": Label(),
            "active_trigger_positions_ms": None,
            "selected_psi_algorithm_id": LEGACY_90_DEG,
            "last_capture_trigger_positions_ms": [100, 164, 222, 276, 330, 384],
            "load_active_trigger_positions_from_json": lambda: None,
            "build_formal_capture_positions_ms": lambda positions: positions,
            "queued_positions": [],
            "queue_phase_capture_with_trigger_positions": lambda positions: state[
                "queued_positions"
            ].append(positions),
        }
        state.update(overrides)
        return SimpleNamespace(**state)

    def test_button_reports_missing_pico(self):
        viewer = self.make_viewer()

        CameraPhaseCommandController.request_pico_phase_capture_from_button(viewer)

        self.assertEqual(viewer.status_label.text, "Status: Pico not connected")
        self.assertEqual(viewer.pico_label.text, "Pico: Not connected")

    def test_button_requires_calibration_for_nonlegacy_psi_mode(self):
        worker = SimpleNamespace(request_start_phase_from_gui=lambda: self.fail("must not start"))
        viewer = self.make_viewer(
            pico_worker=worker,
            selected_psi_algorithm_id="six_bucket_60_deg",
        )

        CameraPhaseCommandController.request_pico_phase_capture_from_button(viewer)

        self.assertEqual(
            viewer.status_label.text,
            "Status: Calibrate this PSI mode before capturing",
        )
        self.assertEqual(
            viewer.phase_label.text,
            "Phase Capture: No calibration saved for selected mode",
        )

    def test_button_queues_existing_calibrated_trigger_positions(self):
        positions = [100, 164, 222, 276, 330, 384]
        viewer = self.make_viewer(
            pico_worker=SimpleNamespace(),
            active_trigger_positions_ms=positions,
        )

        CameraPhaseCommandController.request_pico_phase_capture_from_button(viewer)

        self.assertEqual(viewer.queued_positions, [positions])
        self.assertEqual(
            viewer.status_label.text,
            "Status: Sending Start Phase Capture with active calibrated triggers",
        )


if __name__ == "__main__":
    unittest.main()
