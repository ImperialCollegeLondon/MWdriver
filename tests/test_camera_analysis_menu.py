import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import numpy as np
from qtpy.QtWidgets import QApplication, QMainWindow

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mwdriver.ui.camera_controls_controller import CameraControlsController
from mwdriver.capture.phase_capture_process import run_phase_check
from mwdriver.capture.phase_capture_worker import PhaseSaveWorker
from mwdriver.analysis import check_measured_vs_expected_phase as checker


class AnalysisMenuTests(unittest.TestCase):
    def test_analysis_menu_separates_actions_and_toggles_diagnostics(self):
        application = QApplication.instance() or QApplication([])
        window = QMainWindow()
        viewer = SimpleNamespace(
            viewer=SimpleNamespace(window=SimpleNamespace(_qt_window=window)),
            selected_psi_algorithm_id="six_bucket_60deg",
            diagnostic_plots_enabled=False,
            exposure_s=0.02,
            gain_value=0,
        )
        callbacks = (
            "reload_dataset_from_file", "reload_analysis_from_file", "reload_stack_from_directory",
            "update_stream_menu_actions", "allow_streaming", "disallow_streaming",
            "request_pico_phase_capture_from_button", "start_recalibration_from_button",
            "run_latest_phase_analysis", "run_selected_phase_analysis", "show_camera_preview",
            "on_psi_algorithm_changed", "place_camera_status_controls_in_layer_controls",
            "place_camera_adjust_controls_in_layer_controls", "on_exposure_slider_changed",
            "on_exposure_spinbox_changed", "on_gain_slider_changed", "on_gain_spinbox_changed",
        )
        for callback in callbacks:
            setattr(viewer, callback, lambda *args: None)
        viewer.set_diagnostic_plots_enabled = MethodType(
            CameraControlsController.set_diagnostic_plots_enabled, viewer,
        )
        with patch("mwdriver.ui.camera_controls_controller.QTimer.singleShot"):
            CameraControlsController.init_controls(viewer)
        self.assertEqual(viewer.camera_menu.title(), "Camera")
        self.assertEqual(viewer.analysis_menu.title(), "Analysis")
        self.assertIn(viewer.reload_stack_action, viewer.analysis_menu.actions())
        self.assertIn(viewer.reload_dataset_action, viewer.analysis_menu.actions())
        self.assertIn(viewer.reload_analysis_action, viewer.analysis_menu.actions())
        self.assertEqual(viewer.reload_stack_action.text(), "Reload Dataset...")
        self.assertEqual(viewer.reload_dataset_action.text(), "Reload Stack...")
        file_menu = next((action.menu() for action in window.menuBar().actions()
                          if action.menu() is not None and action.text().replace("&", "").lower() == "file"), None)
        if file_menu is not None:
            self.assertNotIn(viewer.reload_dataset_action, file_menu.actions())
            self.assertNotIn(viewer.reload_analysis_action, file_menu.actions())
        self.assertIn(viewer.analyze_latest_phase_button, viewer.analysis_menu.actions())
        self.assertNotIn(viewer.analyze_latest_phase_button, viewer.camera_menu.actions())
        self.assertIn(viewer.start_phase_capture_button, viewer.camera_menu.actions())
        action = viewer.diagnostic_plots_action
        self.assertTrue(action.isCheckable())
        self.assertFalse(action.isChecked())
        action.trigger()
        self.assertTrue(viewer.diagnostic_plots_enabled)
        action.trigger()
        self.assertFalse(viewer.diagnostic_plots_enabled)
        window.close()

    def test_worker_forwards_disabled_diagnostics_to_checker_process(self):
        worker = PhaseSaveWorker(frames=[], diagnostic_plots=False)
        with patch("mwdriver.capture.phase_capture_worker.run_phase_check") as run_check:
            run_check.return_value = SimpleNamespace(stdout="", stderr="", message="done")
            worker.run_automatic_phase_check(".")
        self.assertFalse(run_check.call_args.kwargs["diagnostic_plots"])
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "checker.py"
            script.touch()
            commands = []

            def runner(command, **kwargs):
                commands.append(command)
                return SimpleNamespace(returncode=1, stdout="", stderr="")

            run_phase_check(str(script), directory, directory, 1,
                            diagnostic_plots=False, runner=runner)
            self.assertIn("--no-diagnostic-plots", commands[0])

    def test_checker_disables_extra_plots_but_keeps_standard_outputs(self):
        diagnostics = {
            "peak_x": 1, "peak_y": 1, "mask_radius": 8, "peak_distance": 20,
            "spectrum": np.ones((4, 4)), "fourier_mask": np.ones((4, 4)),
            "complex_fields": np.ones((6, 4, 4)), "relative_phase_sign": 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory)
            np.save(session / "capture_raw_stack.npy", np.ones((6, 4, 4)))
            with (
                patch.object(checker, "estimate_phase_curve", return_value=(np.arange(6) * 90, diagnostics)),
                patch.object(checker, "create_field_maps") as fields,
                patch.object(checker, "create_fourier_filter_plot") as fourier,
                patch.object(checker, "create_annotated_plot") as standard,
            ):
                checker.main([directory, "--no-diagnostic-plots"])
            fields.assert_not_called()
            fourier.assert_not_called()
            standard.assert_called_once()
            report = json.loads((session / "measured_phase_check.json").read_text())
            self.assertIn("measured_phase_deg", report)
            self.assertNotIn("field_map_paths", report)


if __name__ == "__main__":
    unittest.main()