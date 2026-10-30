import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from matplotlib.figure import Figure
from matplotlib.image import imread

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mwdriver.analysis.recalibration_plots import save_recalibration_iterations_plot
from mwdriver.capture.phase_capture_controller import PhaseCaptureController
from mwdriver.capture.phase_capture_storage import PhaseSaveResult


class RecalibrationPlotTests(unittest.TestCase):
    def test_result_history_survives_deleted_sessions_and_restart(self):
        expected = [0, 60, 120, 180, 240, 300]
        positions = [100, 175, 234, 289, 346, 400]
        label = SimpleNamespace(setText=lambda text: None)
        viewer = SimpleNamespace(
            recalibration_active=True,
            diagnostic_plots_enabled=True,
            pending_phase_capture_layer=None,
            selected_psi_algorithm_id="six_bucket_60deg",
            recalibration_verification_capture_count=0,
            recalibration_pending_pass_candidate=None,
            recalibration_auto_restart_count=0,
            recalibration_coarse_positions_ms=positions,
            recalibration_candidates=[],
            phase_label=label,
            status_label=label,
            compute_recalibrated_trigger_positions_ms=lambda **kwargs: positions,
            save_recalibrated_trigger_positions=lambda **kwargs: None,
            queue_or_defer_recalibration_phase_capture=lambda *args: "queued",
            abort_recalibration=lambda message: self.fail(message),
            build_recalibration_candidate=lambda **kwargs: {
                "phase_error_deg": [0] * 6,
                "max_abs_error": 0,
                "count_error_over_5": 0,
                "count_error_over_10": 0,
                "score": (0, 0),
            },
        )
        stages = [(0, "coarse_capture"), (0, "verification_capture"), (1, "coarse_capture")]
        for attempt, stage in stages:
            viewer.recalibration_auto_restart_count = attempt
            viewer.recalibration_stage = stage
            with tempfile.TemporaryDirectory() as directory:
                check_path = Path(directory) / "measured_phase_check.json"
                check_path.write_text(json.dumps({
                    "expected_phase_deg": expected,
                    "measured_phase_deg": expected,
                }), encoding="utf-8")
                PhaseCaptureController.on_phase_save_result(viewer, PhaseSaveResult(
                    save_success=True,
                    session_dir=directory,
                    raw_stack_path=None,
                    metadata_path=None,
                    phase_check_json_path=str(check_path),
                ))
                self.assertTrue((Path(directory) / "recalibration_iterations.png").is_file())
                self.assertTrue((Path(directory) / "recalibration_iteration_errors.png").is_file())
                self.assertTrue((Path(directory) / "recalibration_iterations_time_vs_phase.png").is_file())
                self.assertEqual(len(viewer.recalibration_plot_history), stages.index((attempt, stage)) + 1)

        self.assertEqual([record["label"] for record in viewer.recalibration_plot_history], [
            "Attempt 1: coarse",
            "Attempt 1: verification 1",
            "Attempt 2: coarse",
        ])
        for record in viewer.recalibration_plot_history:
            np.testing.assert_allclose(
                record["trigger_positions_ms"],
                positions,
            )

    def test_all_iterations_share_axes_with_expected_reference(self):
        expected = [0, 60, 120, 180, 240, 300]
        history = [
            {
                "label": label,
                "expected_phase_deg": expected,
                "measured_phase_deg": [offset + value * scale for value in expected],
                "trigger_positions_ms": [100, 167, 235, 302, 369, 437],
            }
            for label, offset, scale in (
                ("Attempt 1: coarse", 10, 1.5),
                ("Attempt 1: verification 1", 20, 1.1),
                ("Attempt 2: coarse", 30, 0.8),
                ("Attempt 2: verification 1", 40, 1.0),
            )
        ]
        figures = []
        original_savefig = Figure.savefig

        def capture_figure(figure, *args, **kwargs):
            figures.append(figure)
            return original_savefig(figure, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(Figure, "savefig", new=capture_figure):
                output_path = save_recalibration_iterations_plot(history, directory)
            image = imread(output_path)
            self.assertEqual(output_path.name, "recalibration_iterations.png")
            self.assertGreater(float(np.std(image)), 0.01)
            error_image = imread(Path(directory) / "recalibration_iteration_errors.png")
            self.assertGreater(float(np.std(error_image)), 0.01)
            time_image = imread(Path(directory) / "recalibration_iterations_time_vs_phase.png")
            self.assertGreater(float(np.std(time_image)), 0.01)

        self.assertEqual(len(figures), 3)
        self.assertEqual(len(figures[0].axes), 1)
        lines = figures[0].axes[0].lines
        self.assertEqual(len(lines), 5)
        self.assertEqual([line.get_label() for line in lines], ["Expected"] + [record["label"] for record in history])
        np.testing.assert_array_equal(lines[0].get_ydata(), expected)
        for line, record in zip(lines[1:], history):
            np.testing.assert_array_equal(line.get_xdata(), expected)
            measured = np.asarray(record["measured_phase_deg"])
            np.testing.assert_array_equal(line.get_ydata(), measured - measured[0])

        self.assertEqual(len(figures[1].axes), 1)
        error_axis = figures[1].axes[0]
        self.assertEqual(error_axis.get_ylabel(), "Measured - expected phase (deg)")
        self.assertEqual(len(error_axis.lines), 5)
        self.assertEqual(error_axis.lines[0].get_label(), "Zero error")
        np.testing.assert_array_equal(error_axis.lines[0].get_ydata(), np.zeros(6))
        for line, phase_line, record in zip(error_axis.lines[1:], lines[1:], history):
            measured = np.asarray(record["measured_phase_deg"])
            np.testing.assert_array_equal(line.get_xdata(), expected)
            np.testing.assert_allclose(line.get_ydata(), measured - measured[0] - expected)
            self.assertEqual(line.get_label(), record["label"])
            self.assertEqual(line.get_color(), phase_line.get_color())

        time_axis = figures[2].axes[0]
        self.assertEqual(time_axis.get_xlabel(), "Measured phase (deg)")
        self.assertEqual(time_axis.get_ylabel(), "Requested trigger time from falling-edge start (ms)")
        self.assertEqual(len(time_axis.lines), len(history))
        for line, phase_line, record in zip(time_axis.lines, lines[1:], history):
            measured = np.asarray(record["measured_phase_deg"])
            np.testing.assert_array_equal(line.get_xdata(), measured - measured[0])
            np.testing.assert_array_equal(line.get_ydata(), record["trigger_positions_ms"])
            self.assertEqual(line.get_label(), record["label"])
            self.assertEqual(line.get_color(), phase_line.get_color())


if __name__ == "__main__":
    unittest.main()