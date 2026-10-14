import ast
import sys
import unittest
from pathlib import Path
from types import MethodType

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recalibration import (
    is_strict_recalibration_pass,
    validate_coarse_measurement,
    validate_trigger_positions_ms,
)


class RecalibrationValidationTests(unittest.TestCase):
    def validate_positions(self, positions):
        return validate_trigger_positions_ms(
            positions,
            frame_count=6,
            falling_edge_ms=650,
            minimum_spacing_ms=6,
        )

    def validate_measurement(self, values):
        return validate_coarse_measurement(
            values,
            frame_count=6,
            minimum_span_deg=220.0,
            minimum_total_abs_step_deg=220.0,
            minimum_max_step_deg=35.0,
            maximum_backtrack_deg=25.0,
        )

    def test_trigger_positions_are_rounded_and_returned_as_integers(self):
        self.assertEqual(
            self.validate_positions([100.2, 120.4, 140.5, 160.6, 180.7, 200.8]),
            [100, 120, 140, 161, 181, 201],
        )

    def test_trigger_positions_require_six_values(self):
        with self.assertRaisesRegex(ValueError, "Exactly 6"):
            self.validate_positions([100, 120, 140, 160, 180])

    def test_trigger_positions_must_be_in_range_and_spaced(self):
        with self.assertRaisesRegex(ValueError, "out of range"):
            self.validate_positions([100, 120, 140, 160, 180, 650])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            self.validate_positions([100, 120, 140, 160, 180, 170])
        with self.assertRaisesRegex(ValueError, "spacing is too small"):
            self.validate_positions([100, 120, 140, 160, 180, 185])

    def test_coarse_measurement_is_normalised_to_first_phase(self):
        self.assertEqual(
            self.validate_measurement([15, 65, 115, 165, 215, 265]),
            [0.0, 50.0, 100.0, 150.0, 200.0, 250.0],
        )

    def test_coarse_measurement_rejects_bad_shape_nonfinite_and_low_span(self):
        with self.assertRaisesRegex(ValueError, "exactly 6"):
            self.validate_measurement([0, 50, 100, 150, 200])
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            self.validate_measurement([0, 50, 100, 150, 200, float("nan")])
        with self.assertRaisesRegex(ValueError, "span is too short"):
            self.validate_measurement([0, 10, 20, 30, 40, 50])
        with self.assertRaisesRegex(ValueError, "excessive local backtracking"):
            self.validate_measurement([0, 60, 120, 90, 150, 250])

    def test_only_all_points_within_tolerance_are_a_strict_pass(self):
        self.assertTrue(
            is_strict_recalibration_pass([0, 1, -2, 3, -4, 5], 5.0)
        )
        self.assertFalse(
            is_strict_recalibration_pass([0, 1, -2, 3, -4, 5.01], 5.0)
        )
        self.assertFalse(
            is_strict_recalibration_pass([0, 1, -2, 3, -4, float("nan")], 5.0)
        )

    def test_coarse_solver_accepts_list_returned_by_validator(self):
        camera_path = Path(__file__).resolve().parents[1] / "src" / "camera.py"
        camera_tree = ast.parse(camera_path.read_text(encoding="utf-8"))
        viewer_class = next(
            node for node in camera_tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "ThorlabsCameraViewer"
        )
        solver_node = next(
            node for node in viewer_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "compute_recalibrated_trigger_positions_ms"
        )
        solver_module = ast.Module(body=[solver_node], type_ignores=[])
        solver_namespace = {
            "np": np,
            "PHASE_CAPTURE_COUNT": 6,
            "RECALIBRATION_EDGE_GUARD_MS": 100,
            "RECALIBRATION_FALLING_EDGE_MS": 650,
        }
        exec(compile(solver_module, str(camera_path), "exec"), solver_namespace)

        class FakeViewer:
            def validate_coarse_recalibration_measurement(self, values):
                return validate_coarse_measurement(
                    values,
                    frame_count=6,
                    minimum_span_deg=220.0,
                    minimum_total_abs_step_deg=220.0,
                    minimum_max_step_deg=35.0,
                    maximum_backtrack_deg=25.0,
                )

            def get_recalibration_target_phase_deg(self):
                return np.asarray([0, 90, 180, 270, 360, 450], dtype=np.float64)

            def enforce_minimum_trigger_spacing(self, positions):
                return positions

            def validate_trigger_positions_ms(self, positions):
                return positions

        solver = MethodType(
            solver_namespace["compute_recalibrated_trigger_positions_ms"],
            FakeViewer(),
        )
        result = solver(
            coarse_positions_ms=[100, 190, 280, 369, 459, 549],
            measured_phase_deg=[0, 102, 201, 305, 403, 535],
        )

        self.assertEqual(len(result), 6)
        self.assertEqual(result[0], 100)
        self.assertEqual(result, sorted(result))


if __name__ == "__main__":
    unittest.main()