import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recalibration import (
    build_even_trigger_positions_ms,
    compute_coarse_refined_positions_ms,
    enforce_minimum_trigger_spacing_ms,
    is_strict_recalibration_pass,
    score_recalibration_candidate,
    should_auto_restart_recalibration,
    validate_coarse_measurement,
    validate_trigger_positions_ms,
)


class RecalibrationValidationTests(unittest.TestCase):
    def test_even_trigger_positions_respect_guard_bounds(self):
        self.assertEqual(
            build_even_trigger_positions_ms(6, 100, 549),
            [100, 190, 280, 369, 459, 549],
        )

    def test_spacing_enforcement_respects_camera_minimum_and_bounds(self):
        positions = enforce_minimum_trigger_spacing_ms(
            [100, 110, 120, 130, 140, 150],
            frame_count=6,
            minimum_spacing_ms=25,
            lower_bound_ms=100,
            upper_bound_ms=549,
        )
        self.assertEqual(positions, [100, 125, 150, 175, 200, 225])
        with self.assertRaisesRegex(ValueError, "need 500 ms"):
            enforce_minimum_trigger_spacing_ms(
                [100, 110, 120, 130, 140, 150],
                frame_count=6,
                minimum_spacing_ms=100,
                lower_bound_ms=100,
                upper_bound_ms=549,
            )

    def test_candidate_scoring_normalises_to_first_frame_and_orders_score(self):
        candidate = score_recalibration_candidate(
            session_dir="session",
            measured_phase_deg=[15, 107, 203, 307, 410, 515],
            target_phase_deg=[0, 90, 180, 270, 360, 450],
            trigger_positions_ms=[100, 190, 280, 370, 460, 550],
            tolerance_deg=5.0,
            retry_error_limit_deg=10.0,
        )

        self.assertEqual(candidate["phase_error_deg"], [0, 2, 8, 22, 35, 50])
        self.assertEqual(candidate["count_error_over_5"], 4)
        self.assertEqual(candidate["count_error_over_10"], 3)
        self.assertEqual(candidate["score"][:3], (3, 50.0, 4))

    def test_auto_restart_thresholds(self):
        self.assertTrue(
            should_auto_restart_recalibration(
                {"count_error_over_5": 2, "count_error_over_10": 1}, 2, 3
            )
        )
        self.assertTrue(
            should_auto_restart_recalibration(
                {"count_error_over_5": 3, "count_error_over_10": 0}, 2, 3
            )
        )
        self.assertFalse(
            should_auto_restart_recalibration(
                {"count_error_over_5": 2, "count_error_over_10": 0}, 2, 3
            )
        )
        self.assertFalse(should_auto_restart_recalibration(None, 2, 3))

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

    def test_coarse_solver_accepts_list_returned_by_validation(self):
        result = compute_coarse_refined_positions_ms(
            coarse_positions_ms=[100, 190, 280, 369, 459, 549],
            measured_phase_deg=[0, 102, 201, 305, 403, 535],
            target_phase_deg=[0, 90, 180, 270, 360, 450],
            frame_count=6,
            falling_edge_ms=650,
            edge_guard_ms=100,
            minimum_spacing_ms=25,
            validation_spacing_ms=6,
            minimum_span_deg=220.0,
            minimum_total_abs_step_deg=220.0,
            minimum_max_step_deg=35.0,
            maximum_backtrack_deg=25.0,
        )

        self.assertEqual(len(result), 6)
        self.assertEqual(result[0], 100)
        self.assertEqual(result, sorted(result))


if __name__ == "__main__":
    unittest.main()