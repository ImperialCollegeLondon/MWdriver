import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import recalibration as legacy_recalibration

from mwdriver.domain.recalibration import (
    DEFAULT_RECALIBRATION_CONFIG,
    RecalibrationConfig,
    build_even_trigger_positions_ms,
    compute_coarse_refined_positions_ms,
    enforce_minimum_trigger_spacing_ms,
    decide_incomplete_capture_action,
    decide_verification_capture_action,
    is_strict_recalibration_pass,
    score_recalibration_candidate,
    select_best_recalibration_candidate,
    should_auto_restart_recalibration,
    validate_coarse_measurement,
    validate_trigger_positions_ms,
)


class RecalibrationValidationTests(unittest.TestCase):
    def test_legacy_module_reexports_packaged_api(self):
        self.assertIs(legacy_recalibration.RecalibrationConfig, RecalibrationConfig)
        self.assertIs(
            legacy_recalibration.compute_coarse_refined_positions_ms,
            compute_coarse_refined_positions_ms,
        )

    def test_verification_capture_progression_actions(self):
        config = RecalibrationConfig(max_refinement_rounds=2)
        cases = [
            (True, True, 2, "confirm"),
            (True, False, 2, "await_confirmation"),
            (True, False, 3, "keep_best"),
            (False, False, 2, "refine"),
            (False, False, 3, "evaluate_best"),
        ]

        for passed, has_pending, capture_count, expected in cases:
            with self.subTest(
                passed=passed,
                has_pending=has_pending,
                capture_count=capture_count,
            ):
                self.assertEqual(
                    decide_verification_capture_action(
                        verification_passed=passed,
                        has_pending_pass_candidate=has_pending,
                        verification_capture_count=capture_count,
                        config=config,
                    ),
                    expected,
                )

    def test_incomplete_capture_decision_outcomes(self):
        config = RecalibrationConfig(max_incomplete_verification_retries=2)
        cases = [
            (False, "verification_capture", 0, True, "none"),
            (True, "verification_capture", 0, False, "retry"),
            (True, "verification_capture", 1, True, "retry"),
            (True, "verification_capture", 2, True, "keep_best"),
            (True, "verification_capture", 2, False, "abort"),
            (True, "coarse_capture", 0, True, "abort"),
        ]

        for active, stage, retry_count, has_candidate, expected in cases:
            with self.subTest(
                active=active,
                stage=stage,
                retry_count=retry_count,
                has_candidate=has_candidate,
            ):
                decision = decide_incomplete_capture_action(
                    recalibration_active=active,
                    recalibration_stage=stage,
                    incomplete_retry_count=retry_count,
                    has_previous_candidate=has_candidate,
                    config=config,
                )
                self.assertEqual(decision.action, expected)

    def test_config_derives_verification_limit_from_refinement_rounds(self):
        config = RecalibrationConfig(max_refinement_rounds=5)
        self.assertEqual(config.max_verification_captures, 6)
        self.assertEqual(DEFAULT_RECALIBRATION_CONFIG.frame_count, 6)

    def test_even_trigger_positions_respect_guard_bounds(self):
        self.assertEqual(
            build_even_trigger_positions_ms(DEFAULT_RECALIBRATION_CONFIG),
            [100, 190, 280, 369, 459, 549],
        )

    def test_spacing_enforcement_respects_camera_minimum_and_bounds(self):
        positions = enforce_minimum_trigger_spacing_ms(
            [100, 110, 120, 130, 140, 150],
            config=DEFAULT_RECALIBRATION_CONFIG,
            camera_required_spacing_ms=25,
        )
        self.assertEqual(positions, [100, 125, 150, 175, 200, 225])
        camera_limited_positions = enforce_minimum_trigger_spacing_ms(
            [100, 110, 120, 130, 140, 150],
            config=DEFAULT_RECALIBRATION_CONFIG,
            camera_required_spacing_ms=44,
        )
        self.assertEqual(
            np.diff(camera_limited_positions).tolist(),
            [44, 44, 44, 44, 44],
        )
        with self.assertRaisesRegex(ValueError, "need 500 ms"):
            enforce_minimum_trigger_spacing_ms(
                [100, 110, 120, 130, 140, 150],
                config=DEFAULT_RECALIBRATION_CONFIG,
                camera_required_spacing_ms=100,
            )

    def test_candidate_scoring_normalises_to_first_frame_and_orders_score(self):
        candidate = score_recalibration_candidate(
            session_dir="session",
            measured_phase_deg=[15, 107, 203, 307, 410, 515],
            target_phase_deg=[0, 90, 180, 270, 360, 450],
            trigger_positions_ms=[100, 190, 280, 370, 460, 550],
            config=DEFAULT_RECALIBRATION_CONFIG,
        )

        self.assertEqual(candidate["phase_error_deg"], [0, 2, 8, 22, 35, 50])
        self.assertEqual(candidate["count_error_over_5"], 4)
        self.assertEqual(candidate["count_error_over_10"], 3)
        self.assertEqual(candidate["score"][:3], (3, 50.0, 4))

    def test_best_candidate_uses_lowest_score_and_preserves_first_tie(self):
        first = {"score": (1, 10.0), "session_dir": "first"}
        tied = {"score": (1, 10.0), "session_dir": "tied"}
        worse = {"score": (2, 1.0), "session_dir": "worse"}

        self.assertIs(
            select_best_recalibration_candidate([first, tied, worse]),
            first,
        )
        self.assertIs(
            select_best_recalibration_candidate([worse, tied]),
            tied,
        )

    def test_best_candidate_requires_nonempty_candidates(self):
        with self.assertRaisesRegex(ValueError, "At least one"):
            select_best_recalibration_candidate([])

    def test_auto_restart_thresholds(self):
        self.assertTrue(
            should_auto_restart_recalibration(
                {"count_error_over_5": 2, "count_error_over_10": 1},
                DEFAULT_RECALIBRATION_CONFIG,
            )
        )
        self.assertTrue(
            should_auto_restart_recalibration(
                {"count_error_over_5": 3, "count_error_over_10": 0},
                DEFAULT_RECALIBRATION_CONFIG,
            )
        )
        self.assertFalse(
            should_auto_restart_recalibration(
                {"count_error_over_5": 2, "count_error_over_10": 0},
                DEFAULT_RECALIBRATION_CONFIG,
            )
        )
        self.assertFalse(
            should_auto_restart_recalibration(None, DEFAULT_RECALIBRATION_CONFIG)
        )

    def validate_positions(self, positions):
        return validate_trigger_positions_ms(
            positions,
            DEFAULT_RECALIBRATION_CONFIG,
        )

    def validate_measurement(self, values):
        return validate_coarse_measurement(
            values,
            DEFAULT_RECALIBRATION_CONFIG,
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
            is_strict_recalibration_pass(
                [0, 1, -2, 3, -4, 5], DEFAULT_RECALIBRATION_CONFIG
            )
        )
        self.assertFalse(
            is_strict_recalibration_pass(
                [0, 1, -2, 3, -4, 5.01], DEFAULT_RECALIBRATION_CONFIG
            )
        )
        self.assertFalse(
            is_strict_recalibration_pass(
                [0, 1, -2, 3, -4, float("nan")], DEFAULT_RECALIBRATION_CONFIG
            )
        )

    def test_coarse_solver_accepts_list_returned_by_validation(self):
        result = compute_coarse_refined_positions_ms(
            coarse_positions_ms=[100, 190, 280, 369, 459, 549],
            measured_phase_deg=[0, 102, 201, 305, 403, 535],
            target_phase_deg=[0, 90, 180, 270, 360, 450],
            config=DEFAULT_RECALIBRATION_CONFIG,
            camera_required_spacing_ms=25,
        )

        self.assertEqual(len(result), 6)
        self.assertEqual(result[0], 100)
        self.assertEqual(result, sorted(result))


if __name__ == "__main__":
    unittest.main()