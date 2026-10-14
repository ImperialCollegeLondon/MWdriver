import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recalibration import (
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


if __name__ == "__main__":
    unittest.main()