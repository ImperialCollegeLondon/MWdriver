import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mwdriver.analysis.phase_unwrap import unwrap_phase_from_center


class PhaseUnwrapTests(unittest.TestCase):
    def test_line_centered_unwrap_recovers_gradients_around_masked_gap(self):
        height, width = 9, 11
        yy, xx = np.indices((height, width))
        true_phase = 0.8 * yy + 0.9 * xx
        wrapped = np.angle(np.exp(1j * true_phase))
        mask = np.zeros((height, width), dtype=bool)
        mask[2:4, 6:8] = True
        phase = np.ma.masked_array(wrapped, mask=mask)

        vertical, unwrapped = unwrap_phase_from_center(
            phase,
            mask,
        )

        vertical_values = vertical.filled(np.nan)
        unwrapped_values = unwrapped.filled(np.nan)
        vertical_pairs = ~mask[1:, :] & ~mask[:-1, :]
        horizontal_pairs = ~mask[:, 1:] & ~mask[:, :-1]
        np.testing.assert_allclose(
            np.diff(vertical_values, axis=0)[vertical_pairs],
            0.8,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            np.diff(unwrapped_values, axis=1)[horizontal_pairs],
            0.9,
            atol=1e-10,
        )
        np.testing.assert_array_equal(np.ma.getmaskarray(unwrapped), mask)

    def test_tilted_ellipse_uses_each_line_valid_midpoint_as_seed(self):
        height = width = 80
        yy, xx = np.indices((height, width), dtype=float)
        center_x = 39.0
        center_y = 40.0
        angle = np.deg2rad(32.0)
        delta_x = xx - center_x
        delta_y = yy - center_y
        major = np.cos(angle) * delta_x + np.sin(angle) * delta_y
        minor = -np.sin(angle) * delta_x + np.cos(angle) * delta_y
        valid = (major / 27.0) ** 2 + (minor / 15.0) ** 2 <= 1.0
        true_phase = 0.73 * yy + 0.91 * xx
        phase = np.ma.masked_array(
            np.angle(np.exp(1j * true_phase)),
            mask=~valid,
        )

        _, unwrapped = unwrap_phase_from_center(phase, ~valid)
        unwrapped_values = unwrapped.filled(np.nan)
        horizontal_pairs = valid[:, 1:] & valid[:, :-1]
        vertical_pairs = valid[1:, :] & valid[:-1, :]
        horizontal_difference = np.diff(unwrapped_values, axis=1)
        vertical_difference = np.diff(unwrapped_values, axis=0)

        np.testing.assert_allclose(
            horizontal_difference[horizontal_pairs],
            0.91,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            vertical_difference[vertical_pairs],
            0.73,
            atol=1e-10,
        )

    def test_unwrap_rejects_mismatched_phase_and_mask_shapes(self):
        phase = np.ma.asarray(np.zeros((4, 5)))
        with self.assertRaisesRegex(ValueError, "matching 2D"):
            unwrap_phase_from_center(phase, np.zeros((5, 4), dtype=bool))


if __name__ == "__main__":
    unittest.main()
