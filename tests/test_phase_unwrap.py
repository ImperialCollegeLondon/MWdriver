import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase_unwrap import unwrap_phase_from_center


def reference_center_out_unwrap(phase, mask, center_y, center_x):
    values = np.zeros_like(phase, dtype=np.float64)
    values[center_y:, :] = np.unwrap(phase[center_y:, :].filled(0.0), axis=0)
    values[center_y::-1, :] = np.unwrap(
        phase[center_y::-1, :].filled(0.0), axis=0
    )
    vertical = np.ma.masked_array(values.copy(), mask=mask)
    values[:, center_x:] = np.unwrap(values[:, center_x:], axis=1)
    values[:, center_x::-1] = np.unwrap(values[:, center_x::-1], axis=1)
    return vertical, np.ma.masked_array(values, mask=mask)


class PhaseUnwrapTests(unittest.TestCase):
    def test_center_out_unwrap_matches_existing_axis_order_and_mask(self):
        height, width = 9, 11
        yy, xx = np.indices((height, width))
        true_phase = 0.8 * yy + 0.9 * xx
        wrapped = np.angle(np.exp(1j * true_phase))
        mask = np.zeros((height, width), dtype=bool)
        mask[2:4, 6:8] = True
        phase = np.ma.masked_array(wrapped, mask=mask)

        expected_vertical, expected_unwrapped = reference_center_out_unwrap(
            phase,
            mask,
            center_y=4,
            center_x=5,
        )
        vertical, unwrapped = unwrap_phase_from_center(
            phase,
            mask,
            center_y=4,
            center_x=5,
        )

        np.testing.assert_allclose(vertical.filled(np.nan), expected_vertical.filled(np.nan))
        np.testing.assert_allclose(unwrapped.filled(np.nan), expected_unwrapped.filled(np.nan))
        np.testing.assert_array_equal(np.ma.getmaskarray(unwrapped), mask)

    def test_unwrap_rejects_mismatched_shape_and_out_of_bounds_center(self):
        phase = np.ma.asarray(np.zeros((4, 5)))
        with self.assertRaisesRegex(ValueError, "matching 2D"):
            unwrap_phase_from_center(phase, np.zeros((5, 4), dtype=bool), 2, 2)
        with self.assertRaisesRegex(ValueError, "inside the phase image"):
            unwrap_phase_from_center(phase, np.zeros((4, 5), dtype=bool), 4, 2)


if __name__ == "__main__":
    unittest.main()
