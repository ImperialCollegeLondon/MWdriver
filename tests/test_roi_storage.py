import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mwdriver.ui.roi_storage import load_roi_files, save_roi_files


class RoiStorageTests(unittest.TestCase):
    def test_rectangle_round_trip_preserves_mask_and_scaled_shape_vertices(self):
        params = {
            "roi_type": "rectangle",
            "raw_coordinates": {
                "x_min": 4,
                "x_max": 12,
                "y_min": 2,
                "y_max": 10,
            },
            "mask_mode": "mask_outside_area",
        }
        mask = np.zeros((12, 16), dtype=bool)
        mask[2:10, 4:12] = True

        with tempfile.TemporaryDirectory() as temporary_directory:
            params_path, mask_path = save_roi_files(
                temporary_directory,
                params,
                mask,
            )
            self.assertTrue(params_path.is_file())
            self.assertTrue(mask_path.is_file())
            loaded_params, loaded_mask, shape_data, shape_type = load_roi_files(
                temporary_directory,
                image_height=12,
                image_width=16,
                display_downsample=2,
            )

        self.assertEqual(loaded_params, params)
        np.testing.assert_array_equal(loaded_mask, mask)
        self.assertEqual(shape_type, "rectangle")
        np.testing.assert_array_equal(
            shape_data,
            [[1, 2], [1, 6], [5, 6], [5, 2]],
        )

    def test_ellipse_reload_uses_ellipse_shape_type(self):
        params = {
            "roi_type": "circle_or_ellipse",
            "raw_coordinates": {"x_min": 2, "x_max": 8, "y_min": 4, "y_max": 10},
        }
        mask = np.ones((12, 12), dtype=bool)
        with tempfile.TemporaryDirectory() as temporary_directory:
            save_roi_files(temporary_directory, params, mask)
            _, _, shape_data, shape_type = load_roi_files(
                temporary_directory,
                image_height=12,
                image_width=12,
                display_downsample=2,
            )

        self.assertEqual(shape_type, "ellipse")
        np.testing.assert_array_equal(
            shape_data,
            [[2, 1], [2, 4], [5, 4], [5, 1]],
        )

    def test_reload_rejects_mismatched_mask_shape_and_unknown_roi_type(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            save_roi_files(
                temporary_directory,
                {"roi_type": "rectangle", "raw_coordinates": {}},
                np.ones((2, 3), dtype=bool),
            )
            with self.assertRaisesRegex(ValueError, "does not match camera shape"):
                load_roi_files(temporary_directory, 3, 3, 1)

            save_roi_files(
                temporary_directory,
                {"roi_type": "polygon", "raw_coordinates": {
                    "x_min": 0, "x_max": 2, "y_min": 0, "y_max": 2,
                }},
                np.ones((3, 3), dtype=bool),
            )
            with self.assertRaisesRegex(ValueError, "unsupported ROI type"):
                load_roi_files(temporary_directory, 3, 3, 1)


if __name__ == "__main__":
    unittest.main()
