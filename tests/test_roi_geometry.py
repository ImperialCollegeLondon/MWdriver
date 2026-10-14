import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roi_geometry import build_inside_mask


class RoiGeometryTests(unittest.TestCase):
    def test_rectangle_mask_scales_coordinates_and_clips_to_image(self):
        mask, roi_type, metadata = build_inside_mask(
            roi_vertices=[[1, 2], [4, 6]],
            roi_type="rectangle",
            image_height=8,
            image_width=10,
            display_downsample=2,
        )

        self.assertEqual(roi_type, "rectangle")
        self.assertEqual(mask.shape, (8, 10))
        self.assertEqual(int(mask.sum()), 36)
        self.assertTrue(mask[2, 4])
        self.assertTrue(mask[7, 9])
        self.assertFalse(mask[1, 4])
        self.assertEqual(metadata["raw_coordinates"], {
            "x_min": 4,
            "x_max": 10,
            "y_min": 2,
            "y_max": 8,
        })
        self.assertEqual(metadata["display_downsample"], 2)

    def test_ellipse_mask_and_metadata(self):
        mask, roi_type, metadata = build_inside_mask(
            roi_vertices=[[1, 2], [5, 8]],
            roi_type="ellipse",
            image_height=7,
            image_width=10,
            display_downsample=1,
        )

        self.assertEqual(roi_type, "ellipse")
        self.assertEqual(metadata["roi_type"], "circle_or_ellipse")
        self.assertTrue(mask[3, 5])
        self.assertFalse(mask[1, 2])
        self.assertEqual(metadata["raw_coordinates"]["average_radius"], 2.5)
        self.assertEqual(metadata["image_size"], {"width": 10, "height": 7})

    def test_invalid_roi_types_and_degenerate_bounds_return_no_mask(self):
        self.assertEqual(
            build_inside_mask([[1, 2], [1, 5]], "rectangle", 8, 10, 1),
            (None, None, None),
        )
        self.assertEqual(
            build_inside_mask([[1, 2], [4, 5]], "polygon", 8, 10, 1),
            (None, None, None),
        )
        self.assertEqual(
            build_inside_mask([[1, 2]], "ellipse", 8, 10, 1),
            (None, None, None),
        )


if __name__ == "__main__":
    unittest.main()
