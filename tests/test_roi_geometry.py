import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mwdriver.domain.roi_geometry import (
    apply_mask_to_frame,
    build_inside_mask,
    build_roi_selection,
    build_valid_mask,
    downsample_mask,
)


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

    def test_tilted_ellipse_mask_uses_rotated_axes(self):
        center = np.asarray([24.0, 24.0])
        first_axis = np.asarray([5.0, 5.0])
        second_axis = np.asarray([-2.0, 2.0])
        corners = np.asarray([
            center - first_axis - second_axis,
            center + first_axis - second_axis,
            center + first_axis + second_axis,
            center - first_axis + second_axis,
        ])

        mask, _, _ = build_inside_mask(
            corners,
            roi_type="ellipse",
            image_height=50,
            image_width=50,
            display_downsample=1,
        )

        self.assertTrue(mask[28, 28])
        self.assertFalse(mask[26, 30])

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

    def test_valid_mask_polarity_and_display_downsampling(self):
        inside = np.asarray(
            [[True, True, False, False], [True, True, False, False],
             [False, False, True, True], [False, False, True, True]],
            dtype=bool,
        )
        outside_mode_mask, outside_mode = build_valid_mask(inside, "outside")
        inside_mode_mask, inside_mode = build_valid_mask(inside, "inside")
        self.assertEqual(outside_mode, "mask_outside_area")
        self.assertEqual(inside_mode, "mask_inside_area")
        np.testing.assert_array_equal(outside_mode_mask, inside)
        np.testing.assert_array_equal(inside_mode_mask, np.logical_not(inside))

        display_mask = downsample_mask(outside_mode_mask, 2)
        np.testing.assert_array_equal(display_mask, [[True, False], [False, True]])
        self.assertTrue(display_mask.flags.c_contiguous)

    def test_roi_selection_returns_consistent_mask_metadata_and_display_mask(self):
        inside = np.asarray(
            [[True, False], [False, True]],
            dtype=bool,
        )
        for mode, expected_mode, expected_mask in (
            ("outside", "mask_outside_area", inside),
            ("inside", "mask_inside_area", np.logical_not(inside)),
        ):
            info = {"roi_type": "rectangle"}
            valid_mask, display_mask, result_info, mode_text = build_roi_selection(
                inside,
                info,
                mode,
                display_downsample=1,
                describe_rejected_area=True,
            )
            self.assertEqual(mode_text, expected_mode)
            self.assertIs(result_info, info)
            np.testing.assert_array_equal(valid_mask, expected_mask)
            np.testing.assert_array_equal(display_mask, expected_mask)
            self.assertEqual(info["mask_mode"], expected_mode)
            self.assertIn("False means masked/rejected area", info["mask_meaning"])

    def test_display_mask_application_preserves_disabled_or_mismatched_frames(self):
        frame = np.arange(9, dtype=np.uint8).reshape(3, 3)
        mask = np.asarray([[True, False], [False, True]], dtype=bool)
        self.assertIs(apply_mask_to_frame(frame, mask, enabled=True), frame)
        self.assertIs(apply_mask_to_frame(frame, None, enabled=True), frame)
        self.assertIs(apply_mask_to_frame(frame, mask, enabled=False), frame)

        full_mask = np.asarray(
            [[True, False, True], [False, True, True], [True, True, False]],
            dtype=bool,
        )
        masked = apply_mask_to_frame(frame, full_mask, enabled=True)
        np.testing.assert_array_equal(
            masked,
            [[0, 0, 2], [0, 4, 5], [6, 7, 0]],
        )
        np.testing.assert_array_equal(frame, np.arange(9, dtype=np.uint8).reshape(3, 3))


if __name__ == "__main__":
    unittest.main()
