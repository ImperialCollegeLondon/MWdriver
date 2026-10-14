import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from intensity_profile import (
    build_initial_profile_line,
    clamp_profile_line,
    move_profile_line,
    sample_line_profile,
)


class IntensityProfileTests(unittest.TestCase):
    def test_initial_line_is_centered_and_uses_quarter_width_endpoints(self):
        line = build_initial_profile_line(
            image_height=1080,
            image_width=1440,
            display_downsample=2,
        )
        np.testing.assert_array_equal(line, [[270, 180], [270, 540]])
        self.assertEqual(line.dtype, np.float64)

    def test_profile_line_clamping_respects_display_size_without_mutating_input(self):
        source = np.asarray([[-2, -5], [600, 800]], dtype=float)
        clamped = clamp_profile_line(source, 1080, 1440, 2)
        np.testing.assert_array_equal(clamped, [[0, 0], [539, 719]])
        np.testing.assert_array_equal(source, [[-2, -5], [600, 800]])

    def test_drag_movement_handles_endpoints_whole_line_and_initial_anchor(self):
        line = np.asarray([[1, 2], [3, 4], [5, 6]], dtype=float)
        moved, last = move_profile_line(line, "p1", [10, 11])
        np.testing.assert_array_equal(moved, [[10, 11], [3, 4], [5, 6]])
        np.testing.assert_array_equal(last, [10, 11])

        moved, last = move_profile_line(line, "p2", [20, 21])
        np.testing.assert_array_equal(moved, [[1, 2], [3, 4], [20, 21]])
        np.testing.assert_array_equal(last, [20, 21])

        moved, last = move_profile_line(line, "line", [8, 9], [5, 6])
        np.testing.assert_array_equal(moved, line + [3, 3])
        np.testing.assert_array_equal(last, [8, 9])

        moved, last = move_profile_line(line, "line", [8, 9], None)
        self.assertIsNone(moved)
        np.testing.assert_array_equal(last, [8, 9])

        no_change, unchanged_last = move_profile_line(line, None, [8, 9], [5, 6])
        self.assertIsNone(no_change)
        np.testing.assert_array_equal(unchanged_last, [5, 6])

    def test_samples_raw_frame_along_downsampled_display_line(self):
        frame = np.arange(100, dtype=np.uint16).reshape(10, 10)
        profile = sample_line_profile(
            raw_frame=frame,
            line_vertices=[[1, 1], [3, 3]],
            display_downsample=2,
            max_points=20,
        )
        np.testing.assert_array_equal(
            profile,
            [22, 22, 33, 33, 44, 44, 55, 55, 66, 66],
        )
        self.assertEqual(profile.dtype, np.float64)

    def test_sampling_caps_point_count_and_clips_to_frame_bounds(self):
        frame = np.arange(25, dtype=np.uint16).reshape(5, 5)
        profile = sample_line_profile(
            raw_frame=frame,
            line_vertices=[[-5, -5], [20, 20]],
            display_downsample=1,
            max_points=4,
        )
        self.assertEqual(len(profile), 4)
        self.assertEqual(profile[0], 0.0)
        self.assertEqual(profile[-1], 24.0)

    def test_missing_frame_short_line_or_bad_vertices_returns_none(self):
        frame = np.zeros((5, 5), dtype=np.uint16)
        self.assertIsNone(sample_line_profile(None, [[0, 0], [4, 4]], 1, 20))
        self.assertIsNone(sample_line_profile(frame, [[0, 0]], 1, 20))
        self.assertIsNone(sample_line_profile(frame, [[1, 1], [1.5, 1.5]], 1, 20))
        self.assertIsNone(sample_line_profile(frame, [[0], [1]], 1, 20))


if __name__ == "__main__":
    unittest.main()
