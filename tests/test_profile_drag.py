import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mwdriver.domain.profile_drag import distance_point_to_segment, get_line_drag_target


class ProfileDragTests(unittest.TestCase):
    def test_distance_to_segment_handles_projection_and_degenerate_segment(self):
        self.assertAlmostEqual(distance_point_to_segment([5, 3], [0, 0], [10, 0]), 3.0)
        self.assertAlmostEqual(distance_point_to_segment([5, 0], [0, 0], [10, 0]), 0.0)
        self.assertAlmostEqual(distance_point_to_segment([4, 3], [1, 1], [1, 1]), 3.60555127546)

    def test_drag_target_prefers_endpoints_then_segment(self):
        line = [[0, 0], [0, 10]]
        self.assertEqual(get_line_drag_target([1, 1], line, threshold=2), "p1")
        self.assertEqual(get_line_drag_target([1, 9], line, threshold=2), "p2")
        self.assertEqual(get_line_drag_target([1, 5], line, threshold=2), "line")
        self.assertIsNone(get_line_drag_target([5, 5], line, threshold=2))

    def test_invalid_lines_have_no_drag_target(self):
        self.assertIsNone(get_line_drag_target([0, 0], [], threshold=2))
        self.assertIsNone(get_line_drag_target([0, 0], [[1, 1]], threshold=2))


if __name__ == "__main__":
    unittest.main()
