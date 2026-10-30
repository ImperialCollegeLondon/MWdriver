import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mwdriver.domain.roi_geometry import build_inside_mask, enclosing_circle_vertices


class AutoAnalysisRoiTests(unittest.TestCase):
    def test_circle_encloses_asymmetric_valid_area(self):
        valid = np.zeros((60, 90), dtype=bool)
        valid[15:36, 20:57] = True
        vertices = enclosing_circle_vertices(valid)
        centre = vertices.mean(axis=0)
        np.testing.assert_allclose(centre, np.mean(np.argwhere(valid), axis=0))
        self.assertAlmostEqual(vertices[1, 1] - vertices[0, 1], vertices[2, 0] - vertices[1, 0])
        mask, _, _ = build_inside_mask(vertices, "ellipse", *valid.shape, 1)
        self.assertTrue(np.all(mask[valid]))

    def test_detached_noise_pixels_do_not_expand_circle(self):
        valid = np.zeros((100, 140), dtype=bool)
        valid[30:60, 40:80] = True
        expected = enclosing_circle_vertices(valid)
        valid[95, 130:132] = True
        valid[5, 5] = True
        vertices = enclosing_circle_vertices(valid)
        np.testing.assert_array_equal(vertices, expected)
        mask, _, _ = build_inside_mask(vertices, "ellipse", *valid.shape, 1)
        self.assertFalse(mask[95, 130])
        self.assertFalse(mask[5, 5])

    def test_circle_handles_single_pixel_and_rejects_empty_mask(self):
        valid = np.zeros((10, 10), dtype=bool)
        valid[3, 5] = True
        vertices = enclosing_circle_vertices(valid)
        np.testing.assert_allclose(vertices.mean(axis=0), [3, 5])
        with self.assertRaises(ValueError):
            enclosing_circle_vertices(np.zeros((10, 10), dtype=bool))


if __name__ == "__main__":
    unittest.main()