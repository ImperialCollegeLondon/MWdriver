import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from camera_frames import prepare_display_frame


class CameraFrameTests(unittest.TestCase):
    def test_downsampling_preserves_stride_selection_and_returns_contiguous_frame(self):
        raw_frame = np.arange(48, dtype=np.uint16).reshape(6, 8)

        display_frame = prepare_display_frame(raw_frame, display_downsample=2)

        np.testing.assert_array_equal(display_frame, raw_frame[::2, ::2])
        self.assertEqual(display_frame.shape, (3, 4))
        self.assertTrue(display_frame.flags.c_contiguous)
        self.assertEqual(raw_frame.shape, (6, 8))

    def test_factor_one_preserves_values_and_returns_contiguous_frame(self):
        raw_frame = np.arange(20, dtype=np.uint16).reshape(4, 5)[:, ::2]

        display_frame = prepare_display_frame(raw_frame, display_downsample=1)

        np.testing.assert_array_equal(display_frame, raw_frame)
        self.assertTrue(display_frame.flags.c_contiguous)
        self.assertFalse(np.shares_memory(display_frame, raw_frame))

    def test_rejects_nonpositive_downsample(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            prepare_display_frame(np.zeros((2, 2)), display_downsample=0)


if __name__ == "__main__":
    unittest.main()