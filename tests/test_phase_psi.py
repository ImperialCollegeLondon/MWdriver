import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase_psi import (
    ALGORITHMS,
    LEGACY_90_DEG,
    SIX_BUCKET_60_DEG,
    get_algorithm_id_from_metadata,
    reconstruct_phase,
)


class PhaseReconstructionTests(unittest.TestCase):
    def make_interferogram(self, algorithm_id, phase):
        steps = np.deg2rad(ALGORITHMS[algorithm_id]["expected_phase_deg"])
        return 120.0 + 35.0 * np.cos(phase[None, :] + steps[:, None])

    def assert_phase_recovered(self, algorithm_id):
        expected = np.linspace(-np.pi, np.pi, 101)
        frames = self.make_interferogram(algorithm_id, expected)
        actual = reconstruct_phase(frames, algorithm_id)
        error = np.angle(np.exp(1j * (actual - expected)))
        np.testing.assert_allclose(error, 0.0, atol=2e-12)

    def test_legacy_90_degree_six_frame_reconstruction(self):
        self.assert_phase_recovered(LEGACY_90_DEG)

    def test_60_degree_six_bucket_reconstruction(self):
        self.assert_phase_recovered(SIX_BUCKET_60_DEG)

    def test_reconstruction_requires_six_frames(self):
        with self.assertRaises(ValueError):
            reconstruct_phase(np.zeros((5, 2, 2)), SIX_BUCKET_60_DEG)

    def test_reconstruction_rejects_unknown_algorithm(self):
        with self.assertRaises(ValueError):
            reconstruct_phase(np.zeros((6, 2, 2)), "unknown")

    def test_metadata_without_algorithm_defaults_to_legacy_mode(self):
        self.assertEqual(
            get_algorithm_id_from_metadata({}),
            LEGACY_90_DEG,
        )

    def test_metadata_selects_60_degree_mode(self):
        metadata = {
            "phase_capture_model": {"algorithm_id": SIX_BUCKET_60_DEG}
        }
        self.assertEqual(
            get_algorithm_id_from_metadata(metadata),
            SIX_BUCKET_60_DEG,
        )


if __name__ == "__main__":
    unittest.main()