"""Shared definitions and reconstruction algorithms for six-frame PSI."""

from __future__ import annotations

import numpy as np


LEGACY_90_DEG = "six_frame_90deg"
SIX_BUCKET_60_DEG = "six_bucket_60deg"
DEFAULT_ALGORITHM = LEGACY_90_DEG

ALGORITHMS = {
    LEGACY_90_DEG: {
        "label": "90 degree six-frame",
        "phase_step_deg": 90.0,
        "expected_phase_deg": (0.0, 90.0, 180.0, 270.0, 360.0, 450.0),
    },
    SIX_BUCKET_60_DEG: {
        "label": "60 degree six-bucket",
        "phase_step_deg": 60.0,
        "expected_phase_deg": (0.0, 60.0, 120.0, 180.0, 240.0, 300.0),
    },
}


def get_algorithm(algorithm_id=DEFAULT_ALGORITHM):
    """Return the configuration for a known PSI algorithm ID."""
    try:
        return ALGORITHMS[algorithm_id]
    except KeyError as error:
        raise ValueError(f"Unknown PSI algorithm: {algorithm_id!r}") from error


def get_algorithm_id_from_metadata(metadata):
    """Resolve the PSI algorithm recorded in capture metadata."""
    phase_capture_model = metadata.get("phase_capture_model", {})
    algorithm_id = phase_capture_model.get("algorithm_id", DEFAULT_ALGORITHM)
    get_algorithm(algorithm_id)
    return algorithm_id


def reconstruct_phase(frames, algorithm_id=DEFAULT_ALGORITHM):
    """Recover wrapped phase from six intensity frames along axis zero."""
    data = np.asarray(frames, dtype=np.float64)
    if data.ndim < 1 or data.shape[0] != 6:
        raise ValueError(f"Expected six frames along axis zero, got {data.shape}.")

    get_algorithm(algorithm_id)
    if algorithm_id == LEGACY_90_DEG:
        numerator = 4.0 * data[3] - 3.0 * data[1] - data[5]
        denominator = data[0] - 4.0 * data[2] + 3.0 * data[4]
        return np.arctan2(numerator, denominator)

    phase_steps = np.deg2rad(
        np.asarray(ALGORITHMS[algorithm_id]["expected_phase_deg"])
    )
    cosine_component = np.tensordot(
        np.cos(phase_steps), data, axes=(0, 0)
    )
    sine_component = np.tensordot(
        np.sin(phase_steps), data, axes=(0, 0)
    )
    return np.arctan2(-sine_component, cosine_component)