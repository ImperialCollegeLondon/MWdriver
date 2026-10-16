"""Center-out phase unwrapping calculations for six-frame analysis."""

import numpy as np
from skimage.restoration import unwrap_phase


def _unwrap_line_from_valid_midpoint(line, valid):
    result = np.zeros_like(line, dtype=np.float64)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        return result

    segment_boundaries = np.flatnonzero(np.diff(valid_indices) > 1) + 1
    for segment in np.split(valid_indices, segment_boundaries):
        midpoint = (segment.size - 1) // 2
        center_index = segment[midpoint]
        forward_indices = segment[midpoint:]
        backward_indices = segment[:midpoint + 1][::-1]
        result[forward_indices] = np.unwrap(line[forward_indices])
        result[backward_indices] = np.unwrap(line[backward_indices])
    return result


def unwrap_phase_from_center(wrapped_phase, mask):
    phase = np.ma.asarray(wrapped_phase, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if phase.ndim != 2 or mask.shape != phase.shape:
        raise ValueError("Wrapped phase and mask must be matching 2D arrays.")
    if np.ma.getmaskarray(phase).shape != mask.shape:
        raise ValueError("Wrapped phase and mask must be matching 2D arrays.")

    invalid = mask | np.ma.getmaskarray(phase)
    phase_values = phase.filled(0.0)
    invalid |= ~np.isfinite(phase_values)
    valid = ~invalid

    vertical_values = np.zeros_like(phase_values, dtype=np.float64)
    for column in range(phase.shape[1]):
        vertical_values[:, column] = _unwrap_line_from_valid_midpoint(
            phase_values[:, column],
            valid[:, column],
        )

    vertical_unwrapped = np.ma.masked_array(
        vertical_values,
        mask=invalid,
    )
    unwrapped = unwrap_phase(
        np.ma.array(phase_values, mask=invalid),
        rng=0,
    )

    return vertical_unwrapped, np.ma.masked_array(unwrapped, mask=invalid)
