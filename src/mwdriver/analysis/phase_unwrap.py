"""Center-out phase unwrapping calculations for six-frame analysis."""

import numpy as np


def unwrap_phase_from_center(wrapped_phase, mask, center_y, center_x):
    phase = np.ma.asarray(wrapped_phase, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if phase.ndim != 2 or mask.shape != phase.shape:
        raise ValueError("Wrapped phase and mask must be matching 2D arrays.")
    if not (0 <= center_y < phase.shape[0] and 0 <= center_x < phase.shape[1]):
        raise ValueError("Unwrap center must be inside the phase image.")

    unwrapped = np.zeros_like(phase, dtype=np.float64)
    unwrapped[center_y:, :] = np.unwrap(
        phase[center_y:, :].filled(0.0),
        axis=0,
    )
    unwrapped[center_y::-1, :] = np.unwrap(
        phase[center_y::-1, :].filled(0.0),
        axis=0,
    )
    vertical_unwrapped = np.ma.masked_array(unwrapped.copy(), mask=mask)
    unwrapped[:, center_x:] = np.unwrap(unwrapped[:, center_x:], axis=1)
    unwrapped[:, center_x::-1] = np.unwrap(
        unwrapped[:, center_x::-1],
        axis=1,
    )
    return vertical_unwrapped, np.ma.masked_array(unwrapped, mask=mask)
