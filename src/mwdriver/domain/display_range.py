"""Robust display ranges for scientific image data."""

import numpy as np


LOWER_DISPLAY_PERCENTILE = 0.1
UPPER_DISPLAY_PERCENTILE = 99.9


def percentile_display_limits(values):
    values = np.asarray(values, dtype=np.float64)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return None

    lower, upper = np.percentile(
        finite_values,
        [LOWER_DISPLAY_PERCENTILE, UPPER_DISPLAY_PERCENTILE],
    )
    lower = float(lower)
    upper = float(upper)
    return lower, upper