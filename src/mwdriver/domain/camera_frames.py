"""Pure camera-frame display preparation helpers."""

import numpy as np


def prepare_display_frame(raw_frame, display_downsample):
    if display_downsample < 1:
        raise ValueError("Display downsample must be at least 1.")

    if display_downsample > 1:
        display_frame = raw_frame[
            ::display_downsample,
            ::display_downsample,
        ]
    else:
        display_frame = raw_frame

    return np.ascontiguousarray(display_frame)