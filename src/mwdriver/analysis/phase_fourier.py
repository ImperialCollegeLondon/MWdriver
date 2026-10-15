"""Pure Fourier carrier detection for phase-capture analysis."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FourierCarrier:
    spectrum: np.ndarray
    peak_y: int
    peak_x: int
    center_y: int
    center_x: int
    peak_distance: float
    mask_radius: int


def detect_fourier_carrier(frame, dc_radius=35, lower_half_margin=10, refine_radius=45):
    frame = np.asarray(frame)
    if frame.ndim != 2:
        raise ValueError("Fourier carrier detection expects a 2D frame.")

    height, width = frame.shape
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(frame)))
    center_y = height // 2
    center_x = width // 2
    yy, xx = np.ogrid[:height, :width]
    dc_region = (
        (yy - center_y) ** 2 + (xx - center_x) ** 2
    ) <= dc_radius ** 2
    search_spectrum = spectrum.copy()
    search_spectrum[dc_region] = 0
    search_spectrum[:center_y + lower_half_margin, :] = 0
    coarse_peak_y, coarse_peak_x = np.unravel_index(
        np.argmax(search_spectrum),
        search_spectrum.shape,
    )

    refine_window = (
        (yy - coarse_peak_y) ** 2 + (xx - coarse_peak_x) ** 2
    ) <= refine_radius ** 2
    refine_spectrum = spectrum.copy()
    refine_spectrum[~refine_window] = 0
    refine_spectrum[dc_region] = 0
    peak_y, peak_x = np.unravel_index(
        np.argmax(refine_spectrum),
        refine_spectrum.shape,
    )
    peak_distance = float(np.hypot(peak_y - center_y, peak_x - center_x))
    mask_radius = int(np.clip(0.35 * peak_distance, 6, 30))
    return FourierCarrier(
        spectrum=spectrum,
        peak_y=int(peak_y),
        peak_x=int(peak_x),
        center_y=int(center_y),
        center_x=int(center_x),
        peak_distance=peak_distance,
        mask_radius=mask_radius,
    )
