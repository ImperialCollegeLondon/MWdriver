"""Callable wavefront and Zernike aberration calculations."""

from dataclasses import dataclass
from math import factorial

import numpy as np


ZERNIKE_MODES = [
    (1, 0, 0, "Piston"),
    (2, 1, -1, "Tilt Y"),
    (3, 1, 1, "Tilt X"),
    (4, 2, 0, "Defocus"),
    (5, 2, -2, "Oblique Astigmatism"),
    (6, 2, 2, "Vertical Astigmatism"),
    (7, 3, -1, "Vertical Coma"),
    (8, 3, 1, "Horizontal Coma"),
    (9, 3, -3, "Vertical Trefoil"),
    (10, 3, 3, "Oblique Trefoil"),
    (11, 4, 0, "Primary Spherical"),
    (12, 4, -2, "Secondary Astigmatism"),
    (13, 4, 2, "Secondary Astigmatism"),
    (14, 4, -4, "Quadrafoil"),
    (15, 4, 4, "Quadrafoil"),
]


@dataclass(frozen=True)
class AberrationResult:
    fit_mask: np.ndarray
    zernike_coefficients_before_waves: np.ndarray
    zernike_coefficients_after_waves: np.ndarray
    aberration_component_waves: np.ndarray
    tilt_removed_phase: np.ma.MaskedArray
    residual_surface_nm: np.ma.MaskedArray
    wavefront_aberration_nm: np.ma.MaskedArray
    aberration_rms_nm: float
    aberration_pv_nm: float
    pupil_center_x: float
    pupil_center_y: float
    zernike_center_x: float
    zernike_center_y: float
    zernike_radius: float
    tilt_coefficient_x: float
    tilt_coefficient_y: float
    defocus_coefficient: float
    astigmatism_coefficient_x: float
    astigmatism_coefficient_y: float
    spherical_coefficient: float


def zernike_radial(n, m, rho_values):
    m = abs(m)
    if (n - m) % 2 != 0:
        return np.zeros_like(rho_values, dtype=np.float64)

    radial = np.zeros_like(rho_values, dtype=np.float64)
    for index in range((n - m) // 2 + 1):
        numerator = (-1) ** index * factorial(n - index)
        denominator = (
            factorial(index)
            * factorial((n + m) // 2 - index)
            * factorial((n - m) // 2 - index)
        )
        radial += numerator / denominator * rho_values ** (n - 2 * index)
    return radial


def zernike_polynomial(n, m, rho_values, theta_values):
    radial = zernike_radial(n, m, rho_values)
    if m == 0:
        return np.sqrt(n + 1) * radial

    normalization = np.sqrt(2 * (n + 1))
    if m > 0:
        return normalization * radial * np.cos(m * theta_values)
    return normalization * radial * np.sin(abs(m) * theta_values)


def analyze_aberration(
    unwrapped_phase,
    valid_mask,
    wavelength_nm,
    zernike_modes=ZERNIKE_MODES,
):
    phase = np.ma.asarray(unwrapped_phase, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if phase.ndim != 2 or valid_mask.shape != phase.shape:
        raise ValueError("Unwrapped phase and valid mask must be matching 2D arrays.")

    size_y, size_x = phase.shape
    zernike_valid_mask = valid_mask & np.isfinite(phase.filled(np.nan))
    valid_y, valid_x = np.nonzero(zernike_valid_mask)
    if valid_x.size < len(zernike_modes):
        raise ValueError(
            f"Not enough valid ROI pixels for J1-J{len(zernike_modes)} Zernike fitting."
        )

    zernike_center_x = float(np.mean(valid_x))
    zernike_center_y = float(np.mean(valid_y))
    zernike_radius = float(np.max(np.hypot(
        valid_x - zernike_center_x,
        valid_y - zernike_center_y,
    )))
    if zernike_radius <= 0:
        raise ValueError("The ROI pupil radius must be greater than zero.")

    zernike_y_grid, zernike_x_grid = np.indices((size_y, size_x))
    zernike_x_normalized = (zernike_x_grid - zernike_center_x) / zernike_radius
    zernike_y_normalized = (zernike_y_grid - zernike_center_y) / zernike_radius
    zernike_rho = np.hypot(zernike_x_normalized, zernike_y_normalized)
    zernike_theta = np.arctan2(zernike_y_normalized, zernike_x_normalized)
    fit_mask = zernike_valid_mask & (zernike_rho <= 1.0)
    rho_values = zernike_rho[fit_mask]
    theta_values = zernike_theta[fit_mask]
    design_matrix = np.column_stack([
        zernike_polynomial(n, m, rho_values, theta_values)
        for _, n, m, _ in zernike_modes
    ])
    before_waves = phase.filled(np.nan)[fit_mask] / (2.0 * np.pi)
    coefficients_before, _, rank_before, _ = np.linalg.lstsq(
        design_matrix,
        before_waves,
        rcond=None,
    )
    if rank_before != len(zernike_modes):
        raise ValueError(
            f"J1-J{len(zernike_modes)} Zernike fit is rank deficient: "
            f"{rank_before}/{len(zernike_modes)}."
        )

    component_waves = np.asarray([
        np.hypot(coefficients_before[4], coefficients_before[5]),
        np.hypot(coefficients_before[6], coefficients_before[7]),
        np.hypot(coefficients_before[8], coefficients_before[9]),
        abs(coefficients_before[10]),
        np.hypot(coefficients_before[11], coefficients_before[12]),
        np.hypot(coefficients_before[13], coefficients_before[14]),
    ], dtype=np.float64)

    mask = np.logical_not(valid_mask)
    x_grid, y_grid = np.meshgrid(np.arange(size_x), np.arange(size_y))
    x_grid = np.ma.masked_array(x_grid, mask=mask)
    y_grid = np.ma.masked_array(y_grid, mask=mask)
    pupil_center_x = float(np.mean(x_grid))
    pupil_center_y = float(np.mean(y_grid))
    x_normalized = (x_grid - pupil_center_x) * 0.5 / np.std(x_grid - pupil_center_x)
    y_normalized = (y_grid - pupil_center_y) * 0.5 / np.std(y_grid - pupil_center_y)
    radius_grid = np.sqrt(x_normalized ** 2 + y_normalized ** 2)
    theta_grid = np.arctan2(y_normalized, x_normalized)

    phase_zero_piston = phase - np.mean(phase)
    z11 = 2.0 * radius_grid * np.cos(theta_grid)
    z1m1 = 2.0 * radius_grid * np.sin(theta_grid)
    coefficient_tilt_x = np.mean(phase_zero_piston * z11)
    coefficient_tilt_y = np.mean(phase_zero_piston * z1m1)
    tilt_removed = phase_zero_piston - coefficient_tilt_x * z11 - coefficient_tilt_y * z1m1

    after_waves = tilt_removed.filled(np.nan)[fit_mask] / (2.0 * np.pi)
    coefficients_after, _, rank_after, _ = np.linalg.lstsq(
        design_matrix,
        after_waves,
        rcond=None,
    )
    if rank_after != len(zernike_modes):
        raise ValueError(
            f"J1-J{len(zernike_modes)} Zernike fit after piston/tilt removal "
            f"is rank deficient: {rank_after}/{len(zernike_modes)}."
        )

    z20 = np.sqrt(3.0) * (2.0 * radius_grid ** 2 - 1.0)
    z22 = np.sqrt(6.0) * radius_grid ** 2 * np.cos(2.0 * theta_grid)
    z2m2 = np.sqrt(6.0) * radius_grid ** 2 * np.sin(2.0 * theta_grid)
    z40 = np.sqrt(5.0) * (
        6.0 * radius_grid ** 4 - 6.0 * radius_grid ** 2 + 1.0
    )
    coefficient_defocus = np.mean(tilt_removed * z20)
    coefficient_astigmatism_x = np.mean(tilt_removed * z22)
    coefficient_astigmatism_y = np.mean(tilt_removed * z2m2)
    coefficient_spherical = np.mean(tilt_removed * z40)

    residual_phase = (
        tilt_removed
        - coefficient_defocus * z20
        - coefficient_astigmatism_x * z22
        - coefficient_astigmatism_y * z2m2
        - coefficient_spherical * z40
    )
    residual_surface_nm = residual_phase * wavelength_nm / (4.0 * np.pi)
    wavefront_aberration_nm = (
        (tilt_removed - coefficient_defocus * z20)
        * wavelength_nm
        / (2.0 * np.pi)
    )
    aberration_values_nm = wavefront_aberration_nm.compressed()
    aberration_rms_nm = float(np.sqrt(np.mean(aberration_values_nm ** 2)))
    aberration_pv_nm = float(np.max(aberration_values_nm) - np.min(aberration_values_nm))

    return AberrationResult(
        fit_mask=fit_mask,
        zernike_coefficients_before_waves=coefficients_before,
        zernike_coefficients_after_waves=coefficients_after,
        aberration_component_waves=component_waves,
        tilt_removed_phase=tilt_removed,
        residual_surface_nm=residual_surface_nm,
        wavefront_aberration_nm=wavefront_aberration_nm,
        aberration_rms_nm=aberration_rms_nm,
        aberration_pv_nm=aberration_pv_nm,
        pupil_center_x=pupil_center_x,
        pupil_center_y=pupil_center_y,
        zernike_center_x=zernike_center_x,
        zernike_center_y=zernike_center_y,
        zernike_radius=zernike_radius,
        tilt_coefficient_x=float(coefficient_tilt_x),
        tilt_coefficient_y=float(coefficient_tilt_y),
        defocus_coefficient=float(coefficient_defocus),
        astigmatism_coefficient_x=float(coefficient_astigmatism_x),
        astigmatism_coefficient_y=float(coefficient_astigmatism_y),
        spherical_coefficient=float(coefficient_spherical),
    )
