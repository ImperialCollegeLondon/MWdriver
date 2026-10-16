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
    (16, 5, -1, "Secondary Vertical Coma"),
    (17, 5, 1, "Secondary Horizontal Coma"),
    (18, 5, -3, "Secondary Vertical Trefoil"),
    (19, 5, 3, "Secondary Oblique Trefoil"),
    (20, 5, -5, "Pentafoil"),
    (21, 5, 5, "Pentafoil"),
    (22, 6, 0, "Secondary Spherical"),
    (23, 6, -2, "Tertiary Astigmatism"),
    (24, 6, 2, "Tertiary Astigmatism"),
    (25, 6, -4, "Secondary Quadrafoil"),
    (26, 6, 4, "Secondary Quadrafoil"),
    (27, 6, -6, "Hexafoil"),
    (28, 6, 6, "Hexafoil"),
]


@dataclass(frozen=True)
class AberrationResult:
    fit_mask: np.ndarray
    zernike_coefficients_before_waves: np.ndarray
    zernike_coefficients_after_waves: np.ndarray
    aberration_component_waves: np.ndarray
    secondary_spherical_coefficient_waves: float
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


def zernike_polynomial_cartesian(n, m, x_values, y_values):
    """Evaluate normalized Zernike modes as Cartesian polynomials."""
    absolute_m = abs(m)
    if (n - absolute_m) % 2 != 0:
        return np.zeros_like(x_values, dtype=np.float64)

    radius_squared = x_values ** 2 + y_values ** 2
    polynomial = np.zeros_like(radius_squared, dtype=np.float64)
    for index in range((n - absolute_m) // 2 + 1):
        numerator = (-1) ** index * factorial(n - index)
        denominator = (
            factorial(index)
            * factorial((n + absolute_m) // 2 - index)
            * factorial((n - absolute_m) // 2 - index)
        )
        radial_power = n - 2 * index
        radial_coefficient = numerator / denominator
        if m == 0:
            polynomial += (
                radial_coefficient
                * radius_squared ** (radial_power // 2)
            )
            continue

        angular_polynomial = (x_values + 1j * y_values) ** absolute_m
        angular_component = (
            angular_polynomial.imag if m < 0 else angular_polynomial.real
        )
        polynomial += (
            radial_coefficient
            * radius_squared ** ((radial_power - absolute_m) // 2)
            * angular_component
        )

    normalization = (
        np.sqrt(n + 1)
        if m == 0
        else np.sqrt(2 * (n + 1))
    )
    return normalization * polynomial


def analyze_aberration(
    unwrapped_phase,
    valid_mask,
    wavelength_nm,
    zernike_modes=ZERNIKE_MODES,
    pupil_vertices=None,
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
    pupil_center_x = float(np.mean(valid_x))
    pupil_center_y = float(np.mean(valid_y))
    pupil_radius = float(np.max(np.hypot(
        valid_x - pupil_center_x,
        valid_y - pupil_center_y,
    )))
    if pupil_radius <= 0:
        raise ValueError("The ROI pupil radius must be greater than zero.")

    zernike_y_grid, zernike_x_grid = np.indices((size_y, size_x))
    if pupil_vertices is None:
        zernike_center_x = float(np.mean(valid_x))
        zernike_center_y = float(np.mean(valid_y))
        zernike_radius = float(np.max(np.hypot(
            valid_x - zernike_center_x,
            valid_y - zernike_center_y,
        )))
        if zernike_radius <= 0:
            raise ValueError("The ROI pupil radius must be greater than zero.")
        zernike_x_normalized = (
            zernike_x_grid - zernike_center_x
        ) / zernike_radius
        zernike_y_normalized = (
            zernike_y_grid - zernike_center_y
        ) / zernike_radius
        zernike_radius_x = zernike_radius
        zernike_radius_y = zernike_radius
    else:
        vertices = np.asarray(pupil_vertices, dtype=np.float64)
        if vertices.shape != (4, 2):
            raise ValueError("Pupil vertices must have shape (4, 2) in y/x order.")
        pupil_center_y, pupil_center_x = np.mean(vertices, axis=0)
        first_axis = (vertices[1] - vertices[0]) / 2.0
        second_axis = (vertices[2] - vertices[1]) / 2.0
        pupil_transform = np.column_stack((first_axis, second_axis))
        determinant = float(np.linalg.det(pupil_transform))
        if abs(determinant) <= np.finfo(np.float64).eps:
            raise ValueError("The ellipse pupil geometry is degenerate.")
        inverse_transform = np.linalg.inv(pupil_transform)
        delta_y = zernike_y_grid - pupil_center_y
        delta_x = zernike_x_grid - pupil_center_x
        zernike_x_normalized = (
            inverse_transform[0, 0] * delta_y
            + inverse_transform[0, 1] * delta_x
        )
        zernike_y_normalized = (
            inverse_transform[1, 0] * delta_y
            + inverse_transform[1, 1] * delta_x
        )
        zernike_center_x = float(pupil_center_x)
        zernike_center_y = float(pupil_center_y)
        zernike_radius_x = float(np.linalg.norm(first_axis))
        zernike_radius_y = float(np.linalg.norm(second_axis))

    zernike_rho = np.hypot(zernike_x_normalized, zernike_y_normalized)
    zernike_theta = np.arctan2(zernike_y_normalized, zernike_x_normalized)
    fit_mask = zernike_valid_mask & (zernike_rho <= 1.0)
    x_values = zernike_x_normalized[fit_mask]
    y_values = zernike_y_normalized[fit_mask]
    design_matrix = np.column_stack([
        zernike_polynomial_cartesian(n, m, x_values, y_values)
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

    mask = np.logical_not(zernike_valid_mask)
    phase_values = phase.filled(np.nan)[zernike_valid_mask]
    correction_x_values = (valid_x - pupil_center_x) / pupil_radius
    correction_y_values = (valid_y - pupil_center_y) / pupil_radius
    rho_squared_values = correction_x_values ** 2 + correction_y_values ** 2

    tilt_design_matrix = np.column_stack((
        np.ones(valid_x.size),
        correction_x_values,
        correction_y_values,
    ))
    piston_and_tilt, _, tilt_rank, _ = np.linalg.lstsq(
        tilt_design_matrix,
        phase_values,
        rcond=None,
    )
    if tilt_rank != 3:
        raise ValueError("Piston and tilt fit is rank deficient.")

    correction_design_matrix = np.column_stack((
        np.ones(valid_x.size),
        correction_x_values,
        correction_y_values,
        rho_squared_values,
    ))
    correction_coefficients, _, correction_rank, _ = np.linalg.lstsq(
        correction_design_matrix,
        phase_values,
        rcond=None,
    )
    if correction_rank != 4:
        raise ValueError("Piston, tilt and defocus fit is rank deficient.")

    normalized_x_grid = np.ma.array(
        (zernike_x_grid - pupil_center_x) / pupil_radius,
        mask=~zernike_valid_mask,
    )
    normalized_y_grid = np.ma.array(
        (zernike_y_grid - pupil_center_y) / pupil_radius,
        mask=~zernike_valid_mask,
    )
    rho_squared_grid = normalized_x_grid ** 2 + normalized_y_grid ** 2
    zernike_x_grid_masked = np.ma.array(
        zernike_x_normalized,
        mask=~zernike_valid_mask,
    )
    zernike_y_grid_masked = np.ma.array(
        zernike_y_normalized,
        mask=~zernike_valid_mask,
    )
    tilt_removed = (
        phase
        - piston_and_tilt[0]
        - piston_and_tilt[1] * normalized_x_grid
        - piston_and_tilt[2] * normalized_y_grid
    )
    corrected_phase = (
        phase
        - correction_coefficients[0]
        - correction_coefficients[1] * normalized_x_grid
        - correction_coefficients[2] * normalized_y_grid
        - correction_coefficients[3] * rho_squared_grid
    )

    after_waves = corrected_phase.filled(np.nan)[fit_mask] / (2.0 * np.pi)
    coefficients_after, _, rank_after, _ = np.linalg.lstsq(
        design_matrix,
        after_waves,
        rcond=None,
    )
    if rank_after != len(zernike_modes):
        raise ValueError(
            f"J1-J{len(zernike_modes)} Zernike fit after piston/tilt/defocus removal "
            f"is rank deficient: {rank_after}/{len(zernike_modes)}."
        )

    component_waves = np.asarray([
        np.hypot(coefficients_after[4], coefficients_after[5]),
        np.hypot(coefficients_after[6], coefficients_after[7]),
        np.hypot(coefficients_after[8], coefficients_after[9]),
        abs(coefficients_after[10]),
        np.hypot(coefficients_after[11], coefficients_after[12]),
        np.hypot(coefficients_after[13], coefficients_after[14]),
        abs(coefficients_after[21]),
    ], dtype=np.float64)

    z2m2 = zernike_polynomial_cartesian(
        2, -2, zernike_x_grid_masked, zernike_y_grid_masked
    )
    z22 = zernike_polynomial_cartesian(
        2, 2, zernike_x_grid_masked, zernike_y_grid_masked
    )
    z40 = zernike_polynomial_cartesian(
        4, 0, zernike_x_grid_masked, zernike_y_grid_masked
    )
    coefficient_defocus = correction_coefficients[3]
    coefficient_astigmatism_y = 2.0 * np.pi * coefficients_after[4]
    coefficient_astigmatism_x = 2.0 * np.pi * coefficients_after[5]
    coefficient_spherical = 2.0 * np.pi * coefficients_after[10]

    residual_phase = (
        corrected_phase
        - coefficient_astigmatism_x * z22
        - coefficient_astigmatism_y * z2m2
        - coefficient_spherical * z40
    )
    residual_phase = residual_phase - np.ma.mean(residual_phase)
    residual_surface_nm = residual_phase * wavelength_nm / (4.0 * np.pi)
    wavefront_aberration_nm = (
        corrected_phase * wavelength_nm / (2.0 * np.pi)
    )
    aberration_values_nm = wavefront_aberration_nm.compressed()
    aberration_rms_nm = float(np.sqrt(np.mean(aberration_values_nm ** 2)))
    aberration_pv_nm = float(np.max(aberration_values_nm) - np.min(aberration_values_nm))

    return AberrationResult(
        fit_mask=fit_mask,
        zernike_coefficients_before_waves=coefficients_before,
        zernike_coefficients_after_waves=coefficients_after,
        aberration_component_waves=component_waves,
        secondary_spherical_coefficient_waves=float(coefficients_after[21]),
        tilt_removed_phase=tilt_removed,
        residual_surface_nm=residual_surface_nm,
        wavefront_aberration_nm=wavefront_aberration_nm,
        aberration_rms_nm=aberration_rms_nm,
        aberration_pv_nm=aberration_pv_nm,
        pupil_center_x=pupil_center_x,
        pupil_center_y=pupil_center_y,
        zernike_center_x=zernike_center_x,
        zernike_center_y=zernike_center_y,
        zernike_radius=float(np.sqrt(zernike_radius_x * zernike_radius_y)),
        tilt_coefficient_x=float(correction_coefficients[1]),
        tilt_coefficient_y=float(correction_coefficients[2]),
        defocus_coefficient=float(coefficient_defocus),
        astigmatism_coefficient_x=float(coefficient_astigmatism_x),
        astigmatism_coefficient_y=float(coefficient_astigmatism_y),
        spherical_coefficient=float(coefficient_spherical),
    )
