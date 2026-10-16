import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mwdriver.analysis.phase_aberration import (
    ZERNIKE_MODES,
    analyze_aberration,
    zernike_polynomial,
    zernike_polynomial_cartesian,
)


class PhaseAberrationTests(unittest.TestCase):
    def make_pupil(self):
        height = width = 64
        yy, xx = np.indices((height, width))
        center_y = center_x = 31.5
        rho = np.hypot(yy - center_y, xx - center_x) / 25.0
        valid = rho <= 1.0
        phase = np.ma.masked_array(
            0.8 * (yy - center_y) / 25.0
            + 0.3 * (xx - center_x) / 25.0
            + 0.15 * rho ** 2,
            mask=~valid,
        )
        return phase, valid

    def test_analysis_returns_zernike_tilt_and_surface_metrics(self):
        phase, valid = self.make_pupil()
        result = analyze_aberration(phase, valid, wavelength_nm=632.8)

        self.assertEqual(result.fit_mask.shape, phase.shape)
        self.assertEqual(result.zernike_coefficients_before_waves.shape, (28,))
        self.assertEqual(result.zernike_coefficients_after_waves.shape, (28,))
        self.assertEqual(result.aberration_component_waves.shape, (7,))
        self.assertEqual(result.tilt_removed_phase.shape, phase.shape)
        self.assertEqual(result.residual_surface_nm.shape, phase.shape)
        self.assertEqual(result.wavefront_aberration_nm.shape, phase.shape)
        np.testing.assert_array_equal(result.fit_mask, valid)
        self.assertTrue(np.isfinite(result.aberration_rms_nm))
        self.assertTrue(np.isfinite(result.aberration_pv_nm))
        self.assertGreaterEqual(result.aberration_pv_nm, 0.0)
        self.assertEqual(len(ZERNIKE_MODES), 28)
        self.assertAlmostEqual(result.pupil_center_x, 31.5)
        self.assertAlmostEqual(result.pupil_center_y, 31.5)

    def test_analysis_recovers_secondary_spherical_j22_content(self):
        yy, xx = np.indices((64, 64))
        center = 31.5
        radius = 25.0
        pupil_radius = np.hypot(yy - center, xx - center)
        valid = pupil_radius <= radius
        valid_y, valid_x = np.nonzero(valid)
        fit_center_x = np.mean(valid_x)
        fit_center_y = np.mean(valid_y)
        fit_radius = np.max(
            np.hypot(valid_x - fit_center_x, valid_y - fit_center_y)
        )
        normalized_x = (xx - fit_center_x) / fit_radius
        normalized_y = (yy - fit_center_y) / fit_radius
        normalized_radius = np.hypot(normalized_x, normalized_y)
        target_coefficient_waves = 0.37
        phase = np.ma.masked_array(
            2.0
            * np.pi
            * target_coefficient_waves
            * zernike_polynomial(
                6,
                0,
                normalized_radius,
                np.arctan2(normalized_y, normalized_x),
            ),
            mask=~valid,
        )

        result = analyze_aberration(phase, valid, wavelength_nm=632.8)

        self.assertAlmostEqual(
            result.secondary_spherical_coefficient_waves,
            target_coefficient_waves,
            places=10,
        )
        self.assertAlmostEqual(result.aberration_component_waves[6], 0.37)

    def test_cartesian_zernike_polynomials_match_standard_modes(self):
        x_values = np.linspace(-1.0, 1.0, 31)
        y_values = np.linspace(-1.0, 1.0, 31)
        xx, yy = np.meshgrid(x_values, y_values)
        rho = np.hypot(xx, yy)
        theta = np.arctan2(yy, xx)

        for _, n, m, _ in ZERNIKE_MODES:
            with self.subTest(n=n, m=m):
                np.testing.assert_allclose(
                    zernike_polynomial_cartesian(n, m, xx, yy),
                    zernike_polynomial(n, m, rho, theta),
                    atol=1e-12,
                )

    def test_ellipse_scaled_cartesian_fit_recovers_known_mode(self):
        yy, xx = np.indices((80, 80), dtype=float)
        center_y = 39.0
        center_x = 40.0
        angle = np.deg2rad(32.0)
        major_radius = 25.0
        minor_radius = 13.0
        first_axis = np.asarray([
            np.sin(angle) * major_radius,
            np.cos(angle) * major_radius,
        ])
        second_axis = np.asarray([
            np.cos(angle) * minor_radius,
            -np.sin(angle) * minor_radius,
        ])
        center = np.asarray([center_y, center_x])
        vertices = np.asarray([
            center - first_axis - second_axis,
            center + first_axis - second_axis,
            center + first_axis + second_axis,
            center - first_axis + second_axis,
        ])
        transform = np.column_stack((first_axis, second_axis))
        local_coordinates = np.linalg.inv(transform) @ np.stack((
            yy - center_y,
            xx - center_x,
        )).reshape(2, -1)
        local_x = local_coordinates[0].reshape(xx.shape)
        local_y = local_coordinates[1].reshape(yy.shape)
        valid = local_x ** 2 + local_y ** 2 <= 1.0
        mode_n, mode_m = 4, 2
        target_coefficient_waves = 0.31
        phase = np.ma.masked_array(
            2.0
            * np.pi
            * target_coefficient_waves
            * zernike_polynomial_cartesian(
                mode_n,
                mode_m,
                local_x,
                local_y,
            ),
            mask=~valid,
        )

        result = analyze_aberration(
            phase,
            valid,
            wavelength_nm=632.8,
            pupil_vertices=vertices,
        )

        mode_index = next(
            index
            for index, (_, n, m, _) in enumerate(ZERNIKE_MODES)
            if (n, m) == (mode_n, mode_m)
        )
        self.assertAlmostEqual(
            result.zernike_coefficients_before_waves[mode_index],
            target_coefficient_waves,
            places=10,
        )
        np.testing.assert_allclose(
            np.delete(
                result.zernike_coefficients_before_waves,
                mode_index,
            ),
            0.0,
            atol=1e-10,
        )

    def test_tilt_removal_eliminates_plane_on_tilted_ellipse_mask(self):
        yy, xx = np.indices((64, 64), dtype=float)
        center_x = 31.0
        center_y = 32.0
        angle = np.deg2rad(35.0)
        cosine = np.cos(angle)
        sine = np.sin(angle)
        delta_x = xx - center_x
        delta_y = yy - center_y
        major_coordinate = cosine * delta_x + sine * delta_y
        minor_coordinate = -sine * delta_x + cosine * delta_y
        valid = (
            (major_coordinate / 22.0) ** 2
            + (minor_coordinate / 12.0) ** 2
            <= 1.0
        )
        first_axis = np.asarray([sine * 22.0, cosine * 22.0])
        second_axis = np.asarray([cosine * 12.0, -sine * 12.0])
        center = np.asarray([center_y, center_x])
        vertices = np.asarray([
            center - first_axis - second_axis,
            center + first_axis - second_axis,
            center + first_axis + second_axis,
            center - first_axis + second_axis,
        ])
        phase_values = 1.7 + 0.08 * delta_x - 0.13 * delta_y
        phase = np.ma.masked_array(phase_values, mask=~valid)

        result = analyze_aberration(
            phase,
            valid,
            wavelength_nm=632.8,
            pupil_vertices=vertices,
        )

        self.assertLess(
            np.max(np.abs(result.tilt_removed_phase.compressed())),
            1e-10,
        )

    def test_defocus_removal_uses_fitted_zernike_on_elliptical_mask(self):
        yy, xx = np.indices((64, 64), dtype=float)
        center_x = 31.0
        center_y = 32.0
        angle = np.deg2rad(35.0)
        cosine = np.cos(angle)
        sine = np.sin(angle)
        delta_x = xx - center_x
        delta_y = yy - center_y
        major_coordinate = cosine * delta_x + sine * delta_y
        minor_coordinate = -sine * delta_x + cosine * delta_y
        valid = (
            (major_coordinate / 22.0) ** 2
            + (minor_coordinate / 12.0) ** 2
            <= 1.0
        )
        first_axis = np.asarray([sine * 22.0, cosine * 22.0])
        second_axis = np.asarray([cosine * 12.0, -sine * 12.0])
        center = np.asarray([center_y, center_x])
        vertices = np.asarray([
            center - first_axis - second_axis,
            center + first_axis - second_axis,
            center + first_axis + second_axis,
            center - first_axis + second_axis,
        ])
        valid_y, valid_x = np.nonzero(valid)
        pupil_center_x = np.mean(valid_x)
        pupil_center_y = np.mean(valid_y)
        pupil_radius = np.max(
            np.hypot(valid_x - pupil_center_x, valid_y - pupil_center_y)
        )
        normalized_x = (xx - pupil_center_x) / pupil_radius
        normalized_y = (yy - pupil_center_y) / pupil_radius
        rho = np.hypot(normalized_x, normalized_y)
        theta = np.arctan2(normalized_y, normalized_x)
        target_defocus_waves = 0.2
        phase = np.ma.masked_array(
            2.0
            * np.pi
            * target_defocus_waves
            * zernike_polynomial(2, 0, rho, theta),
            mask=~valid,
        )

        result = analyze_aberration(
            phase,
            valid,
            wavelength_nm=632.8,
            pupil_vertices=vertices,
        )

        self.assertLess(
            np.max(np.abs(result.wavefront_aberration_nm.compressed())),
            1e-8,
        )
        np.testing.assert_allclose(
            result.zernike_coefficients_after_waves[:4],
            0.0,
            atol=1e-10,
        )

    def test_defocus_removal_does_not_increase_residual_on_tilted_ellipse(self):
        yy, xx = np.indices((80, 80), dtype=float)
        center_x = 39.0
        center_y = 40.0
        angle = np.deg2rad(35.0)
        delta_x = xx - center_x
        delta_y = yy - center_y
        major = np.cos(angle) * delta_x + np.sin(angle) * delta_y
        minor = -np.sin(angle) * delta_x + np.cos(angle) * delta_y
        valid = (major / 26.0) ** 2 + (minor / 13.0) ** 2 <= 1.0
        first_axis = np.asarray([np.sin(angle) * 26.0, np.cos(angle) * 26.0])
        second_axis = np.asarray([np.cos(angle) * 13.0, -np.sin(angle) * 13.0])
        center = np.asarray([center_y, center_x])
        vertices = np.asarray([
            center - first_axis - second_axis,
            center + first_axis - second_axis,
            center + first_axis + second_axis,
            center - first_axis + second_axis,
        ])
        valid_y, valid_x = np.nonzero(valid)
        center_fit_x = np.mean(valid_x)
        center_fit_y = np.mean(valid_y)
        radius = np.max(
            np.hypot(valid_x - center_fit_x, valid_y - center_fit_y)
        )
        normalized_x = (xx - center_fit_x) / radius
        normalized_y = (yy - center_fit_y) / radius
        rho = np.hypot(normalized_x, normalized_y)
        theta = np.arctan2(normalized_y, normalized_x)
        phase = np.ma.masked_array(
            0.2 * zernike_polynomial(2, 0, rho, theta),
            mask=~valid,
        )

        result = analyze_aberration(
            phase,
            valid,
            wavelength_nm=632.8,
            pupil_vertices=vertices,
        )

        before_std = np.std(result.tilt_removed_phase.compressed())
        after_std = np.std(
            result.wavefront_aberration_nm.compressed() * 2.0 * np.pi / 632.8
        )
        self.assertLess(after_std, before_std * 1e-8)

    def test_low_order_removal_is_roi_least_squares_with_spherical_content(self):
        yy, xx = np.indices((80, 80), dtype=float)
        center_x = 39.0
        center_y = 40.0
        angle = np.deg2rad(35.0)
        delta_x = xx - center_x
        delta_y = yy - center_y
        major = np.cos(angle) * delta_x + np.sin(angle) * delta_y
        minor = -np.sin(angle) * delta_x + np.cos(angle) * delta_y
        valid = (major / 26.0) ** 2 + (minor / 13.0) ** 2 <= 1.0
        first_axis = np.asarray([np.sin(angle) * 26.0, np.cos(angle) * 26.0])
        second_axis = np.asarray([np.cos(angle) * 13.0, -np.sin(angle) * 13.0])
        center = np.asarray([center_y, center_x])
        vertices = np.asarray([
            center - first_axis - second_axis,
            center + first_axis - second_axis,
            center + first_axis + second_axis,
            center - first_axis + second_axis,
        ])
        valid_y, valid_x = np.nonzero(valid)
        center_fit_x = np.mean(valid_x)
        center_fit_y = np.mean(valid_y)
        radius = np.max(
            np.hypot(valid_x - center_fit_x, valid_y - center_fit_y)
        )
        normalized_x = (xx - center_fit_x) / radius
        normalized_y = (yy - center_fit_y) / radius
        rho = np.hypot(normalized_x, normalized_y)
        theta = np.arctan2(normalized_y, normalized_x)
        defocus = 0.2 * zernike_polynomial(2, 0, rho, theta)
        spherical = 0.25 * zernike_polynomial(4, 0, rho, theta)
        phase = np.ma.masked_array(defocus + spherical, mask=~valid)

        result = analyze_aberration(
            phase,
            valid,
            wavelength_nm=632.8,
            pupil_vertices=vertices,
        )

        valid_x = normalized_x[valid]
        valid_y = normalized_y[valid]
        correction_design = np.column_stack((
            np.ones(valid_x.size),
            valid_x,
            valid_y,
            valid_x ** 2 + valid_y ** 2,
        ))
        correction_coefficients, _, _, _ = np.linalg.lstsq(
            correction_design,
            phase.data[valid],
            rcond=None,
        )
        expected_phase = (
            phase.data[valid] - correction_design @ correction_coefficients
        )
        actual_phase = (
            result.wavefront_aberration_nm.compressed() * 2.0 * np.pi / 632.8
        )
        np.testing.assert_allclose(actual_phase, expected_phase, atol=1e-9)
        self.assertEqual(result.zernike_coefficients_after_waves.shape, (28,))

    def test_analysis_rejects_nonmatching_phase_and_mask_shapes(self):
        with self.assertRaisesRegex(ValueError, "matching 2D"):
            analyze_aberration(
                np.zeros((8, 8)),
                np.ones((7, 8), dtype=bool),
                wavelength_nm=632.8,
            )

    def test_analysis_rejects_too_few_valid_pixels(self):
        phase = np.ma.masked_array(np.zeros((8, 8)), mask=True)
        valid = np.zeros((8, 8), dtype=bool)
        with self.assertRaisesRegex(ValueError, "Not enough valid ROI pixels"):
            analyze_aberration(phase, valid, wavelength_nm=632.8)


if __name__ == "__main__":
    unittest.main()
