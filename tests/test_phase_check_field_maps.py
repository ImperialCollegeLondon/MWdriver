import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from matplotlib.figure import Figure
from matplotlib.image import imread

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mwdriver.analysis.check_measured_vs_expected_phase import (
    create_field_maps,
    create_fourier_filter_plot,
    extract_complex_fields,
    estimate_phase_curve,
    _resolve_phase_roi,
)


class FieldMapTests(unittest.TestCase):
    def test_default_roi_is_central_400_pixel_square_on_camera_frames(self):
        roi = _resolve_phase_roi((1080, 1440), None)
        expected = np.zeros((1080, 1440), dtype=bool)
        expected[340:740, 520:920] = True
        np.testing.assert_array_equal(roi, expected)

    def test_default_roi_matches_explicit_roi_before_fft_and_correlation(self):
        rows, columns = np.indices((96, 128))
        roi = np.zeros((96, 128), dtype=bool)
        roi[24:72, 40:88] = True
        carrier = 2 * np.pi * (20 * columns / 128 + 12 * rows / 96)
        stack = np.asarray([100 + 20 * np.cos(carrier + phase) for phase in (0, 0.8, 1.6)])
        with patch.object(np.fft, "fft2", wraps=np.fft.fft2) as fft:
            measured_default, default_info = estimate_phase_curve(stack)
        centred = fft.call_args.args[0]
        np.testing.assert_array_equal(centred[:, ~roi], 0)
        np.testing.assert_allclose(centred[:, roi].mean(axis=1), 0, atol=1e-12)
        measured_explicit, explicit_info = estimate_phase_curve(stack, roi_mask=roi)
        np.testing.assert_array_equal(measured_default, measured_explicit)
        np.testing.assert_array_equal(default_info["complex_fields"], explicit_info["complex_fields"])

    def test_fft_centres_roi_intensity_and_zeros_outside_roi(self):
        rows, columns = np.indices((96, 128))
        roi = (rows - 48) ** 2 + (columns - 64) ** 2 < 25 ** 2
        carrier = 2 * np.pi * (20 * columns / 128 + 12 * rows / 96)
        stack = np.asarray([
            20 * np.cos(carrier + phase) + np.where(roi, 200 + phase, 50 - phase)
            for phase in (0, 1, 2)
        ])
        for mask in (roi, np.ones_like(roi)):
            with self.subTest(full_frame=bool(np.all(mask))):
                if np.all(mask):
                    stack = np.asarray([200 + 20 * np.cos(carrier + phase) for phase in (0, 1, 2)])
                original = stack.copy()
                with patch.object(np.fft, "fft2", wraps=np.fft.fft2) as fft:
                    extract_complex_fields(stack, roi_mask=mask)
                centred = fft.call_args.args[0]
                np.testing.assert_allclose(centred[:, mask].mean(axis=1), 0, atol=1e-12)
                np.testing.assert_allclose(
                    centred[:, mask],
                    stack[:, mask] - stack[:, mask].mean(axis=1, keepdims=True),
                )
                if np.any(~mask):
                    np.testing.assert_array_equal(centred[:, ~mask], 0)
            np.testing.assert_array_equal(stack, original)

    def test_roi_background_subtraction_rejects_empty_or_wrong_shape_masks(self):
        stack = np.ones((2, 20, 30))
        for mask in (np.zeros((20, 30), dtype=bool), np.ones((10, 30), dtype=bool)):
            with self.subTest(shape=mask.shape):
                with self.assertRaises(ValueError):
                    extract_complex_fields(stack, roi_mask=mask)

    def test_half_plane_rejects_conjugate_and_retains_broad_sideband(self):
        rows, columns = np.indices((96, 128))
        phases = np.deg2rad([0, 60, 120, 180, 240, 300])
        amplitude = 20 * (1 + 0.3 * np.cos(2 * np.pi * 6 * columns / 128))
        for carrier_x, carrier_y in ((20, 12), (-20, 12), (20, -12)):
            with self.subTest(carrier=(carrier_x, carrier_y)):
                carrier = 2 * np.pi * (carrier_x * columns / 128 + carrier_y * rows / 96)
                stack = np.asarray([100 + amplitude * np.cos(carrier + phase) for phase in phases])
                measured, diagnostics = estimate_phase_curve(stack, roi_mask=np.ones(stack.shape[1:], dtype=bool))
                np.testing.assert_allclose(measured, np.rad2deg(phases), atol=1e-8)
                mask = diagnostics["fourier_mask"]
                centre_x, centre_y = diagnostics["dc_x"], diagnostics["dc_y"]
                peak_x, peak_y = diagnostics["peak_x"], diagnostics["peak_y"]
                self.assertEqual(mask[centre_y, centre_x], 0)
                self.assertEqual(mask[2 * centre_y - peak_y, 2 * centre_x - peak_x], 0)
                self.assertGreater(mask[peak_y, peak_x], 0.98)
                self.assertGreater(mask[peak_y, peak_x + 6], 0.9)
                self.assertGreater(mask[peak_y, peak_x - 6], 0.9)
                recovered = np.abs(diagnostics["complex_fields"][0])
                self.assertLess(float(np.max(np.abs(recovered - amplitude / 2))), 0.15)

    def test_fourier_plot_displays_actual_extraction_mask_and_filtered_spectrum(self):
        rows, columns = np.indices((96, 128))
        stack = np.asarray([
            100 + 20 * np.cos(2 * np.pi * (12 * columns / 128 + 9 * rows / 96) + step)
            for step in (0, 1)
        ])
        fields, diagnostics = extract_complex_fields(stack)
        mask = diagnostics["fourier_mask"]
        spectrum = diagnostics["spectrum"]
        np.testing.assert_allclose(np.abs(np.fft.fftshift(np.fft.fft2(fields[0]))), spectrum * mask, atol=1e-8)
        figures = []
        savefig = Figure.savefig

        def capture(figure, *args, **kwargs):
            figures.append(figure)
            return savefig(figure, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(Figure, "savefig", new=capture):
                output = create_fourier_filter_plot(diagnostics, directory)
            self.assertGreater(float(np.std(imread(output))), 0.01)
        panels = figures[0].axes[:4]
        np.testing.assert_array_equal(panels[0].images[0].get_array(), np.log10(1 + spectrum))
        np.testing.assert_array_equal(panels[2].images[0].get_array(), mask)
        np.testing.assert_array_equal(panels[3].images[0].get_array(), np.log10(1 + spectrum * mask))
        self.assertEqual(len(figures[0].axes), 8)

    def test_amplitudes_relative_phases_and_shared_colourbars(self):
        phase = np.deg2rad([0, 60, 120, 180, 240, 300])
        amplitude = np.arange(1, 7)[:, None, None] * np.ones((6, 8, 10))
        carrier = np.arange(10)[None, None, :] * 0.3
        fields = amplitude * np.exp(1j * (carrier - phase[:, None, None]))
        figures = []
        savefig = Figure.savefig

        def capture(figure, *args, **kwargs):
            figures.append(figure)
            return savefig(figure, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(Figure, "savefig", new=capture):
                outputs = create_field_maps(fields, directory, phase_sign=-1)
            for output in outputs.values():
                self.assertGreater(float(np.std(imread(output))), 0.01)

        for figure in figures:
            self.assertEqual(len(figure.axes), 7)
            self.assertTrue(figure.axes[-1].get_ylabel())
        for index in range(6):
            amplitude_image = figures[0].axes[index].images[0]
            phase_image = figures[1].axes[index].images[0]
            np.testing.assert_allclose(amplitude_image.get_array(), amplitude[index])
            np.testing.assert_allclose(amplitude_image.get_clim(), (0, 6))
            self.assertEqual(phase_image.get_clim(), (-180, 180))
            expected = -np.rad2deg(np.angle(np.exp(-1j * phase[index])))
            np.testing.assert_allclose(
                np.exp(1j * np.deg2rad(phase_image.get_array())),
                np.exp(1j * np.deg2rad(expected)), atol=1e-10,
            )


if __name__ == "__main__":
    unittest.main()