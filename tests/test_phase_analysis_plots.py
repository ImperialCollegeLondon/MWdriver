import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mwdriver.analysis.phase_aberration import ZERNIKE_MODES
from mwdriver.analysis.phase_analysis_plots import render_aberration_plots, render_wavefront_surfaces
from mwdriver.analysis.phase_analysis_new import (
    save_intermediate_png,
    save_result_images,
)
from mwdriver.analysis import phase_analysis_new


class PhaseAnalysisPlotTests(unittest.TestCase):
    def test_wavefront_surfaces_share_scale_and_exclude_masked_faces(self):
        from mpl_toolkits.mplot3d import Axes3D
        from mwdriver.analysis.phase_analysis_plots import figure_to_rgb

        yy, xx = np.indices((24, 30))
        masked = (xx < 3) | (yy < 2) | ((xx == 15) & (yy == 12))
        surface = np.ma.array(xx + yy, mask=masked, dtype=float)
        surface.data[masked] = 1e12
        surfaces = (surface, surface * 500.0, surface * -100.0)
        limits = np.percentile(surface.compressed(), [0.1, 99.1])
        original_plot = Axes3D.plot_trisurf

        def check_mesh(axis, mesh, heights, **kwargs):
            self.assertEqual(kwargs["alpha"], 1.0)
            self.assertFalse(kwargs["antialiased"])
            self.assertEqual(kwargs["edgecolor"], "none")
            self.assertEqual(kwargs["cmap"], "viridis")
            self.assertGreaterEqual(mesh.x.min(), 3)
            self.assertGreaterEqual(mesh.y.min(), 2)
            active = mesh.get_masked_triangles()
            for triangle in active:
                pixels_x = mesh.x[triangle].astype(int)
                pixels_y = mesh.y[triangle].astype(int)
                self.assertFalse(np.any(masked[pixels_y, pixels_x]))
            return original_plot(axis, mesh, heights, **kwargs)

        def check_figure(figure):
            self.assertEqual(len(figure.axes), 4)
            for axis in figure.axes[:3]:
                np.testing.assert_allclose(axis.get_xlim(), (3, 29))
                np.testing.assert_allclose(axis.get_ylim(), (23, 2))
                np.testing.assert_allclose(axis.get_zlim(), limits)
                self.assertEqual(axis.get_zlabel(), "Wavefront OPD (nm)")
                self.assertEqual(len(axis.collections), 1)
                self.assertEqual(axis.collections[0].norm.vmin, limits[0])
                self.assertEqual(axis.collections[0].norm.vmax, limits[1])
                self.assertEqual(axis.collections[0].cmap.name, "viridis")
                self.assertEqual(axis.collections[0].get_alpha(), 1.0)
            self.assertEqual(figure.axes[3]._colorbar.mappable.cmap.name, "viridis")
            return figure_to_rgb(figure)

        with patch.object(Axes3D, "plot_trisurf", new=check_mesh):
            with patch("mwdriver.analysis.phase_analysis_plots.figure_to_rgb", side_effect=check_figure):
                image = render_wavefront_surfaces(*surfaces)
        self.assertEqual(image.shape, (1080, 1440, 3))

    def test_analysis_exports_roi_without_duplicate_input_frames(self):
        yy, xx = np.indices((32, 40))
        roi_mask = (yy >= 3) & (yy < 29) & (xx >= 4) & (xx < 36)
        phase = 0.02 * xx + 0.03 * yy
        steps = phase_analysis_new.PSI_ALGORITHMS[
            phase_analysis_new.DEFAULT_PSI_ALGORITHM
        ]["expected_phase_deg"]
        frames = np.asarray([
            100.0 + (20.0 + xx) * np.cos(phase + np.deg2rad(step))
            for step in steps
        ])

        with tempfile.TemporaryDirectory() as directory:
            session_dir = Path(directory)
            raw_path = session_dir / "capture_raw_stack.npy"
            roi_path = session_dir / "roi_mask.npy"
            export_path = session_dir / "results.npz"
            np.save(raw_path, frames)
            np.save(roi_path, roi_mask)
            with patch.object(phase_analysis_new, "save_intermediate_png") as save_png:
                result = phase_analysis_new.analyze_capture(
                    session_dir,
                    raw_path,
                    export_path=export_path,
                    roi_mask_path=roi_path,
                )

            with np.load(export_path, allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["roi_mask"], roi_mask)
                self.assertEqual(archive["roi_mask"].dtype, np.dtype(bool))
                self.assertIn("wrapped", archive.files)
                self.assertEqual(archive["wavefront_surfaces_plot_rgb"].shape, (1080, 1440, 3))
                self.assertTrue((session_dir / "phase_analysis_result_images" / "wavefront_surfaces.png").is_file())
                for key, coefficients in (
                    ("zernike_coefficients_before_correction_waves", result.zernike_coefficients_before_waves),
                    ("zernike_coefficients_after_correction_waves", result.zernike_coefficients_after_waves),
                ):
                    coefficient_dict = json.loads(archive[key].item())
                    self.assertEqual(list(coefficient_dict), [
                        f"J{mode_index:02d} {mode_name}"
                        for mode_index, _, _, mode_name in ZERNIKE_MODES
                    ])
                    np.testing.assert_allclose(list(coefficient_dict.values()), coefficients)
                components = json.loads(archive["aberration_component_waves"].item())
                self.assertEqual(list(components), [
                    "Astigmatism", "Coma", "Trefoil", "Primary Spherical"
                ])
                corrected = result.zernike_coefficients_after_waves
                np.testing.assert_allclose(list(components.values()), [
                    np.hypot(corrected[4], corrected[5]),
                    np.hypot(corrected[6], corrected[7]),
                    np.hypot(corrected[8], corrected[9]),
                    abs(corrected[10]),
                ])
                for key in archive.files:
                    self.assertFalse(archive[key].dtype.hasobject)
                for index in range(6):
                    self.assertNotIn(f"phase_{index}", archive.files)
            saved_names = [call.args[0].name for call in save_png.call_args_list]
            self.assertTrue(saved_names)
            self.assertFalse(any(name.startswith("phase_intermediate_01") for name in saved_names))
            self.assertIn("phase_intermediate_02_numerator.png", saved_names)
            np.testing.assert_array_equal(np.load(raw_path), frames)

    def test_aberration_values_are_large_horizontal_and_antialiased(self):
        import matplotlib.pyplot as plt
        from mwdriver.analysis.phase_analysis_plots import figure_to_rgb

        rendered_count = 0

        def check_and_render(figure):
            nonlocal rendered_count
            axis = figure.axes[0]
            self.assertGreater(len(axis.texts), 0)
            for text in axis.texts:
                self.assertEqual(text.get_rotation(), 0)
                self.assertGreaterEqual(text.get_fontsize(), 14)
                self.assertTrue(text.get_antialiased())
                self.assertEqual(text.get_fontfamily(), ["DejaVu Sans"])
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            for text in axis.texts:
                bounds = text.get_window_extent(renderer)
                self.assertGreaterEqual(bounds.x0, 0)
                self.assertGreaterEqual(bounds.y0, 0)
                self.assertLessEqual(bounds.x1, 1440)
                self.assertLessEqual(bounds.y1, 1080)
            rendered_count += 1
            return figure_to_rgb(figure)

        with plt.rc_context({"text.antialiased": False}):
            with patch(
                "mwdriver.analysis.phase_analysis_plots.figure_to_rgb",
                side_effect=check_and_render,
            ):
                render_aberration_plots(
                    ZERNIKE_MODES,
                    np.linspace(-0.2, 0.2, len(ZERNIKE_MODES)),
                    np.linspace(-0.08, 0.08, len(ZERNIKE_MODES)),
                    np.asarray([0.03, 0.02, 0.01, 0.04]),
                )
        self.assertEqual(rendered_count, 3)

    def test_render_aberration_plots_returns_labelled_rgb_images(self):
        coefficients_before = np.linspace(-0.2, 0.2, len(ZERNIKE_MODES))
        coefficients_after = coefficients_before * 0.4
        component_waves = np.asarray(
            [0.03, 0.02, 0.01, 0.04]
        )

        with matplotlib.rc_context({"figure.dpi": 72}):
            result = render_aberration_plots(
                ZERNIKE_MODES,
                coefficients_before,
                coefficients_after,
                component_waves,
            )

        self.assertEqual(len(result.zernike_bar_labels), len(ZERNIKE_MODES))
        self.assertEqual(result.zernike_bar_labels[0], "J01\nPiston")
        self.assertEqual(result.zernike_bar_labels[-1], "J11\nPrimary Spherical")
        for image in (
            result.zernike_before_rgb,
            result.zernike_after_rgb,
            result.composition_rgb,
        ):
            self.assertEqual(image.shape, (1080, 1440, 3))
            self.assertEqual(image.dtype, np.uint8)
            self.assertGreater(np.unique(image.reshape(-1, 3), axis=0).shape[0], 1)

        with tempfile.TemporaryDirectory() as directory:
            result_dir = Path(directory)
            save_result_images(result_dir, {
                "zernike_before": result.zernike_before_rgb,
                "zernike_after": result.zernike_after_rgb,
                "aberration_composition": result.composition_rgb,
            })
            for path in result_dir.glob("*.png"):
                self.assertEqual(matplotlib.image.imread(path).shape[:2], (1080, 1440))

    def test_intermediate_png_matches_camera_canvas_despite_matplotlib_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "intermediate.png"
            with matplotlib.rc_context({"figure.dpi": 72, "savefig.bbox": "tight"}):
                save_intermediate_png(
                    path,
                    np.arange(120).reshape(10, 12),
                    "Intermediate phase",
                    "viridis",
                )
            self.assertEqual(matplotlib.image.imread(path).shape[:2], (1080, 1440))


if __name__ == "__main__":
    unittest.main()