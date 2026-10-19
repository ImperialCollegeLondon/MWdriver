import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mwdriver.domain.wavefront_surface import SURFACE_RESULTS, build_wavefront_surface


class WavefrontSurfaceTests(unittest.TestCase):
    def test_deletion_prepares_2d_and_current_gl_context(self):
        from mwdriver.ui.camera_view_controller import CameraViewController

        operations = []

        class Dims:
            @property
            def ndisplay(self):
                return 3

            @ndisplay.setter
            def ndisplay(self, value):
                operations.append(("ndisplay", value))

        surface = SimpleNamespace(name="surface", _type_string="surface")
        canvas = SimpleNamespace(set_current=lambda: operations.append(("context", "current")))
        controller = SimpleNamespace(viewer=SimpleNamespace(
            layers=[surface], dims=Dims(),
            window=SimpleNamespace(_qt_viewer=SimpleNamespace(canvas=SimpleNamespace(_scene_canvas=canvas))),
        ))
        CameraViewController.on_layer_removing(controller, SimpleNamespace(index=0))
        self.assertEqual(operations, [("ndisplay", 2), ("context", "current")])

    def test_time_series_range_excludes_hidden_surface_height(self):
        from napari.components import ViewerModel
        from mwdriver.ui.camera_view_controller import CameraViewController

        viewer = ViewerModel()
        preview = viewer.add_image(np.zeros((10, 12)))
        roi = viewer.add_shapes(name="ROI")
        capture = viewer.add_image(np.zeros((6, 10, 12)))
        mesh = build_wavefront_surface(
            {"wavefront_aberration_nm": np.arange(120).reshape(10, 12) * 20.0},
            "surface_defocus_removed",
        )
        surface = viewer.add_surface(mesh["data"], scale=mesh["scale"])
        controller = SimpleNamespace(
            viewer=viewer, layer=preview, roi_layer=roi,
            phase_analysis_layer_records=[{"layer": surface}],
            phase_capture_time_series_layers=[capture],
        )
        for _ in range(3):
            CameraViewController.apply_layer_visibility_plan(controller, "analysis", surface)
            CameraViewController.apply_layer_visibility_plan(controller, "time_series", capture)
            self.assertEqual(viewer.dims.ndisplay, 2)
            self.assertEqual(tuple(viewer.dims.range[0]), (0, 5, 1))
            self.assertEqual(viewer.dims.nsteps[0], 6)
            self.assertEqual(viewer.dims.axis_labels[0], "time")
            viewer.dims.set_current_step(0, 5)
            self.assertEqual(viewer.dims.current_step[0], 5)
            self.assertFalse(surface.visible)
            self.assertTrue(capture.visible)

    def test_large_surface_uses_full_resolution(self):
        results = {"wavefront_aberration_nm": np.ones((400, 500))}
        mesh = build_wavefront_surface(results, "surface_defocus_removed")
        vertices, faces, heights = mesh["data"]
        self.assertEqual(vertices.shape, (400 * 500, 3))
        self.assertEqual(faces.shape, (2 * 399 * 499, 3))
        self.assertEqual(heights.shape, (400 * 500,))

    def test_controller_switches_surface_stages_and_returns_to_image(self):
        from napari.layers import Surface, Image
        from mwdriver.ui.camera_view_controller import CameraViewController

        results = {
            "tilt_removed": np.arange(36, dtype=float).reshape(6, 6) / 100,
            "wavefront_aberration_nm": np.arange(36, dtype=float).reshape(6, 6),
            "residual_nm": np.arange(36, dtype=float).reshape(6, 6) / 10,
        }
        layers = []
        selection = SimpleNamespace(active=None)
        selection.select_only = lambda layer: setattr(selection, "active", layer)

        class Layers(list):
            def remove(self, layer):
                raise AssertionError("Analysis switching must not delete GPU-backed layers")

        layer_list = Layers()
        layer_list.selection = selection

        def add_surface(data, **kwargs):
            layer = Surface(data, **kwargs)
            layer_list.append(layer)
            layers.append(layer)
            return layer

        def add_image(data, **kwargs):
            layer = Image(data, **kwargs)
            layer_list.append(layer)
            return layer

        selector = SimpleNamespace(currentData=lambda: selected[0], currentText=lambda: selected[0])
        from mwdriver.ui.camera_view_controller import CameraViewController

        class Controller(CameraViewController):
            pass

        viewer = Controller()
        viewer.__dict__.update(
            phase_analysis_results=results, phase_analysis_selector=selector,
            phase_analysis_source_stack_name="test", phase_analysis_layer=None,
            phase_analysis_layer_records=[], apply_layer_visibility_plan=MagicMock(),
            viewer=SimpleNamespace(layers=layer_list, add_surface=add_surface, add_image=add_image, reset_view=MagicMock(), dims=SimpleNamespace(ndisplay=3)),
        )
        selected = ["surface_tilt_removed"]
        for key in SURFACE_RESULTS:
            selected[0] = key
            CameraViewController.show_selected_phase_analysis_result(viewer)
            self.assertEqual(viewer.phase_analysis_layer._type_string, "surface")
            self.assertEqual(viewer.phase_analysis_layer_records[-1]["selected_key"], key)
        self.assertEqual(len(layers), 1)
        selected[0] = "tilt_removed"
        CameraViewController.show_selected_phase_analysis_result(viewer)
        self.assertEqual(viewer.phase_analysis_layer._type_string, "image")
        self.assertIn(layers[0], layer_list)
        image_layer = viewer.phase_analysis_layer
        for _ in range(3):
            selected[0] = "surface_residual"
            viewer.show_selected_phase_analysis_result()
            self.assertEqual(viewer.phase_analysis_layer._type_string, "surface")
            self.assertIs(viewer.phase_analysis_layer, layers[0])
            selected[0] = "tilt_removed"
            viewer.show_selected_phase_analysis_result()
            self.assertEqual(viewer.phase_analysis_layer._type_string, "image")
            self.assertIs(viewer.phase_analysis_layer, image_layer)
            self.assertFalse(viewer._analysis_layer_switching)
        self.assertEqual(len(layer_list), 2)

    def test_switch_exclusively_selects_reused_surface(self):
        from napari.components import ViewerModel
        from mwdriver.ui.camera_view_controller import CameraViewController

        class Controller(CameraViewController):
            def apply_layer_visibility_plan(self, mode, selected_layer=None):
                pass

        controller = Controller()
        controller.viewer = ViewerModel()
        controller.phase_analysis_layer = None
        controller.phase_analysis_layer_records = []
        controller.phase_analysis_source_stack_name = "test"
        controller.phase_analysis_results = {"tilt_removed": np.arange(36).reshape(6, 6)}
        selected = ["tilt_removed"]
        controller.phase_analysis_selector = SimpleNamespace(
            currentData=lambda: selected[0], currentText=lambda: selected[0]
        )
        for key in ("tilt_removed", "surface_tilt_removed", "tilt_removed", "surface_tilt_removed"):
            selected[0] = key
            controller.show_selected_phase_analysis_result()
            self.assertEqual(set(controller.viewer.layers.selection), {controller.phase_analysis_layer})
            self.assertIs(controller.viewer.layers.selection.active, controller.phase_analysis_layer)

    def test_selection_callbacks_are_ignored_during_layer_replacement(self):
        from mwdriver.ui.camera_view_controller import CameraViewController

        controller = SimpleNamespace(_analysis_layer_switching=True)
        CameraViewController.on_active_layer_changed(controller)
        CameraViewController.show_selected_phase_analysis_result(controller)

    def test_mesh_units_masking_and_shared_limits(self):
        yy, xx = np.indices((12, 14))
        valid = (xx >= 2) & (yy >= 2)
        valid[6, 7] = False
        values = np.where(valid, xx + yy, np.nan)
        results = {
            "tilt_removed": values * 2.0 * np.pi / 600.0,
            "wavefront_aberration_nm": values * 0.5,
            "residual_nm": values * 0.1,
            "wavelength_nm": 600.0,
        }
        for key, factor in zip(SURFACE_RESULTS, (1.0, 0.5, 0.2)):
            mesh = build_wavefront_surface(results, key)
            vertices, faces, heights = mesh["data"]
            np.testing.assert_allclose(heights, (vertices[:, 1] + vertices[:, 2]) * factor)
            self.assertTrue(np.all(np.isfinite(vertices)))
            np.testing.assert_allclose(mesh["contrast_limits"], (-24, 24))
            for triangle in vertices[faces]:
                rows = triangle[:, 1].astype(int)
                columns = triangle[:, 2].astype(int)
                self.assertTrue(np.all(valid[rows.min():rows.max()+1, columns.min():columns.max()+1]))

    def test_napari_accepts_surface_data(self):
        from napari.layers import Surface

        results = {"wavefront_aberration_nm": np.arange(36, dtype=float).reshape(6, 6)}
        mesh = build_wavefront_surface(results, "surface_defocus_removed")
        layer = Surface(mesh["data"], colormap="viridis", contrast_limits=mesh["contrast_limits"], scale=mesh["scale"], shading="smooth")
        self.assertEqual(layer.ndim, 3)
        self.assertEqual(layer.colormap.name, "viridis")