import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mwdriver.ui.camera_dock_controller import CameraDockController
from qtpy.QtCore import QEvent


class FakeLayout:
    def __init__(self):
        self.widgets = []

    def indexOf(self, widget):
        return self.widgets.index(widget) if widget in self.widgets else -1

    def addWidget(self, widget, *_args):
        self.widgets.append(widget)


class FakeTimer:
    def __init__(self):
        self.delays = []

    def start(self, delay):
        self.delays.append(delay)


class CameraDockControllerTests(unittest.TestCase):
    def test_interpolation_control_stays_hidden_for_all_layers(self):
        analysis_image = SimpleNamespace(_type_string="image")
        analysis_plot = SimpleNamespace(_type_string="image", rgb=True)
        surface = SimpleNamespace(_type_string="surface")
        preview = SimpleNamespace(_type_string="image")
        interpolation_label = SimpleNamespace(text=lambda: "interpolation:")
        controls = SimpleNamespace(findChildren=lambda _kind: [interpolation_label])
        controller = CameraDockController()
        selection = SimpleNamespace(active=None)
        controller.viewer = SimpleNamespace(
            layers=SimpleNamespace(selection=selection),
            window=SimpleNamespace(_qt_viewer=SimpleNamespace(controls=controls)),
        )
        controller.phase_analysis_layer_records = [
            {"layer": layer} for layer in (analysis_image, analysis_plot, surface)
        ]
        controller.set_control_row_visible = MagicMock()
        controller.place_camera_adjust_controls_in_layer_controls = MagicMock()
        controller.place_camera_status_controls_in_layer_controls = MagicMock()
        controller.place_phase_analysis_controls_in_layer_controls = MagicMock()

        for layer, visible in (
            (analysis_image, False), (analysis_plot, False),
            (surface, False), (preview, False), (analysis_image, False),
        ):
            with self.subTest(layer=layer, visible=visible):
                selection.active = layer
                controller.simplify_layer_controls()
                controller.set_control_row_visible.assert_called_with(
                    interpolation_label, visible
                )

    def test_widget_insertion_is_idempotent_and_refreshes_content(self):
        controller = CameraDockController()
        widget = SimpleNamespace(visible=False)
        widget.setVisible = lambda visible: setattr(widget, "visible", visible)
        content = SimpleNamespace(adjustments=0)
        content.adjustSize = lambda: setattr(content, "adjustments", content.adjustments + 1)
        layout = FakeLayout()
        controller.mwdriver_controls_layout = layout
        controller.mwdriver_controls_content = content

        controller.add_widget_to_mwdriver_dock(widget)
        controller.add_widget_to_mwdriver_dock(widget)

        self.assertEqual(layout.widgets, [widget])
        self.assertTrue(widget.visible)
        self.assertEqual(content.adjustments, 2)

    def test_resize_filter_debounces_registered_resize_events_only(self):
        controller = CameraDockController()
        watched_widget = object()
        controller.controls_resize_filter_widgets = [watched_widget]
        controller.controls_resize_timer = FakeTimer()

        resized = SimpleNamespace(type=lambda: QEvent.Resize)
        unrelated = SimpleNamespace(type=lambda: QEvent.Show)

        self.assertFalse(controller.eventFilter(watched_widget, resized))
        self.assertFalse(controller.eventFilter(watched_widget, unrelated))
        self.assertFalse(controller.eventFilter(object(), resized))

        self.assertEqual(controller.controls_resize_timer.delays, [80])


if __name__ == "__main__":
    unittest.main()