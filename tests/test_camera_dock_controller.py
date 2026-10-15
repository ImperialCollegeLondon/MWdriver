import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

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