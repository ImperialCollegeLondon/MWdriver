"""Napari dock layout, chrome and resize handling for the camera viewer."""

from qtpy.QtCore import QEvent, QTimer, Qt
from qtpy.QtWidgets import (
    QAbstractButton,
    QFormLayout,
    QGridLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class CameraDockController:
    def initialize_camera_dock_state(self):
        self.mwdriver_controls_scroll = None
        self.mwdriver_controls_content = None
        self.mwdriver_controls_layout = None
        self.controls_container = None
        self.controls_resize_filter_widgets = []
        self.controls_resize_timer = None

    def simplify_napari_interface(self):
        """
        Hide unnecessary napari built-in buttons around the layer list.

        This keeps the real layers and functions unchanged:
        - ROI layer is still used for drawing / moving / resizing ROI.
        - Camera Preview layer is still used for live camera display.
        - Other program functions are not removed.

        It only hides napari's own UI buttons such as add layer, delete layer,
        console, grid, home, 2D/3D and other default viewer buttons.
        """

        try:
            qt_viewer = self.viewer.window._qt_viewer
        except Exception as e:
            print("Cannot access napari qt_viewer:", e)
            return

        # Hide the top layer-button bar: points / shapes / labels / delete etc.
        possible_layer_button_attrs = [
            "layerButtons",
            "layer_buttons",
            "buttons",
        ]

        for attr in possible_layer_button_attrs:
            try:
                obj = getattr(qt_viewer, attr, None)
                if obj is not None:
                    obj.setVisible(False)
            except Exception:
                pass

        # Hide the bottom viewer-button bar: console / grid / home / 2D-3D etc.
        possible_viewer_button_attrs = [
            "viewerButtons",
            "viewer_buttons",
            "viewerButtonsContainer",
        ]

        for attr in possible_viewer_button_attrs:
            try:
                obj = getattr(qt_viewer, attr, None)
                if obj is not None:
                    obj.setVisible(False)
            except Exception:
                pass

        # Extra compatibility for different napari versions.
        # Hide buttons by tooltip / objectName / text, but keep layer eye buttons.
        keywords_to_hide = [
            "new points layer",
            "new shapes layer",
            "new labels layer",
            "new image layer",
            "add points",
            "add shapes",
            "add labels",
            "delete selected layers",
            "delete layer",
            "console",
            "roll dimensions",
            "transpose dimensions",
            "grid mode",
            "home",
            "reset view",
            "toggle ndisplay",
            "2d/3d",
            "activity",
        ]

        try:
            buttons = qt_viewer.findChildren(QAbstractButton)

            for button in buttons:
                tooltip = button.toolTip().strip().lower()
                object_name = button.objectName().strip().lower()
                text = button.text().strip().lower()
                combined = f"{tooltip} {object_name} {text}"

                # Do not hide layer visibility buttons, otherwise the ROI / preview
                # rows may look broken in some napari versions.
                if "visibility" in combined or "visible" in combined:
                    continue

                if any(keyword in combined for keyword in keywords_to_hide):
                    button.setVisible(False)

        except Exception as e:
            print("Hide napari buttons warning:", e)

        print("Napari interface simplified: ROI and Camera Preview remain visible.")

    def simplify_layer_controls(self):
        allowed_labels = {
            "contrast limits",
            "gamma",
            "colormap",
        }

        image_control_labels = {
            "opacity",
            "blending",
            "contrast limits",
            "auto-contrast",
            "gamma",
            "colormap",
            "projection mode",
            "interpolation",
            "rendering",
            "depiction",
            "attenuation",
            "iso threshold",
            "plane",
            "shading",
        }

        try:
            qt_viewer = self.viewer.window._qt_viewer
            controls_container = qt_viewer.controls
        except Exception as e:
            print("Cannot access napari layer controls:", e)
            return

        labels = controls_container.findChildren(QLabel)

        for label in labels:
            text = label.text().strip().lower().replace(":", "")

            if text not in image_control_labels:
                continue

            should_show = text in allowed_labels
            self.set_control_row_visible(label, should_show)

        self.place_profile_controls_in_layer_controls()
        self.place_camera_adjust_controls_in_layer_controls()
        self.place_camera_status_controls_in_layer_controls()
        self.place_phase_analysis_controls_in_layer_controls()

    def set_control_row_visible(self, label, visible):
        parent = label.parentWidget()
        checked_parents = []

        for _ in range(5):
            if parent is None:
                break
            checked_parents.append(parent)
            parent = parent.parentWidget()

        for parent in checked_parents:
            layout = parent.layout()
            if layout is None:
                continue

            if isinstance(layout, QFormLayout):
                try:
                    row, role = layout.getWidgetPosition(label)
                except Exception:
                    row, role = -1, None

                if row >= 0:
                    for form_role in (
                        QFormLayout.LabelRole,
                        QFormLayout.FieldRole,
                        QFormLayout.SpanningRole,
                    ):
                        item = layout.itemAt(row, form_role)
                        if item is not None and item.widget() is not None:
                            item.widget().setVisible(visible)
                    return

            if isinstance(layout, QGridLayout):
                for i in range(layout.count()):
                    item = layout.itemAt(i)
                    if item is not None and item.widget() is label:
                        row, col, row_span, col_span = layout.getItemPosition(i)

                        for j in range(layout.count()):
                            item2 = layout.itemAt(j)
                            if item2 is None or item2.widget() is None:
                                continue

                            row2, col2, row_span2, col_span2 = layout.getItemPosition(j)
                            if row2 == row:
                                item2.widget().setVisible(visible)
                        return

            try:
                for i in range(layout.count()):
                    item = layout.itemAt(i)
                    if item is not None and item.widget() is label:
                        if parent is not None and parent is not self.viewer.window._qt_viewer.controls:
                            parent.setVisible(visible)
                        else:
                            label.setVisible(visible)
                        return
            except Exception:
                pass

        label.setVisible(visible)

    def place_phase_analysis_controls_in_layer_controls(self):
        self.add_widget_to_mwdriver_dock(self.phase_analysis_controls_widget)

    def init_mwdriver_controls_dock(self):
        self.mwdriver_controls_scroll = QScrollArea()
        self.mwdriver_controls_scroll.setObjectName("mwdriver_controls_scroll")
        self.mwdriver_controls_scroll.setWidgetResizable(True)
        self.mwdriver_controls_scroll.setMinimumWidth(300)

        self.mwdriver_controls_content = QWidget()
        self.mwdriver_controls_content.setObjectName("mwdriver_controls_content")
        self.mwdriver_controls_layout = QVBoxLayout()
        self.mwdriver_controls_layout.setContentsMargins(10, 10, 10, 10)
        self.mwdriver_controls_layout.setSpacing(12)
        self.mwdriver_controls_content.setLayout(self.mwdriver_controls_layout)
        self.mwdriver_controls_scroll.setWidget(self.mwdriver_controls_content)

        self.viewer.window.add_dock_widget(
            self.mwdriver_controls_scroll,
            name="MWdriver Controls",
            area="right",
        )

        self.add_widget_to_mwdriver_dock(self.camera_adjust_widget)
        self.add_widget_to_mwdriver_dock(self.profile_controls_widget)
        self.add_widget_to_mwdriver_dock(self.camera_status_widget)

    def add_widget_to_mwdriver_dock(self, widget):
        if widget is None or self.mwdriver_controls_layout is None:
            return
        if self.mwdriver_controls_layout.indexOf(widget) < 0:
            self.mwdriver_controls_layout.addWidget(
                widget,
                0,
                Qt.AlignLeft | Qt.AlignTop,
            )
        widget.setVisible(True)
        self.mwdriver_controls_content.adjustSize()

    def install_controls_resize_filter(self):
        """
        Install resize filters on napari layer-controls widget and its parent widgets.

        Dragging the splitter usually resizes an outer dock widget, not only
        qt_viewer.controls. Therefore we listen to qt_viewer.controls, its
        parent widgets, and the main window.
        """

        try:
            qt_viewer = self.viewer.window._qt_viewer
            self.controls_container = qt_viewer.controls
            self.controls_resize_filter_widgets = []

            widget = self.controls_container

            # Listen to qt_viewer.controls and several parent widgets.
            for _ in range(8):
                if widget is None:
                    break

                try:
                    widget.installEventFilter(self)
                    self.controls_resize_filter_widgets.append(widget)
                except Exception:
                    pass

                widget = widget.parentWidget()

            # Also listen to the main napari window.
            try:
                self.viewer.window._qt_window.installEventFilter(self)
                self.controls_resize_filter_widgets.append(self.viewer.window._qt_window)
            except Exception:
                pass

            print(
                "Controls resize filter installed on",
                len(self.controls_resize_filter_widgets),
                "widgets."
            )

            self.controls_resize_timer = QTimer(self)
            self.controls_resize_timer.setSingleShot(True)
            self.controls_resize_timer.timeout.connect(
                self.refresh_left_embedded_controls
            )
            self.controls_resize_timer.start(100)

        except Exception as e:
            print("Install controls resize filter warning:", e)

    def eventFilter(self, obj, event):
        """
        Update embedded widget sizes when the left dock area or main window changes size.
        """

        try:
            if (
                obj in self.controls_resize_filter_widgets
                and event.type() == QEvent.Resize
            ):
                if self.controls_resize_timer is not None:
                    self.controls_resize_timer.start(80)

        except Exception as e:
            print("Controls resize event warning:", e)

        return False

    def get_left_controls_available_width(self):
        """
        Get the real available width of the left controls area.

        qt_viewer.controls may not change width when the outer dock is resized,
        so this checks parent widgets and uses the largest reasonable width.
        """

        widths = []

        try:
            qt_viewer = self.viewer.window._qt_viewer
            widget = qt_viewer.controls

            for _ in range(8):
                if widget is None:
                    break

                try:
                    width = widget.width()
                    if width > 0:
                        widths.append(width)
                except Exception:
                    pass

                widget = widget.parentWidget()

        except Exception:
            pass

        if len(widths) == 0:
            return 320

        # Avoid using the whole main-window width.
        reasonable_widths = [w for w in widths if 150 <= w <= 800]

        if len(reasonable_widths) == 0:
            return max(widths)

        return max(reasonable_widths)

    def refresh_left_embedded_controls(self):
        """
        Recalculate all embedded controls after dragging the left/right splitter.
        """

        try:
            self.place_profile_controls_in_layer_controls()
            self.place_camera_adjust_controls_in_layer_controls()
            self.place_camera_status_controls_in_layer_controls()
            self.place_phase_analysis_controls_in_layer_controls()
        except Exception as e:
            print("Refresh left embedded controls warning:", e)
