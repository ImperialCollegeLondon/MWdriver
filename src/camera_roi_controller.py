"""ROI drawing, mask and persistence orchestration for the Napari viewer."""

import os

import numpy as np
from qtpy.QtCore import QTimer
from qtpy.QtGui import QAction
from qtpy.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from mwdriver.domain.roi_geometry import (
    apply_mask_to_frame,
    build_inside_mask,
    build_roi_selection,
    downsample_mask,
)
from roi_controls import create_roi_menu
from roi_storage import load_roi_files, save_roi_files


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class CameraRoiController:
    def init_roi_controls(self):
        """
        Put ROI functions into the top menu bar instead of the bottom dock widget.

        The actual ROI functions are unchanged:
        - Rect ROI
        - Circle ROI
        - Mask Outside Area
        - Mask Inside Area
        - Save ROI
        - Clear ROI
        """

        try:
            qt_window = self.viewer.window._qt_window
            menubar = qt_window.menuBar()
            self.roi_menu, roi_actions = create_roi_menu(
                menubar,
                qt_window,
                {
                    "start_rectangle_roi": self.start_rectangle_roi,
                    "start_circle_roi": self.start_circle_roi,
                    "mask_outside_area": self.mask_outside_area,
                    "mask_inside_area": self.mask_inside_area,
                    "save_roi": self.save_roi,
                    "reload_roi": self.reload_roi,
                    "clear_roi": self.clear_roi,
                },
                QAction,
            )
            for attribute, action in roi_actions.items():
                setattr(self, attribute, action)

            print("ROI tools added to top menu bar.")

        except Exception as e:
            print("ROI menu creation failed, using bottom ROI dock instead:", e)

            roi_widget = QWidget()
            roi_layout = QVBoxLayout()
            roi_button_layout = QHBoxLayout()

            self.rect_roi_button = QPushButton("Rect ROI")
            self.circle_roi_button = QPushButton("Circle ROI")
            self.mask_outside_button = QPushButton("Mask Outside Area")
            self.mask_inside_button = QPushButton("Mask Inside Area")
            self.save_roi_button = QPushButton("Save ROI")
            self.reload_roi_button = QPushButton("Reload ROI")
            self.clear_roi_button = QPushButton("Clear ROI")

            roi_button_layout.addWidget(self.rect_roi_button)
            roi_button_layout.addWidget(self.circle_roi_button)
            roi_button_layout.addWidget(self.mask_outside_button)
            roi_button_layout.addWidget(self.mask_inside_button)
            roi_button_layout.addWidget(self.save_roi_button)
            roi_button_layout.addWidget(self.reload_roi_button)
            roi_button_layout.addWidget(self.clear_roi_button)

            info_label = QLabel(
                "ROI tools: draw one ROI, move/resize it, choose mask mode, then click Save ROI."
            )

            roi_layout.addLayout(roi_button_layout)
            roi_layout.addWidget(info_label)
            roi_widget.setLayout(roi_layout)

            self.viewer.window.add_dock_widget(
                roi_widget,
                name="ROI Control",
                area="bottom"
            )

            self.rect_roi_button.clicked.connect(self.start_rectangle_roi)
            self.circle_roi_button.clicked.connect(self.start_circle_roi)
            self.mask_outside_button.clicked.connect(self.mask_outside_area)
            self.mask_inside_button.clicked.connect(self.mask_inside_area)
            self.save_roi_button.clicked.connect(self.save_roi)
            self.reload_roi_button.clicked.connect(self.reload_roi)
            self.clear_roi_button.clicked.connect(self.clear_roi)

            print("ROI control panel ready.")

    def start_rectangle_roi(self):
        self.prepare_roi_drawing()
        self.roi_layer.data = []
        self.roi_params = None
        self.roi_mask = None

        self.display_mask_enabled = False
        self.current_display_mask = None

        self.last_roi_count = 0
        self.waiting_for_roi_finish = True

        self.roi_layer.mode = "add_rectangle"
        self.viewer.layers.selection.active = self.roi_layer

        self.roi_label.setText("ROI: Draw one rectangle")
        self.status_label.setText("Status: Rectangle ROI mode")

    def start_circle_roi(self):
        self.prepare_roi_drawing()
        self.roi_layer.data = []
        self.roi_params = None
        self.roi_mask = None

        self.display_mask_enabled = False
        self.current_display_mask = None

        self.last_roi_count = 0
        self.waiting_for_roi_finish = True

        self.roi_layer.mode = "add_ellipse"
        self.viewer.layers.selection.active = self.roi_layer

        self.roi_label.setText("ROI: Draw one circle/ellipse")
        self.status_label.setText("Status: Circle ROI mode")

    def prepare_roi_drawing(self):
        self.apply_layer_visibility_plan("roi")
        self.phase_analysis_view_active = False
        if self.view_phase_analysis_button is not None:
            self.view_phase_analysis_button.setText("View Analysis Results")
        self.viewer.reset_view()

    def on_roi_data_changed(self, event=None):
        if self.roi_layer is None:
            return

        current_count = len(self.roi_layer.data)

        if not self.waiting_for_roi_finish:
            self.last_roi_count = current_count
            return

        if current_count > self.last_roi_count:
            self.last_roi_count = current_count
            QTimer.singleShot(100, self.finish_roi_drawing)

    def finish_roi_drawing(self):
        if self.roi_layer is None or len(self.roi_layer.data) == 0:
            return

        last_index = len(self.roi_layer.data) - 1

        self.roi_layer.selected_data = {last_index}
        self.roi_layer.mode = "select"

        self.waiting_for_roi_finish = False

        self.roi_label.setText("ROI: Selected - move or resize it, then choose mask mode")
        self.status_label.setText("Status: ROI selected / editable")

        print("ROI drawing finished. Switched to select mode.")

    def get_current_inside_mask(self):
        if self.roi_layer is None or len(self.roi_layer.data) == 0:
            return None, None, None
        return build_inside_mask(
            roi_vertices=self.roi_layer.data[-1],
            roi_type=str(self.roi_layer.shape_type[-1]),
            image_height=self.height,
            image_width=self.width,
            display_downsample=self.display_downsample,
        )

    def make_display_mask_from_raw_mask(self, raw_mask):
        return downsample_mask(raw_mask, self.display_downsample)

    def mask_outside_area(self):
        inside_mask, roi_type, roi_info = self.get_current_inside_mask()

        if inside_mask is None:
            self.status_label.setText("Status: No valid ROI for mask outside")
            self.roi_label.setText("ROI: Draw ROI first")
            print("No valid ROI for mask outside.")
            return

        self.mask_mode = "outside"
        (
            valid_mask,
            self.current_display_mask,
            self.roi_params,
            mask_mode_text,
        ) = build_roi_selection(
            inside_mask,
            roi_info,
            self.mask_mode,
            self.display_downsample,
        )
        self.display_mask_enabled = True
        self.roi_mask = valid_mask

        self.roi_label.setText("ROI: Mask outside area - keep inside ROI")
        self.status_label.setText("Status: Mask outside area applied")

        print("Mask outside area applied.")

    def mask_inside_area(self):
        inside_mask, roi_type, roi_info = self.get_current_inside_mask()

        if inside_mask is None:
            self.status_label.setText("Status: No valid ROI for mask inside")
            self.roi_label.setText("ROI: Draw ROI first")
            print("No valid ROI for mask inside.")
            return

        self.mask_mode = "inside"
        (
            valid_mask,
            self.current_display_mask,
            self.roi_params,
            mask_mode_text,
        ) = build_roi_selection(
            inside_mask,
            roi_info,
            self.mask_mode,
            self.display_downsample,
        )
        self.display_mask_enabled = True
        self.roi_mask = valid_mask

        self.roi_label.setText("ROI: Mask inside area - keep outside ROI")
        self.status_label.setText("Status: Mask inside area applied")

        print("Mask inside area applied.")

    def apply_display_mask(self, display_frame):
        return apply_mask_to_frame(
            display_frame,
            self.current_display_mask,
            self.display_mask_enabled,
        )

    def clear_roi(self):
        self.roi_layer.data = []
        self.roi_params = None
        self.roi_mask = None

        self.last_roi_count = 0
        self.waiting_for_roi_finish = False
        self.roi_layer.mode = "select"

        self.mask_mode = "outside"
        self.display_mask_enabled = False
        self.current_display_mask = None

        self.roi_label.setText("ROI: Not selected")
        self.status_label.setText("Status: ROI and mask cleared")

        print("ROI and mask cleared.")

    def reload_roi(self):
        try:
            roi_params, roi_mask, shape_data, shape_type = load_roi_files(
                SCRIPT_DIR,
                image_height=self.height,
                image_width=self.width,
                display_downsample=self.display_downsample,
            )

            self.waiting_for_roi_finish = False
            self.roi_layer.data = []
            self.roi_layer.add(shape_data, shape_type=shape_type)
            self.roi_layer.selected_data = {0}
            self.roi_layer.mode = "select"
            self.roi_layer.visible = True
            self.viewer.layers.selection.active = self.roi_layer

            self.roi_params = roi_params
            self.roi_mask = roi_mask
            self.current_display_mask = self.make_display_mask_from_raw_mask(
                roi_mask
            )
            self.display_mask_enabled = True
            self.last_roi_count = 1

            saved_mask_mode = roi_params.get("mask_mode", "mask_outside_area")
            self.mask_mode = (
                "inside" if saved_mask_mode == "mask_inside_area" else "outside"
            )

            self.roi_label.setText(f"ROI: Reloaded - {saved_mask_mode}")
            self.status_label.setText("Status: Saved ROI reloaded")
            print(
                "Saved ROI reloaded:",
                os.path.join(SCRIPT_DIR, "roi_params.json"),
                os.path.join(SCRIPT_DIR, "roi_mask.npy"),
            )

        except Exception as error:
            self.status_label.setText(f"Status: Reload ROI failed: {error}")
            self.roi_label.setText("ROI: Reload failed")
            print("Reload ROI failed:", error)

    def save_roi(self):
        inside_mask, roi_type, roi_info = self.get_current_inside_mask()

        if inside_mask is None:
            self.status_label.setText("Status: No ROI to save")
            self.roi_label.setText("ROI: Not selected")
            print("No ROI to save.")
            return

        (
            valid_mask,
            self.current_display_mask,
            self.roi_params,
            mask_mode_text,
        ) = build_roi_selection(
            inside_mask,
            roi_info,
            self.mask_mode,
            self.display_downsample,
            describe_rejected_area=True,
        )

        self.roi_mask = valid_mask
        self.display_mask_enabled = True

        roi_json_path, roi_mask_path = save_roi_files(
            os.getcwd(),
            self.roi_params,
            self.roi_mask,
        )

        self.roi_label.setText(f"ROI: Saved - {mask_mode_text}")
        self.status_label.setText("Status: ROI and mask saved")

        print("ROI and mask saved.")
        print("ROI parameters:", self.roi_params)
        print("Saved:", roi_json_path)
        print("Saved:", roi_mask_path)
