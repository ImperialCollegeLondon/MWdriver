"""ROI drawing, mask and persistence orchestration for the Napari viewer."""

import os
from pathlib import Path

import numpy as np
from qtpy.QtCore import QTimer
from qtpy.QtGui import QAction
from qtpy.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from mwdriver.domain.roi_geometry import (
    apply_mask_to_frame,
    build_inside_mask,
    downsample_mask,
)
from mwdriver.ui.roi_controls import create_roi_menu
from mwdriver.ui.roi_storage import load_roi_files, save_roi_files


SCRIPT_DIR = str(Path(__file__).resolve().parents[2])


class CameraRoiController:
    def init_roi_controls(self):
        """
        Put ROI functions into the top menu bar instead of the bottom dock widget.

        The layer contains one ellipse; pixels outside it are always masked.
        """

        try:
            qt_window = self.viewer.window._qt_window
            menubar = qt_window.menuBar()
            self.roi_menu, roi_actions = create_roi_menu(
                menubar,
                qt_window,
                {
                    "start_ellipse_roi": self.start_ellipse_roi,
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

            self.ellipse_roi_button = QPushButton("Ellipse ROI")
            self.save_roi_button = QPushButton("Save ROI")
            self.reload_roi_button = QPushButton("Reload ROI")
            self.clear_roi_button = QPushButton("Clear ROI")

            roi_button_layout.addWidget(self.ellipse_roi_button)
            roi_button_layout.addWidget(self.save_roi_button)
            roi_button_layout.addWidget(self.reload_roi_button)
            roi_button_layout.addWidget(self.clear_roi_button)

            info_label = QLabel(
                "Draw one ellipse; pixels outside it are masked automatically."
            )

            roi_layout.addLayout(roi_button_layout)
            roi_layout.addWidget(info_label)
            roi_widget.setLayout(roi_layout)

            self.viewer.window.add_dock_widget(
                roi_widget,
                name="ROI Control",
                area="bottom"
            )

            self.ellipse_roi_button.clicked.connect(self.start_ellipse_roi)
            self.save_roi_button.clicked.connect(self.save_roi)
            self.reload_roi_button.clicked.connect(self.reload_roi)
            self.clear_roi_button.clicked.connect(self.clear_roi)

            print("ROI control panel ready.")

    def start_ellipse_roi(self):
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

        self.roi_label.setText("ROI: Draw one ellipse")
        self.status_label.setText("Status: Ellipse ROI mode")

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

        if self.waiting_for_roi_finish and current_count > self.last_roi_count:
            self.last_roi_count = current_count
            QTimer.singleShot(100, self.finish_roi_drawing)
            return

        self.last_roi_count = current_count
        if not self.waiting_for_roi_finish:
            if current_count > 1:
                QTimer.singleShot(0, self.finish_roi_drawing)
                return
            if (
                current_count == 1
                and str(self.roi_layer.shape_type[0]) != "ellipse"
            ):
                self.roi_layer.data = []
                self.roi_label.setText("ROI: Only an ellipse is supported")
                self.status_label.setText("Status: Non-ellipse ROI cleared")
                return
            self.refresh_roi_mask()

    def finish_roi_drawing(self):
        if self.roi_layer is None or len(self.roi_layer.data) == 0:
            return

        last_index = len(self.roi_layer.data) - 1
        last_shape_type = str(self.roi_layer.shape_type[last_index])
        if last_shape_type != "ellipse":
            self.roi_layer.data = []
            self.roi_layer.mode = "select"
            self.waiting_for_roi_finish = False
            self.refresh_roi_mask()
            self.roi_label.setText("ROI: Only an ellipse is supported")
            self.status_label.setText("Status: Non-ellipse ROI cleared")
            return

        if last_index > 0:
            last_shape = np.asarray(self.roi_layer.data[last_index]).copy()
            self.roi_layer.data = [last_shape]
            self.roi_layer.shape_type = ["ellipse"]
            last_index = 0

        self.roi_layer.selected_data = {last_index}
        self.roi_layer.mode = "select"

        self.waiting_for_roi_finish = False

        self.refresh_roi_mask()
        self.roi_label.setText("ROI: Ellipse selected; outside area masked")
        self.status_label.setText("Status: Ellipse ROI selected / editable")

        print("ROI drawing finished. Switched to select mode.")

    def get_current_inside_mask(self):
        if (
            self.roi_layer is None
            or len(self.roi_layer.data) != 1
            or str(self.roi_layer.shape_type[0]) != "ellipse"
        ):
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

    def refresh_roi_mask(self):
        if self.roi_layer is None or len(self.roi_layer.data) != 1:
            self.roi_mask = None
            self.roi_params = None
            self.current_display_mask = None
            self.display_mask_enabled = False
            return False

        inside_mask, roi_type, roi_info = self.get_current_inside_mask()

        if inside_mask is None:
            self.roi_mask = None
            self.roi_params = None
            self.current_display_mask = None
            self.display_mask_enabled = False
            return False

        self.roi_mask = np.asarray(inside_mask, dtype=bool)
        self.current_display_mask = self.make_display_mask_from_raw_mask(
            self.roi_mask
        )
        self.roi_params = roi_info
        self.display_mask_enabled = True
        return True

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
            if shape_type != "ellipse":
                raise ValueError("Only ellipse ROIs are supported.")

            self.waiting_for_roi_finish = False
            self.roi_layer.data = []
            self.roi_layer.add(shape_data, shape_type=shape_type)
            self.roi_layer.selected_data = {0}
            self.roi_layer.mode = "select"
            self.roi_layer.visible = True
            self.viewer.layers.selection.active = self.roi_layer

            self.roi_params = roi_params
            self.roi_mask = None
            self.refresh_roi_mask()
            self.last_roi_count = 1

            self.roi_label.setText("ROI: Ellipse reloaded; outside area masked")
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

        self.refresh_roi_mask()

        roi_json_path, roi_mask_path = save_roi_files(
            os.getcwd(),
            self.roi_params,
            self.roi_mask,
        )

        self.roi_label.setText("ROI: Ellipse saved")
        self.status_label.setText("Status: ROI and mask saved")

        print("ROI ellipse saved.")
        print("ROI parameters:", self.roi_params)
        print("Saved:", roi_json_path)
        print("Saved:", roi_mask_path)
