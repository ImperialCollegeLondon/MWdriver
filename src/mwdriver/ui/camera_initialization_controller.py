"""Native camera and Napari viewer initialisation for the desktop application."""

import napari
import numpy as np
from qtpy.QtCore import QTimer

from mwdriver.hardware.thorlabs_native_camera import NativeThorlabsCamera


class CameraInitializationController:
    def init_camera(self):
        print("Searching for Thorlabs camera...")
        self.cam = NativeThorlabsCamera()

        self.cam.set_exposure(self.exposure_s)

        try:
            self.cam.set_gain(self.gain_value)
            print(f"Gain: {self.gain_value}")
        except Exception as e:
            print("This camera may not support software gain control:", e)

        self.width, self.height = self.cam.get_detector_size()

        print("Camera info:", self.cam.get_device_info())
        print("Detector size:", self.width, self.height)
        print(f"Exposure: {self.exposure_s * 1000:.1f} ms")

        try:
            self.cam.set_trigger_mode("int")
            self.current_camera_mode = "live"
            print("Camera trigger mode: int / live view")
        except Exception as e:
            print("Warning: set_trigger_mode('int') failed:", e)
            print("Continuing with camera default acquisition mode.")

        self.cam.start_acquisition()
        print("Camera acquisition started in live view mode.")

    def init_viewer(self):
        display_height = self.height // self.display_downsample
        display_width = self.width // self.display_downsample

        dummy = np.zeros((display_height, display_width), dtype=np.uint16)

        self.viewer = napari.Viewer()

        self.viewer.window._qt_window.resize(1600, 900)
        self.viewer.window._qt_window.showMaximized()

        self.layer = self.viewer.add_image(
            dummy,
            name="Camera Preview",
            colormap="gray",
            contrast_limits=(0, 1023)
        )
        self.layer.events.visible.connect(
            self.on_camera_preview_visibility_changed
        )

        self.roi_layer = self.viewer.add_shapes(
            name="ROI",
            ndim=2,
            shape_type="rectangle",
            edge_width=3,
            edge_color="red",
            face_color=[1, 0, 0, 0.15]
        )

        self.roi_layer.events.data.connect(self.on_roi_data_changed)

        self.viewer.layers.selection.active = self.layer
        self.viewer.reset_view()

        print("Napari viewer ready.")
        print(
            f"Preview size: {display_width} x {display_height} "
            f"(downsample x{self.display_downsample})"
        )

        QTimer.singleShot(500, self.simplify_layer_controls)
        QTimer.singleShot(600, self.simplify_napari_interface)
        QTimer.singleShot(700, self.install_controls_resize_filter)
        QTimer.singleShot(1000, self.simplify_napari_interface)

        try:
            self.viewer.layers.events.removed.connect(
                self.on_phase_analysis_layer_removed
            )
            self.viewer.layers.selection.events.active.connect(
                self.on_active_layer_changed
            )
        except Exception as e:
            print("Layer controls active-change hook warning:", e)

