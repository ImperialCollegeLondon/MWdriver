"""Intensity-profile controls, line interaction and plotting for the camera viewer."""

import numpy as np
import matplotlib.pyplot as plt
from qtpy.QtCore import QTimer

from profile_controls import create_profile_controls
from intensity_profile import (
    build_initial_profile_line,
    clamp_profile_line,
    move_profile_line,
    sample_line_profile as sample_raw_line_profile,
)
from profile_drag import get_line_drag_target


class CameraProfileController:
    def initialize_profile_controller_state(self):
        self.profile_layer = None
        self.profile_enabled = False
        self.profile_enabled_before_phase_capture = False
        self.profile_fig = None
        self.profile_ax = None
        self.profile_plot_line = None
        self.profile_update_counter = 0
        self.profile_update_interval = 3
        self.profile_sample_points = 800
        self.profile_controls_widget = None
        self.profile_dragging = False
        self.profile_drag_target = None
        self.profile_drag_last_pos = None
        self.profile_drag_threshold = 40

    def place_profile_controls_in_layer_controls(self):
        self.add_widget_to_mwdriver_dock(self.profile_controls_widget)

    def get_profile_line_data(self):
        if self.profile_layer is None:
            return None

        if len(self.profile_layer.data) == 0:
            return None

        line = np.asarray(self.profile_layer.data[-1], dtype=float)

        if line.shape[0] < 2:
            return None

        return line.copy()

    def set_profile_line_data(self, line):
        if self.profile_layer is None:
            return

        clamped_line = clamp_profile_line(
            line,
            self.height,
            self.width,
            self.display_downsample,
        )
        self.profile_layer.data = [clamped_line]

        self.profile_update_counter = self.profile_update_interval - 1
        self.update_intensity_profile()

    def get_profile_drag_target(self, mouse_pos):
        line = self.get_profile_line_data()
        if line is None:
            return None
        return get_line_drag_target(
            mouse_position=mouse_pos,
            line_vertices=line,
            threshold=self.profile_drag_threshold,
        )

    def move_profile_line_by_mouse(self, mouse_pos):
        line = self.get_profile_line_data()
        if line is None:
            return
        moved_line, last_position = move_profile_line(
            line,
            self.profile_drag_target,
            mouse_pos,
            self.profile_drag_last_pos,
        )
        self.profile_drag_last_pos = last_position
        if moved_line is None:
            return
        self.set_profile_line_data(moved_line)

    def on_profile_mouse_drag(self, viewer, event):
        """
        Viewer-level mouse drag callback.

        é¿å… generator already executing çš„åŽŸåˆ™ï¼š
        - æ‹–åŠ¨è¿‡ç¨‹ä¸­ä¸åˆ‡æ¢ active layer
        - æ‹–åŠ¨è¿‡ç¨‹ä¸­ä¸è°ƒç”¨ simplify_layer_controls()
        - æ‹–åŠ¨è¿‡ç¨‹ä¸­ä¸è°ƒç”¨ place_profile_controls_in_layer_controls()
        - åªåœ¨æ‹–åŠ¨ç»“æŸåŽåˆ·æ–°ä¸€æ¬¡å³ä¾§é¢æ¿
        """

        if not self.profile_enabled:
            return

        if self.profile_layer is None:
            return

        if len(self.profile_layer.data) == 0:
            return

        if self.profile_dragging:
            return

        if event.button != 1:
            return

        try:
            mouse_pos = np.asarray(event.position[-2:], dtype=float)
        except Exception:
            return

        drag_target = self.get_profile_drag_target(mouse_pos)

        if drag_target is None:
            return

        self.profile_dragging = True
        self.profile_drag_target = drag_target
        self.profile_drag_last_pos = mouse_pos

        try:
            event.handled = True
        except Exception:
            pass

        yield

        try:
            while event.type == "mouse_move":
                try:
                    mouse_pos = np.asarray(event.position[-2:], dtype=float)
                    self.move_profile_line_by_mouse(mouse_pos)

                    try:
                        event.handled = True
                    except Exception:
                        pass

                except Exception as e:
                    print("Profile drag error:", e)

                yield

        finally:
            self.profile_dragging = False
            self.profile_drag_target = None
            self.profile_drag_last_pos = None

            try:
                self.viewer.layers.selection.active = self.layer
            except Exception:
                pass

            QTimer.singleShot(50, self.simplify_layer_controls)
            QTimer.singleShot(100, self.place_profile_controls_in_layer_controls)
            QTimer.singleShot(150, self.place_camera_adjust_controls_in_layer_controls)
            QTimer.singleShot(200, self.place_camera_status_controls_in_layer_controls)

    def init_profile_controls(self):
        (
            self.profile_controls_widget,
            self.profile_button,
            self.clear_profile_button,
        ) = create_profile_controls()

        self.profile_button.clicked.connect(self.start_intensity_profile)
        self.clear_profile_button.clicked.connect(self.clear_intensity_profile)

        QTimer.singleShot(300, self.place_profile_controls_in_layer_controls)
        QTimer.singleShot(350, self.place_camera_adjust_controls_in_layer_controls)
        QTimer.singleShot(800, self.place_profile_controls_in_layer_controls)
        QTimer.singleShot(850, self.place_camera_adjust_controls_in_layer_controls)

        print("Embedded intensity profile controls ready.")

    def start_intensity_profile(self):
        if self.viewer.layers.selection.active is not self.layer:
            self.show_camera_preview()

        line_data = build_initial_profile_line(
            self.height,
            self.width,
            self.display_downsample,
        )

        if self.profile_layer is None:
            self.profile_layer = self.viewer.add_shapes(
                [line_data],
                name="Intensity Profile Line",
                ndim=2,
                shape_type="line",
                edge_width=4,
                edge_color="yellow",
                face_color=[1, 1, 0, 0]
            )

            try:
                self.profile_layer.editable = False
            except Exception:
                pass

        else:
            self.profile_layer.data = [line_data]
            self.profile_layer.shape_type = ["line"]
            self.profile_layer.visible = True

            try:
                self.profile_layer.editable = False
            except Exception:
                pass

        try:
            self.profile_layer.mode = "select"
            self.profile_layer.selected_data = {0}
        except Exception:
            pass

        self.viewer.layers.selection.active = self.layer

        self.profile_enabled = True
        self.profile_update_counter = 0
        self.profile_dragging = False
        self.profile_drag_target = None
        self.profile_drag_last_pos = None

        self.init_profile_plot_window()

        self.profile_label.setText("Profile: On - drag yellow line directly")
        self.status_label.setText(
            "Status: Intensity profile enabled - drag yellow line"
        )

        QTimer.singleShot(50, self.simplify_layer_controls)
        QTimer.singleShot(100, self.place_profile_controls_in_layer_controls)

        print("Intensity profile enabled. Camera Preview remains active.")

    def init_profile_plot_window(self):
        plt.ion()

        if self.profile_fig is None or not plt.fignum_exists(self.profile_fig.number):
            self.profile_fig, self.profile_ax = plt.subplots()
            self.profile_fig.canvas.manager.set_window_title("Intensity Profile")

            self.profile_plot_line, = self.profile_ax.plot([], [])

            self.profile_ax.set_title("Intensity Profile")
            self.profile_ax.set_xlabel("Position along line / pixel")
            self.profile_ax.set_ylabel("Raw intensity")
            self.profile_ax.grid(True)

            self.profile_fig.show()
        else:
            self.profile_ax.cla()
            self.profile_plot_line, = self.profile_ax.plot([], [])
            self.profile_ax.set_title("Intensity Profile")
            self.profile_ax.set_xlabel("Position along line / pixel")
            self.profile_ax.set_ylabel("Raw intensity")
            self.profile_ax.grid(True)
            self.profile_fig.show()

    def clear_intensity_profile(self):
        self.profile_enabled = False
        self.profile_dragging = False
        self.profile_drag_target = None
        self.profile_drag_last_pos = None

        if self.profile_layer is not None:
            try:
                self.viewer.layers.remove(self.profile_layer)
            except Exception:
                pass
            self.profile_layer = None

        if self.profile_fig is not None:
            try:
                plt.close(self.profile_fig)
            except Exception:
                pass

        self.profile_fig = None
        self.profile_ax = None
        self.profile_plot_line = None

        try:
            self.viewer.layers.selection.active = self.layer
        except Exception:
            pass

        self.profile_label.setText("Profile: Off")
        self.status_label.setText("Status: Intensity profile cleared")

        QTimer.singleShot(50, self.simplify_layer_controls)
        QTimer.singleShot(100, self.place_profile_controls_in_layer_controls)

        print("Intensity profile cleared.")

    def sample_line_profile(self):
        if not self.profile_enabled:
            return None

        if self.profile_layer is None or len(self.profile_layer.data) == 0:
            return None

        if self.latest_raw_frame is None:
            return None

        return sample_raw_line_profile(
            raw_frame=self.latest_raw_frame,
            line_vertices=self.profile_layer.data[-1],
            display_downsample=self.display_downsample,
            max_points=self.profile_sample_points,
        )

    def update_intensity_profile(self):
        if not self.profile_enabled:
            return

        if self.profile_fig is None or self.profile_ax is None:
            return

        self.profile_update_counter += 1

        if self.profile_update_counter % self.profile_update_interval != 0:
            return

        profile = self.sample_line_profile()

        if profile is None or len(profile) == 0:
            return

        x = np.arange(len(profile))

        if self.profile_plot_line is None:
            self.profile_plot_line, = self.profile_ax.plot(x, profile)
        else:
            self.profile_plot_line.set_data(x, profile)

        self.profile_ax.set_xlim(0, len(profile) - 1)

        y_min = float(np.min(profile))
        y_max = float(np.max(profile))

        if y_max <= y_min:
            y_max = y_min + 1

        margin = 0.05 * (y_max - y_min)
        self.profile_ax.set_ylim(y_min - margin, y_max + margin)

        self.profile_ax.set_title(
            f"Intensity Profile | min={y_min:.0f}, max={y_max:.0f}, contrast={y_max - y_min:.0f}"
        )

        try:
            self.profile_fig.canvas.draw_idle()
            self.profile_fig.canvas.flush_events()
        except Exception as e:
            print("Profile plot update error:", e)
