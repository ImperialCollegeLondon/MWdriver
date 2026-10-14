"""Pure sampling of raw-frame intensity along a display-space line."""

import numpy as np


def build_initial_profile_line(image_height, image_width, display_downsample):
    display_height = image_height // display_downsample
    display_width = image_width // display_downsample
    center_y = display_height / 2.0
    return np.asarray(
        [
            [center_y, display_width * 0.25],
            [center_y, display_width * 0.75],
        ],
        dtype=float,
    )


def clamp_profile_line(line_vertices, image_height, image_width, display_downsample):
    line = np.asarray(line_vertices, dtype=float).copy()
    display_height = image_height // display_downsample
    display_width = image_width // display_downsample
    line[:, 0] = np.clip(line[:, 0], 0, display_height - 1)
    line[:, 1] = np.clip(line[:, 1], 0, display_width - 1)
    return line


def move_profile_line(line_vertices, drag_target, mouse_position, last_position=None):
    line = np.asarray(line_vertices, dtype=float).copy()
    mouse_position = np.asarray(mouse_position, dtype=float)
    if line.ndim != 2 or line.shape[0] < 2 or line.shape[1] < 2:
        return None, last_position

    if drag_target == "p1":
        line[0, :2] = mouse_position
        return line, mouse_position.copy()
    if drag_target == "p2":
        line[-1, :2] = mouse_position
        return line, mouse_position.copy()
    if drag_target == "line":
        if last_position is None:
            return None, mouse_position.copy()
        delta = mouse_position - np.asarray(last_position, dtype=float)
        line[:, :2] += delta
        return line, mouse_position.copy()
    return None, last_position


def sample_line_profile(raw_frame, line_vertices, display_downsample, max_points):
    if raw_frame is None:
        return None

    line = np.asarray(line_vertices, dtype=float)
    if line.ndim != 2 or line.shape[0] < 2 or line.shape[1] < 2:
        return None

    height, width = raw_frame.shape[:2]
    y1_display, x1_display = line[0, :2]
    y2_display, x2_display = line[-1, :2]
    y1 = y1_display * display_downsample
    x1 = x1_display * display_downsample
    y2 = y2_display * display_downsample
    x2 = x2_display * display_downsample
    length = float(np.hypot(x2 - x1, y2 - y1))
    if length < 2:
        return None

    point_count = int(min(max(length, 10), max_points))
    x_indices = np.clip(
        np.round(np.linspace(x1, x2, point_count)).astype(int),
        0,
        width - 1,
    )
    y_indices = np.clip(
        np.round(np.linspace(y1, y2, point_count)).astype(int),
        0,
        height - 1,
    )
    return np.asarray(raw_frame[y_indices, x_indices], dtype=float)
