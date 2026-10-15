"""Pure geometry for intensity-profile line dragging."""

import numpy as np


def distance_point_to_segment(point, start, end):
    point = np.asarray(point, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    segment = end - start
    segment_length_squared = float(np.dot(segment, segment))
    if segment_length_squared <= 1e-12:
        return float(np.linalg.norm(point - start))

    projection = float(np.dot(point - start, segment) / segment_length_squared)
    projection = max(0.0, min(1.0, projection))
    closest = start + projection * segment
    return float(np.linalg.norm(point - closest))


def get_line_drag_target(mouse_position, line_vertices, threshold):
    line = np.asarray(line_vertices, dtype=float)
    mouse_position = np.asarray(mouse_position, dtype=float)
    if line.ndim != 2 or line.shape[0] < 2 or line.shape[1] < 2:
        return None

    start = line[0, :2]
    end = line[-1, :2]
    if float(np.linalg.norm(mouse_position - start)) <= threshold:
        return "p1"
    if float(np.linalg.norm(mouse_position - end)) <= threshold:
        return "p2"
    if distance_point_to_segment(mouse_position, start, end) <= threshold:
        return "line"
    return None
