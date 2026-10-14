"""Pure conversion of Napari ROI vertices into raw-image masks and metadata."""

import numpy as np


def build_inside_mask(roi_vertices, roi_type, image_height, image_width, display_downsample):
    roi_data = np.asarray(roi_vertices, dtype=float)
    if roi_data.ndim != 2 or roi_data.shape[0] == 0 or roi_data.shape[1] < 2:
        return None, None, None

    y_min = float(np.min(roi_data[:, 0]))
    y_max = float(np.max(roi_data[:, 0]))
    x_min = float(np.min(roi_data[:, 1]))
    x_max = float(np.max(roi_data[:, 1]))
    raw_y_min = max(0, min(int(round(y_min * display_downsample)), image_height - 1))
    raw_y_max = max(0, min(int(round(y_max * display_downsample)), image_height))
    raw_x_min = max(0, min(int(round(x_min * display_downsample)), image_width - 1))
    raw_x_max = max(0, min(int(round(x_max * display_downsample)), image_width))
    if raw_y_max <= raw_y_min or raw_x_max <= raw_x_min:
        return None, None, None

    inside_mask = np.zeros((image_height, image_width), dtype=bool)
    if roi_type == "rectangle":
        inside_mask[raw_y_min:raw_y_max, raw_x_min:raw_x_max] = True
        roi_info_type = "rectangle"
        raw_coordinates = {
            "x_min": raw_x_min,
            "x_max": raw_x_max,
            "y_min": raw_y_min,
            "y_max": raw_y_max,
        }
    elif roi_type == "ellipse":
        center_y = (raw_y_min + raw_y_max) / 2.0
        center_x = (raw_x_min + raw_x_max) / 2.0
        radius_y = (raw_y_max - raw_y_min) / 2.0
        radius_x = (raw_x_max - raw_x_min) / 2.0
        if radius_x <= 0 or radius_y <= 0:
            return None, None, None
        yy, xx = np.ogrid[:image_height, :image_width]
        inside_mask = (
            ((xx - center_x) / radius_x) ** 2
            + ((yy - center_y) / radius_y) ** 2
        ) <= 1.0
        roi_info_type = "circle_or_ellipse"
        raw_coordinates = {
            "center_x": center_x,
            "center_y": center_y,
            "radius_x": radius_x,
            "radius_y": radius_y,
            "average_radius": (radius_x + radius_y) / 2.0,
            "x_min": raw_x_min,
            "x_max": raw_x_max,
            "y_min": raw_y_min,
            "y_max": raw_y_max,
        }
    else:
        return None, None, None

    roi_info = {
        "roi_type": roi_info_type,
        "display_coordinates": {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
        },
        "raw_coordinates": raw_coordinates,
        "image_size": {"width": image_width, "height": image_height},
        "display_downsample": display_downsample,
    }
    return inside_mask, roi_type, roi_info


def build_valid_mask(inside_mask, mask_mode):
    if mask_mode == "inside":
        return np.logical_not(inside_mask), "mask_inside_area"
    return np.asarray(inside_mask, dtype=bool), "mask_outside_area"


def downsample_mask(raw_mask, display_downsample):
    if raw_mask is None:
        return None
    if display_downsample > 1:
        raw_mask = raw_mask[::display_downsample, ::display_downsample]
    return np.ascontiguousarray(raw_mask)


def apply_mask_to_frame(display_frame, display_mask, enabled):
    if not enabled or display_mask is None or display_mask.shape != display_frame.shape:
        return display_frame
    masked_frame = display_frame.copy()
    masked_frame[np.logical_not(display_mask)] = 0
    return masked_frame
