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
        raw_vertices = roi_data[:, :2] * display_downsample
        if len(raw_vertices) == 2:
            center_y = (raw_vertices[0, 0] + raw_vertices[1, 0]) / 2.0
            center_x = (raw_vertices[0, 1] + raw_vertices[1, 1]) / 2.0
            axis_y = (raw_vertices[1, 0] - raw_vertices[0, 0]) / 2.0
            axis_x = (raw_vertices[1, 1] - raw_vertices[0, 1]) / 2.0
            raw_vertices = np.asarray([
                [center_y - axis_y, center_x - axis_x],
                [center_y - axis_y, center_x + axis_x],
                [center_y + axis_y, center_x + axis_x],
                [center_y + axis_y, center_x - axis_x],
            ])
        if raw_vertices.shape != (4, 2):
            return None, None, None

        center_y, center_x = np.mean(raw_vertices, axis=0)
        first_axis = (raw_vertices[1] - raw_vertices[0]) / 2.0
        second_axis = (raw_vertices[2] - raw_vertices[1]) / 2.0
        transform = np.column_stack((first_axis, second_axis))
        if abs(np.linalg.det(transform)) <= np.finfo(float).eps:
            return None, None, None
        inverse_transform = np.linalg.inv(transform)

        yy, xx = np.ogrid[:image_height, :image_width]
        delta_y = yy - center_y
        delta_x = xx - center_x
        first_coordinate = (
            inverse_transform[0, 0] * delta_y
            + inverse_transform[0, 1] * delta_x
        )
        second_coordinate = (
            inverse_transform[1, 0] * delta_y
            + inverse_transform[1, 1] * delta_x
        )
        inside_mask = first_coordinate ** 2 + second_coordinate ** 2 <= 1.0
        roi_info_type = "circle_or_ellipse"
        raw_coordinates = {
            "center_x": center_x,
            "center_y": center_y,
            "radius_x": float(np.linalg.norm(first_axis)),
            "radius_y": float(np.linalg.norm(second_axis)),
            "average_radius": float(
                (np.linalg.norm(first_axis) + np.linalg.norm(second_axis)) / 2.0
            ),
            "x_min": raw_x_min,
            "x_max": raw_x_max,
            "y_min": raw_y_min,
            "y_max": raw_y_max,
            "vertices": raw_vertices.tolist(),
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


def build_roi_selection(
    inside_mask,
    roi_info,
    mask_mode,
    display_downsample,
    describe_rejected_area=False,
):
    valid_mask, saved_mask_mode = build_valid_mask(inside_mask, mask_mode)
    roi_info["mask_mode"] = saved_mask_mode
    roi_info["mask_meaning"] = "roi_mask.npy: True means retained valid area"
    if describe_rejected_area:
        roi_info["mask_meaning"] += "; False means masked/rejected area"
    return (
        valid_mask,
        downsample_mask(valid_mask, display_downsample),
        roi_info,
        saved_mask_mode,
    )


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
