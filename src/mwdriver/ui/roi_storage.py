"""Persistence helpers for saved ROI geometry and masks."""

import json
from pathlib import Path

import numpy as np


def save_roi_files(directory, roi_params, roi_mask):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    params_path = directory / "roi_params.json"
    mask_path = directory / "roi_mask.npy"

    with params_path.open("w", encoding="utf-8") as file:
        json.dump(roi_params, file, indent=4)
    np.save(mask_path, np.asarray(roi_mask, dtype=bool))
    return params_path, mask_path


def load_roi_files(directory, image_height, image_width, display_downsample):
    directory = Path(directory)
    params_path = directory / "roi_params.json"
    mask_path = directory / "roi_mask.npy"
    with params_path.open("r", encoding="utf-8") as file:
        roi_params = json.load(file)
    roi_mask = np.asarray(np.load(mask_path, allow_pickle=False), dtype=bool)

    expected_shape = (image_height, image_width)
    if roi_mask.shape != expected_shape:
        raise ValueError(
            f"saved mask shape {roi_mask.shape} does not match "
            f"camera shape {expected_shape}"
        )

    raw_coordinates = roi_params.get("raw_coordinates", {})
    roi_type = roi_params.get("roi_type")
    if roi_type == "circle_or_ellipse":
        vertices = raw_coordinates.get("vertices")
        if vertices is not None:
            shape_data = np.asarray(vertices, dtype=float) / display_downsample
            return roi_params, roi_mask, shape_data, "ellipse"

    x_min = float(raw_coordinates["x_min"]) / display_downsample
    x_max = float(raw_coordinates["x_max"]) / display_downsample
    y_min = float(raw_coordinates["y_min"]) / display_downsample
    y_max = float(raw_coordinates["y_max"]) / display_downsample

    if roi_type == "rectangle":
        shape_type = "rectangle"
    elif roi_type == "circle_or_ellipse":
        shape_type = "ellipse"
    else:
        raise ValueError(f"unsupported ROI type: {roi_type!r}")

    shape_data = np.array([
        [y_min, x_min],
        [y_min, x_max],
        [y_max, x_max],
        [y_max, x_min],
    ], dtype=float)
    return roi_params, roi_mask, shape_data, shape_type
