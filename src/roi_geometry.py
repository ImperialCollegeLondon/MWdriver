"""Compatibility re-exports for packaged ROI geometry policies."""

from mwdriver.domain.roi_geometry import (
    apply_mask_to_frame,
    build_inside_mask,
    build_roi_selection,
    build_valid_mask,
    downsample_mask,
)

__all__ = [
    "apply_mask_to_frame",
    "build_inside_mask",
    "build_roi_selection",
    "build_valid_mask",
    "downsample_mask",
]
