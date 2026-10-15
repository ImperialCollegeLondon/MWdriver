"""Compatibility re-exports for packaged profile drag geometry."""

from mwdriver.domain.profile_drag import (
    distance_point_to_segment,
    get_line_drag_target,
)

__all__ = ["distance_point_to_segment", "get_line_drag_target"]
