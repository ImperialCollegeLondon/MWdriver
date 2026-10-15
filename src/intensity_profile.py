"""Compatibility re-exports for packaged intensity-profile geometry."""

from mwdriver.domain.intensity_profile import (
    build_initial_profile_line,
    clamp_profile_line,
    move_profile_line,
    sample_line_profile,
)

__all__ = [
    "build_initial_profile_line",
    "clamp_profile_line",
    "move_profile_line",
    "sample_line_profile",
]
