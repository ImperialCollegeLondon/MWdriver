"""Compatibility re-exports for the packaged phase-capture controller."""

from mwdriver.capture.phase_capture_controller import (
    EXTERNAL_TRIGGER_EDGE,
    MODULE_DIR,
    PHASE_CAPTURE_COUNT,
    RECALIBRATION_CONFIG,
    SAVE_DIR,
    TRIANGLE_WAVE_SUPPLY_VOLTAGE,
    PhaseCaptureController,
    triangle_wave_falling_edge_voltage,
)

__all__ = [
    "EXTERNAL_TRIGGER_EDGE",
    "MODULE_DIR",
    "PHASE_CAPTURE_COUNT",
    "RECALIBRATION_CONFIG",
    "SAVE_DIR",
    "TRIANGLE_WAVE_SUPPLY_VOLTAGE",
    "PhaseCaptureController",
    "triangle_wave_falling_edge_voltage",
]
