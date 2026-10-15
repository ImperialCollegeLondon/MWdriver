"""Compatibility re-exports for the packaged recalibration domain module."""

from mwdriver.domain.recalibration import (
    DEFAULT_RECALIBRATION_CONFIG,
    IncompleteCaptureDecision,
    RecalibrationConfig,
    build_even_trigger_positions_ms,
    compute_coarse_refined_positions_ms,
    compute_iterative_refined_positions_ms,
    decide_incomplete_capture_action,
    decide_verification_capture_action,
    enforce_minimum_trigger_spacing_ms,
    is_strict_recalibration_pass,
    score_recalibration_candidate,
    select_best_recalibration_candidate,
    should_auto_restart_recalibration,
    validate_coarse_measurement,
    validate_trigger_positions_ms,
)

__all__ = [
    "DEFAULT_RECALIBRATION_CONFIG",
    "IncompleteCaptureDecision",
    "RecalibrationConfig",
    "build_even_trigger_positions_ms",
    "compute_coarse_refined_positions_ms",
    "compute_iterative_refined_positions_ms",
    "decide_incomplete_capture_action",
    "decide_verification_capture_action",
    "enforce_minimum_trigger_spacing_ms",
    "is_strict_recalibration_pass",
    "score_recalibration_candidate",
    "select_best_recalibration_candidate",
    "should_auto_restart_recalibration",
    "validate_coarse_measurement",
    "validate_trigger_positions_ms",
]
