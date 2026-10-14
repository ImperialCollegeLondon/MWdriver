"""Pure validation rules for six-point phase recalibration."""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RecalibrationConfig:
    frame_count: int = 6
    falling_edge_ms: int = 650
    minimum_trigger_spacing_ms: float = 6.0
    safe_trigger_spacing_ms: float = 25.0
    edge_guard_ms: float = 100.0
    error_tolerance_deg: float = 5.0
    retry_error_limit_deg: float = 10.0
    max_refinement_rounds: int = 3
    max_incomplete_verification_retries: int = 2
    minimum_error_over_5_points: int = 2
    minimum_error_over_5_only_points: int = 3
    max_auto_restarts: int = 1
    max_time_step_ms: float = 20.0
    time_update_damping: float = 0.7
    large_error_time_step_ms: float = 40.0
    large_error_damping: float = 1.0
    minimum_coarse_span_deg: float = 220.0
    minimum_coarse_total_abs_step_deg: float = 220.0
    minimum_coarse_max_step_deg: float = 35.0
    maximum_coarse_backtrack_deg: float = 25.0

    @property
    def max_verification_captures(self):
        return self.max_refinement_rounds + 1


DEFAULT_RECALIBRATION_CONFIG = RecalibrationConfig()


@dataclass(frozen=True)
class IncompleteCaptureDecision:
    action: str


def decide_incomplete_capture_action(
    recalibration_active,
    recalibration_stage,
    incomplete_retry_count,
    has_previous_candidate,
    config,
):
    if not recalibration_active:
        return IncompleteCaptureDecision("none")
    if recalibration_stage == "verification_capture":
        if incomplete_retry_count < config.max_incomplete_verification_retries:
            return IncompleteCaptureDecision("retry")
        if has_previous_candidate:
            return IncompleteCaptureDecision("keep_best")
    return IncompleteCaptureDecision("abort")


def _minimum_spacing(config, camera_required_spacing_ms, requested_spacing_ms=None):
    return max(
        config.safe_trigger_spacing_ms,
        config.minimum_trigger_spacing_ms,
        camera_required_spacing_ms,
        requested_spacing_ms or config.minimum_trigger_spacing_ms,
    )


def build_even_trigger_positions_ms(config):
    frame_count = config.frame_count
    lower_bound_ms = config.edge_guard_ms
    upper_bound_ms = config.falling_edge_ms - 1 - config.edge_guard_ms
    if frame_count < 2:
        raise ValueError("At least two trigger positions are required.")
    if upper_bound_ms <= lower_bound_ms:
        raise ValueError("Upper trigger bound must exceed lower trigger bound.")

    return [
        int(round(value))
        for value in np.linspace(lower_bound_ms, upper_bound_ms, frame_count)
    ]


def enforce_minimum_trigger_spacing_ms(
    trigger_positions_ms,
    config,
    camera_required_spacing_ms,
    requested_spacing_ms=None,
):
    frame_count = config.frame_count
    minimum_spacing_ms = _minimum_spacing(
        config,
        camera_required_spacing_ms,
        requested_spacing_ms,
    )
    lower_bound_ms = config.edge_guard_ms
    upper_bound_ms = config.falling_edge_ms - 1 - config.edge_guard_ms
    if len(trigger_positions_ms) != frame_count:
        raise ValueError(f"Exactly {frame_count} trigger positions are required.")

    required_span = minimum_spacing_ms * (frame_count - 1)
    if required_span > upper_bound_ms - lower_bound_ms:
        raise ValueError(
            f"{frame_count} camera-safe triggers need {required_span:g} ms, "
            f"but the calibration window is only "
            f"{upper_bound_ms - lower_bound_ms:g} ms. "
            "Reduce exposure or use a longer waveform."
        )

    adjusted = [float(value) for value in trigger_positions_ms]

    for index in range(1, frame_count):
        minimum_allowed = adjusted[index - 1] + minimum_spacing_ms
        if adjusted[index] < minimum_allowed:
            adjusted[index] = minimum_allowed

    if adjusted[-1] > upper_bound_ms:
        adjusted[-1] = upper_bound_ms
        for index in range(frame_count - 2, -1, -1):
            maximum_allowed = adjusted[index + 1] - minimum_spacing_ms
            if adjusted[index] > maximum_allowed:
                adjusted[index] = maximum_allowed

    if adjusted[0] < lower_bound_ms:
        adjusted[0] = lower_bound_ms
        for index in range(1, frame_count):
            minimum_allowed = adjusted[index - 1] + minimum_spacing_ms
            if adjusted[index] < minimum_allowed:
                adjusted[index] = minimum_allowed

    return [int(round(value)) for value in adjusted]


def compute_coarse_refined_positions_ms(
    coarse_positions_ms,
    measured_phase_deg,
    target_phase_deg,
    *,
    config,
    camera_required_spacing_ms,
):
    frame_count = config.frame_count
    coarse_positions = np.asarray(coarse_positions_ms, dtype=np.float64)
    measured = np.asarray(measured_phase_deg, dtype=np.float64)
    target_phase = np.asarray(target_phase_deg, dtype=np.float64)

    if coarse_positions.shape != (frame_count,):
        raise ValueError(
            f"Coarse trigger count mismatch: expected {frame_count}, "
            f"got {coarse_positions.shape[0] if coarse_positions.ndim else 0}"
        )
    if measured.shape != (frame_count,):
        raise ValueError(
            f"Measured phase count mismatch: expected {frame_count}, "
            f"got {measured.shape[0] if measured.ndim else 0}"
        )

    measured = np.asarray(
        validate_coarse_measurement(
            measured,
            config,
        ),
        dtype=np.float64,
    )

    for index in range(1, frame_count):
        if measured[index] <= measured[index - 1]:
            measured[index] = measured[index - 1] + 1e-3

    if measured[-1] < float(target_phase[-1]):
        raise ValueError(
            "Coarse calibration span is too short: "
            f"last measured phase {measured[-1]:.2f} deg"
        )

    upper_guarded_bound = config.falling_edge_ms - 1 - config.edge_guard_ms
    target_positions = np.interp(target_phase, measured, coarse_positions)
    target_positions = np.clip(
        target_positions,
        config.edge_guard_ms,
        upper_guarded_bound,
    )
    target_positions = enforce_minimum_trigger_spacing_ms(
        [int(round(value)) for value in target_positions],
        config=config,
        camera_required_spacing_ms=camera_required_spacing_ms,
    )

    return validate_trigger_positions_ms(
        target_positions,
        config=config,
    )


def compute_iterative_refined_positions_ms(
    current_positions_ms,
    measured_phase_deg,
    target_phase_deg,
    *,
    config,
    camera_required_spacing_ms,
):
    frame_count = config.frame_count
    current_positions = np.asarray(current_positions_ms, dtype=np.float64)
    measured = np.asarray(measured_phase_deg, dtype=np.float64)
    target_phase = np.asarray(target_phase_deg, dtype=np.float64)

    if current_positions.shape != (frame_count,):
        raise ValueError("Current trigger count mismatch for iterative refine.")
    if measured.shape != (frame_count,):
        raise ValueError("Measured phase count mismatch for iterative refine.")

    measured = measured - measured[0]
    if not np.all(np.isfinite(measured)):
        raise ValueError("Measured phase contains NaN or Inf during refinement.")

    monotonic_phase = measured.copy()
    for index in range(1, frame_count):
        monotonic_phase[index] = max(
            monotonic_phase[index],
            monotonic_phase[index - 1] + 1e-3,
        )

    desired_positions = np.interp(
        target_phase,
        monotonic_phase,
        current_positions,
    )
    first_slope = (
        (current_positions[1] - current_positions[0])
        / (monotonic_phase[1] - monotonic_phase[0])
    )
    last_slope = (
        (current_positions[-1] - current_positions[-2])
        / (monotonic_phase[-1] - monotonic_phase[-2])
    )

    for index, target_value in enumerate(target_phase):
        if target_value < monotonic_phase[0]:
            desired_positions[index] = (
                current_positions[0]
                + (target_value - monotonic_phase[0]) * first_slope
            )
        elif target_value > monotonic_phase[-1]:
            desired_positions[index] = (
                current_positions[-1]
                + (target_value - monotonic_phase[-1]) * last_slope
            )

    max_abs_error = float(np.max(np.abs(measured - target_phase)))
    if max_abs_error > config.retry_error_limit_deg:
        damping = config.large_error_damping
        max_time_step_ms = config.large_error_time_step_ms
    else:
        damping = config.time_update_damping
        max_time_step_ms = config.max_time_step_ms

    updated = current_positions + damping * (desired_positions - current_positions)
    updated[0] = current_positions[0]
    updated = current_positions + np.clip(
        updated - current_positions,
        -max_time_step_ms,
        max_time_step_ms,
    )

    lower_bound = float(config.edge_guard_ms)
    upper_bound = float(config.falling_edge_ms - 1 - config.edge_guard_ms)
    updated[0] = float(np.clip(updated[0], lower_bound, upper_bound))
    rounded = [int(round(value)) for value in updated]
    rounded = enforce_minimum_trigger_spacing_ms(
        rounded,
        config=config,
        camera_required_spacing_ms=camera_required_spacing_ms,
    )

    return validate_trigger_positions_ms(
        rounded,
        config=config,
    )


def validate_trigger_positions_ms(
    trigger_positions_ms,
    config,
):
    frame_count = config.frame_count
    if len(trigger_positions_ms) != frame_count:
        raise ValueError(f"Exactly {frame_count} trigger positions are required.")

    positions = [int(round(value)) for value in trigger_positions_ms]

    for value in positions:
        if value < 0 or value >= config.falling_edge_ms:
            raise ValueError(f"Trigger position out of range: {value} ms")

    for index in range(1, len(positions)):
        if positions[index] <= positions[index - 1]:
            raise ValueError("Trigger positions must be strictly increasing.")

    actual_minimum_spacing = min(
        positions[index + 1] - positions[index]
        for index in range(frame_count - 1)
    )
    if actual_minimum_spacing < config.minimum_trigger_spacing_ms:
        raise ValueError(
            "Trigger spacing is too small for reliable pulse output: "
            f"min gap {actual_minimum_spacing} ms"
        )

    return positions


def validate_coarse_measurement(
    measured_phase_deg,
    config,
):
    frame_count = config.frame_count
    try:
        measured = [float(value) for value in measured_phase_deg]
    except (TypeError, ValueError):
        measured = []

    if len(measured) != frame_count:
        raise ValueError(
            f"Coarse measured phase must contain exactly {frame_count} values."
        )

    if not all(math.isfinite(value) for value in measured):
        raise ValueError("Coarse measured phase contains NaN or Inf.")

    start_phase = measured[0]
    measured = [value - start_phase for value in measured]
    steps = [measured[index + 1] - measured[index] for index in range(frame_count - 1)]

    total_span = float(measured[-1])
    total_abs_step = sum(abs(step) for step in steps)
    max_abs_step = max(abs(step) for step in steps) if steps else 0.0
    largest_backtrack = max(-min(steps), 0.0) if steps else 0.0

    if total_span < config.minimum_coarse_span_deg:
        raise ValueError(
            "Coarse calibration span is too short: "
            f"{total_span:.2f} deg < {config.minimum_coarse_span_deg:.2f} deg"
        )

    if total_abs_step < config.minimum_coarse_total_abs_step_deg:
        raise ValueError(
            "Coarse total absolute phase step is too small: "
            f"{total_abs_step:.2f} deg < "
            f"{config.minimum_coarse_total_abs_step_deg:.2f} deg"
        )

    if max_abs_step < config.minimum_coarse_max_step_deg:
        raise ValueError(
            "Coarse maximum phase step is too small: "
            f"{max_abs_step:.2f} deg < {config.minimum_coarse_max_step_deg:.2f} deg"
        )

    if largest_backtrack > config.maximum_coarse_backtrack_deg:
        raise ValueError(
            "Coarse phase has excessive local backtracking: "
            f"{largest_backtrack:.2f} deg > "
            f"{config.maximum_coarse_backtrack_deg:.2f} deg"
        )

    return measured


def is_strict_recalibration_pass(phase_error_deg, config):
    try:
        errors = [float(value) for value in phase_error_deg]
    except (TypeError, ValueError):
        return False

    return bool(errors) and all(
        math.isfinite(error) and abs(error) <= config.error_tolerance_deg
        for error in errors
    )


def score_recalibration_candidate(
    session_dir,
    measured_phase_deg,
    target_phase_deg,
    trigger_positions_ms,
    config,
):
    measured = np.asarray(measured_phase_deg, dtype=np.float64)
    target_phase = np.asarray(target_phase_deg, dtype=np.float64)
    if measured.ndim != 1 or measured.size != target_phase.size or measured.size == 0:
        raise ValueError(
            "Verification measurement and target must have the same non-zero length."
        )
    if not np.all(np.isfinite(measured)) or not np.all(np.isfinite(target_phase)):
        raise ValueError("Verification measurement or target contains NaN or Inf.")

    measured_for_error = measured - measured[0]
    phase_error = measured_for_error - target_phase
    absolute_error = np.abs(phase_error)
    count_error_over_tolerance = int(
        np.sum(absolute_error > config.error_tolerance_deg)
    )
    count_error_over_retry_limit = int(
        np.sum(absolute_error > config.retry_error_limit_deg)
    )
    max_abs_error = float(np.max(absolute_error))
    rms_error = float(np.sqrt(np.mean(np.square(phase_error))))
    mae_error = float(np.mean(absolute_error))

    return {
        "session_dir": session_dir,
        "measured_phase_deg": measured.tolist(),
        "phase_error_deg": phase_error.tolist(),
        "trigger_positions_ms": list(trigger_positions_ms),
        "count_error_over_5": count_error_over_tolerance,
        "count_error_over_10": count_error_over_retry_limit,
        "max_abs_error": max_abs_error,
        "rms_error": rms_error,
        "mae_error": mae_error,
        "score": (
            count_error_over_retry_limit,
            max_abs_error,
            count_error_over_tolerance,
            rms_error,
            mae_error,
        ),
    }


def should_auto_restart_recalibration(
    candidate,
    config,
):
    if not isinstance(candidate, dict):
        return False

    count_error_over_10 = int(candidate.get("count_error_over_10", 0))
    count_error_over_5 = int(candidate.get("count_error_over_5", 0))
    if (
        count_error_over_10 > 0
        and count_error_over_5 >= config.minimum_error_over_5_points
    ):
        return True

    return count_error_over_5 >= config.minimum_error_over_5_only_points