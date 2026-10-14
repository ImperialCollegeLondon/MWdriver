"""Pure validation rules for six-point phase recalibration."""

import math


def validate_trigger_positions_ms(
    trigger_positions_ms,
    frame_count,
    falling_edge_ms,
    minimum_spacing_ms,
):
    if len(trigger_positions_ms) != frame_count:
        raise ValueError(f"Exactly {frame_count} trigger positions are required.")

    positions = [int(round(value)) for value in trigger_positions_ms]

    for value in positions:
        if value < 0 or value >= falling_edge_ms:
            raise ValueError(f"Trigger position out of range: {value} ms")

    for index in range(1, len(positions)):
        if positions[index] <= positions[index - 1]:
            raise ValueError("Trigger positions must be strictly increasing.")

    actual_minimum_spacing = min(
        positions[index + 1] - positions[index]
        for index in range(frame_count - 1)
    )
    if actual_minimum_spacing < minimum_spacing_ms:
        raise ValueError(
            "Trigger spacing is too small for reliable pulse output: "
            f"min gap {actual_minimum_spacing} ms"
        )

    return positions


def validate_coarse_measurement(
    measured_phase_deg,
    frame_count,
    minimum_span_deg,
    minimum_total_abs_step_deg,
    minimum_max_step_deg,
    maximum_backtrack_deg,
):
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

    if total_span < minimum_span_deg:
        raise ValueError(
            "Coarse calibration span is too short: "
            f"{total_span:.2f} deg < {minimum_span_deg:.2f} deg"
        )

    if total_abs_step < minimum_total_abs_step_deg:
        raise ValueError(
            "Coarse total absolute phase step is too small: "
            f"{total_abs_step:.2f} deg < "
            f"{minimum_total_abs_step_deg:.2f} deg"
        )

    if max_abs_step < minimum_max_step_deg:
        raise ValueError(
            "Coarse maximum phase step is too small: "
            f"{max_abs_step:.2f} deg < {minimum_max_step_deg:.2f} deg"
        )

    if largest_backtrack > maximum_backtrack_deg:
        raise ValueError(
            "Coarse phase has excessive local backtracking: "
            f"{largest_backtrack:.2f} deg > "
            f"{maximum_backtrack_deg:.2f} deg"
        )

    return measured