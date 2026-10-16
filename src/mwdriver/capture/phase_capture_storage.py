"""Filesystem path allocation helpers for phase-capture sessions."""

import os
import json
from dataclasses import dataclass

import imageio.v2 as imageio
import numpy as np

from mwdriver.domain.phase_psi import DEFAULT_ALGORITHM, get_algorithm


@dataclass(frozen=True)
class PhaseSaveResult:
    save_success: bool
    session_dir: str | None
    raw_stack_path: str | None
    metadata_path: str | None
    phase_check_success: bool | None = None
    phase_check_json_path: str | None = None
    phase_check_figure_path: str | None = None
    phase_voltage_plot_path: str | None = None
    error: str | None = None


def allocate_capture_session_dir(save_dir, session_prefix, timestamp):
    os.makedirs(save_dir, exist_ok=True)

    base_name = f"{session_prefix}_{timestamp}"
    suffix = 0
    while True:
        session_name = (
            base_name if suffix == 0 else f"{base_name}_{suffix:02d}"
        )
        session_dir = os.path.join(save_dir, session_name)
        try:
            os.mkdir(session_dir)
            return session_dir
        except FileExistsError:
            suffix += 1


def raw_to_viewable_uint8(frame):
    frame = np.asarray(frame)
    frame_float = frame.astype(np.float32)

    low = np.percentile(frame_float, 1)
    high = np.percentile(frame_float, 99.5)

    if high <= low:
        low = float(np.min(frame_float))
        high = float(np.max(frame_float))

    if high <= low:
        return np.zeros(frame.shape, dtype=np.uint8)

    frame_uint8 = (frame_float - low) / (high - low) * 255.0
    return np.clip(frame_uint8, 0, 255).astype(np.uint8)


def validate_phase_frame_stack(frames, psi_algorithm_id=DEFAULT_ALGORITHM):
    expected_count = len(get_algorithm(psi_algorithm_id)["expected_phase_deg"])
    try:
        stack = np.asarray(frames)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Phase frames must be equally shaped 2D images."
        ) from error

    if stack.ndim != 3:
        raise ValueError(
            "Expected phase frame stack with shape "
            f"({expected_count}, H, W), got {stack.shape}."
        )
    if stack.shape[0] != expected_count:
        raise ValueError(
            f"Expected {expected_count} phase frames, got {stack.shape[0]}."
        )
    return stack


def save_raw_phase_stack(session_dir, session_prefix, timestamp, frames):
    stack = np.asarray(frames)
    path = os.path.join(
        session_dir,
        f"{session_prefix}_{timestamp}_raw_stack.npy",
    )
    np.save(path, stack)
    return path, stack


def save_phase_pngs(session_dir, timestamp, frames):
    records = []
    for phase_index, frame in enumerate(frames):
        viewable_frame = raw_to_viewable_uint8(frame)
        path = os.path.join(session_dir, f"phase_{phase_index}_{timestamp}.png")
        imageio.imwrite(path, viewable_frame)
        records.append({
            "path": path,
            "raw_min": int(np.min(frame)),
            "raw_max": int(np.max(frame)),
            "png_min": int(np.min(viewable_frame)),
            "png_max": int(np.max(viewable_frame)),
        })
    return records


def save_phase_capture_metadata(
    session_dir,
    timestamp,
    session_prefix,
    frames,
    phase_times,
    frame_timestamps,
    png_paths,
    psi_algorithm_id=DEFAULT_ALGORITHM,
):
    algorithm = get_algorithm(psi_algorithm_id)
    metadata = {
        "timestamp": timestamp,
        "capture_type": "mixed_live_view_hardware_trigger_phase_capture_6_frames",
        "phase_count": len(frames),
        "session_dir": session_dir,
        "raw_stack_path": os.path.join(
            session_dir,
            f"{session_prefix}_{timestamp}_raw_stack.npy",
        ),
        "analysis_notebook_path": None,
        "png_paths": list(png_paths),
        "phase_times": phase_times,
        "frame_timestamps": frame_timestamps,
        "phase_capture_model": {
            "algorithm_id": psi_algorithm_id,
            "algorithm_name": algorithm["label"],
            "trigger_source": "Pico GP12 hardware trigger",
            "trigger_region": "selected one-phase-cycle window on the falling edge",
            "phase_order": "phase_0 to phase_5",
            "sampling_rule": (
                "The selected one-cycle phase window is divided into 6 equal intervals. "
                "Seven boundary points are defined, but only the first 6 are captured. "
                "The final 6/6 boundary point is not captured."
            ),
            "expected_phase_step_deg": algorithm["phase_step_deg"],
            "expected_phase_positions_deg": list(algorithm["expected_phase_deg"]),
        },
        "note": (
            "Frames are ordered as phase_0 to phase_5. "
            "They are externally triggered by Pico GP12 within a selected one-phase-cycle window "
            "on the falling edge of one triangle wave. "
            "The selected one-cycle window is divided into six equal intervals. "
            "Only the first six boundary points are captured, giving expected phase positions "
            f"{', '.join(str(int(value)) for value in algorithm['expected_phase_deg'])} degrees. "
            "The camera returns to live view after capture."
        ),
    }
    path = os.path.join(
        session_dir,
        f"{session_prefix}_{timestamp}_metadata.json",
    )
    with open(path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)
    return path, metadata
