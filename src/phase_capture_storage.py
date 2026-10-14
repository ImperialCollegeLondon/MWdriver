"""Filesystem path allocation helpers for phase-capture sessions."""

import os
import json
from dataclasses import dataclass

import imageio.v2 as imageio
import numpy as np

from phase_psi import DEFAULT_ALGORITHM, get_algorithm


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


def save_roi_snapshots(session_dir, roi_mask, roi_params):
    mask_path = None
    params_path = None
    warnings = []

    if roi_mask is not None:
        try:
            mask = np.asarray(roi_mask, dtype=bool)
            mask_path = os.path.join(session_dir, "roi_mask.npy")
            np.save(mask_path, mask)
        except Exception as error:
            warnings.append(f"Save ROI mask warning: {error}")
            mask_path = None

    if roi_params is not None:
        try:
            params_path = os.path.join(session_dir, "roi_params.json")
            with open(params_path, "w", encoding="utf-8") as file:
                json.dump(roi_params, file, indent=4, ensure_ascii=False)
        except Exception as error:
            warnings.append(f"Save ROI parameters warning: {error}")
            params_path = None

    return {
        "roi_mask_path": mask_path,
        "roi_params_path": params_path,
        "warnings": warnings,
    }


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
    roi_available,
    roi_mask_path,
    roi_params_path,
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
        "png_paths": list(png_paths),
        "phase_times": phase_times,
        "frame_timestamps": frame_timestamps,
        "roi": {
            "roi_available": roi_available,
            "roi_mask_path": roi_mask_path,
            "roi_params_path": roi_params_path,
            "mask_meaning": (
                "True means retained valid area; False means masked/rejected area"
            ),
        },
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
