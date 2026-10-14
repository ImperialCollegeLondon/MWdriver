"""Filesystem path allocation helpers for phase-capture sessions."""

import os
import json
from dataclasses import dataclass

import imageio.v2 as imageio
import numpy as np

from phase_psi import DEFAULT_ALGORITHM, get_algorithm


@dataclass(frozen=True)
class NotebookGenerationResult:
    generated: bool
    path: str | None
    message: str


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
    analysis_notebook_path,
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
        "analysis_notebook_path": analysis_notebook_path,
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


def generate_phase_analysis_notebook(
    template_path,
    session_dir,
    npy_path,
    timestamp,
):
    if not os.path.exists(template_path):
        return NotebookGenerationResult(
            False,
            None,
            f"Phase notebook template not found: {template_path}",
        )

    try:
        with open(template_path, "r", encoding="utf-8") as file:
            notebook = json.load(file)

        npy_filename = os.path.basename(npy_path)
        new_load_cell_source = [
            f'with open("{npy_filename}", "rb") as f:\n',
            "    data = np.load(f)\n",
        ]
        found_load_cell = False
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") != "code":
                continue

            source = cell.get("source", "")
            source_text = (
                "".join(source) if isinstance(source, list) else str(source)
            )
            if "np.load" in source_text and "_raw_stack.npy" in source_text:
                cell["source"] = new_load_cell_source
                found_load_cell = True
            cell["execution_count"] = None
            cell["outputs"] = []

        if not found_load_cell:
            notebook.setdefault("cells", []).insert(1, {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": new_load_cell_source,
            })

        output_path = os.path.join(
            session_dir,
            f"phase_shifts_{timestamp}.ipynb",
        )
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(notebook, file, indent=1, ensure_ascii=False)
        return NotebookGenerationResult(
            True,
            output_path,
            f"Generated phase analysis notebook: {output_path}",
        )
    except Exception as error:
        return NotebookGenerationResult(
            False,
            None,
            f"Create phase analysis notebook error: {error}",
        )