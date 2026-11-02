"""Subprocess boundary for phase-capture verification."""

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Callable

import numpy as np
from mwdriver.runtime import build_worker_command

@dataclass(frozen=True)
class PhaseCheckResult:
    success: bool
    message: str
    return_code: int | None
    output_figure_path: str | None = None
    output_json_path: str | None = None
    stdout: str | bytes = ""
    stderr: str | bytes = ""


def run_phase_check(
    check_script,
    session_dir,
    working_directory,
    timeout_s,
    *,
    python_executable=None,
    runner: Callable = subprocess.run,
    roi_mask=None,
    diagnostic_plots=True,
):
    session_dir = os.path.abspath(session_dir)
    frozen = getattr(sys, "frozen", False)
    if not frozen and not os.path.exists(check_script):
        message = (
            "Automatic phase check skipped: checker script not found: "
            f"{check_script}"
        )
        return PhaseCheckResult(False, message, None)

    command = build_worker_command(
        "phase-check",
        check_script,
        [session_dir],
        python_executable=python_executable,
    )
    if not diagnostic_plots:
        command.append("--no-diagnostic-plots")
    temporary_mask_path = None
    output_figure_path = os.path.join(
        session_dir,
        "measured_vs_expected_phase_annotated.png",
    )
    output_json_path = os.path.join(session_dir, "measured_phase_check.json")

    try:
        if roi_mask is not None:
            temporary_mask = tempfile.NamedTemporaryFile(
                prefix="mwdriver_recalibration_roi_",
                suffix=".npy",
                delete=False,
            )
            temporary_mask_path = temporary_mask.name
            temporary_mask.close()
            np.save(temporary_mask_path, np.asarray(roi_mask, dtype=bool))
            command.extend(["--roi-mask", temporary_mask_path])

        completed_process = runner(
            command,
            cwd=session_dir if frozen else working_directory,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        stdout_text = completed_process.stdout or ""
        stderr_text = completed_process.stderr or ""
        success = (
            completed_process.returncode == 0
            and os.path.exists(output_figure_path)
            and os.path.exists(output_json_path)
        )
        if success:
            message = (
                "Automatic phase check completed: "
                "measured_vs_expected_phase_annotated.png and "
                "measured_phase_check.json saved"
            )
        else:
            message = (
                "Automatic phase check failed or output files are missing. "
                f"Return code: {completed_process.returncode}"
            )
        return PhaseCheckResult(
            bool(success),
            message,
            int(completed_process.returncode),
            output_figure_path,
            output_json_path,
            stdout_text,
            stderr_text,
        )
    except subprocess.TimeoutExpired as error:
        message = f"Automatic phase check timed out after {timeout_s} seconds."
        return PhaseCheckResult(
            False,
            message,
            None,
            output_figure_path,
            output_json_path,
            error.stdout or "",
            error.stderr or "",
        )
    except Exception as error:
        message = f"Automatic phase check error: {error}"
        return PhaseCheckResult(
            False,
            message,
            None,
            output_figure_path,
            output_json_path,
        )
    finally:
        if temporary_mask_path and os.path.exists(temporary_mask_path):
            os.remove(temporary_mask_path)