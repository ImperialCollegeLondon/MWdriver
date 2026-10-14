"""Subprocess boundary for phase-capture verification."""

import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable


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
):
    session_dir = os.path.abspath(session_dir)
    if not os.path.exists(check_script):
        message = (
            "Automatic phase check skipped: checker script not found: "
            f"{check_script}"
        )
        return PhaseCheckResult(False, message, None)

    if python_executable is None:
        python_executable = sys.executable
    command = [python_executable, check_script, session_dir]
    output_figure_path = os.path.join(
        session_dir,
        "measured_vs_expected_phase_annotated.png",
    )
    output_json_path = os.path.join(session_dir, "measured_phase_check.json")

    try:
        completed_process = runner(
            command,
            cwd=working_directory,
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