"""Qt worker coordinating phase-capture persistence and checker execution."""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from qtpy.QtCore import QObject, Signal

from mwdriver.capture.phase_capture_process import run_phase_check
from mwdriver.capture.phase_capture_storage import (
    PhaseSaveResult,
    allocate_capture_session_dir,
    save_phase_capture_metadata,
    save_phase_pngs,
    save_raw_phase_stack,
    validate_phase_frame_stack,
)
from mwdriver.domain.phase_psi import DEFAULT_ALGORITHM
from mwdriver.domain.recalibration import DEFAULT_RECALIBRATION_CONFIG


MODULE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SAVE_DIR = str(MODULE_DIR / "captured_frames")
DEFAULT_PHASE_CHECK_SCRIPT = str(MODULE_DIR / "check_measured_vs_expected_phase.py")
DEFAULT_PHASE_CHECK_TIMEOUT_S = 180


class PhaseSaveWorker(QObject):
    status_ready = Signal(str)
    result_ready = Signal(object)
    finished = Signal(str)

    def __init__(
        self,
        frames,
        phase_times=None,
        frame_timestamps=None,
        roi_mask=None,
        save_dir=DEFAULT_SAVE_DIR,
        session_prefix="phase_capture",
        trigger_positions_ms=None,
        psi_algorithm_id=DEFAULT_ALGORITHM,
        phase_check_script=DEFAULT_PHASE_CHECK_SCRIPT,
        phase_check_timeout_s=DEFAULT_PHASE_CHECK_TIMEOUT_S,
        voltage_mapper=None,
        recalibration_config=DEFAULT_RECALIBRATION_CONFIG,
    ):
        super().__init__()
        self.frames = frames
        self.phase_times = phase_times
        self.frame_timestamps = frame_timestamps
        self.trigger_positions_ms = trigger_positions_ms
        self.psi_algorithm_id = psi_algorithm_id
        self.roi_mask = roi_mask
        self.save_dir = save_dir
        self.session_prefix = str(session_prefix)
        self.phase_check_script = phase_check_script
        self.phase_check_timeout_s = phase_check_timeout_s
        self.voltage_mapper = voltage_mapper
        self.recalibration_config = recalibration_config

    def run_automatic_phase_check(self, session_dir):
        session_dir = os.path.abspath(session_dir)
        command = [sys.executable, self.phase_check_script, session_dir]
        print("====================================")
        print("Starting automatic phase check")
        print("Command:", command)
        print("Capture folder:", session_dir)
        print("====================================")
        self.status_ready.emit(
            "Phase frames saved; running automatic phase check..."
        )
        result = run_phase_check(
            self.phase_check_script,
            session_dir,
            str(MODULE_DIR),
            self.phase_check_timeout_s,
            roi_mask=self.roi_mask,
        )
        if result.stdout:
            print("Automatic phase-check stdout:")
            print(result.stdout)
        if result.stderr:
            print("Automatic phase-check stderr:")
            print(result.stderr)
        print(result.message)
        self.status_ready.emit(result.message)
        return result

    def save_phase_vs_voltage_plot(
        self,
        session_dir,
        timestamp,
        measured_phase_deg,
    ):
        try:
            if (
                self.trigger_positions_ms is None
                or len(self.trigger_positions_ms) != self.recalibration_config.frame_count
            ):
                print(
                    "Phase-vs-voltage plot skipped: "
                    "trigger positions unavailable for this session."
                )
                return None
            if (
                measured_phase_deg is None
                or len(measured_phase_deg) != self.recalibration_config.frame_count
            ):
                print(
                    "Phase-vs-voltage plot skipped: "
                    "measured phase unavailable for this session."
                )
                return None
            if self.voltage_mapper is None:
                print("Phase-vs-voltage plot skipped: voltage mapper unavailable.")
                return None

            voltages = [
                self.voltage_mapper(position_ms)
                for position_ms in self.trigger_positions_ms
            ]
            figure = Figure(figsize=(7, 5))
            FigureCanvasAgg(figure)
            axis = figure.subplots()
            axis.plot(measured_phase_deg, voltages, "o-", color="tab:blue")
            for index, (voltage, phase) in enumerate(zip(voltages, measured_phase_deg)):
                axis.annotate(
                    f"{voltage:.2f} V",
                    (phase, voltage),
                    textcoords="offset points",
                    xytext=(6, 6 if index % 2 == 0 else -16),
                    ha="left",
                )
            axis.set_xlabel("Measured phase (deg)")
            axis.set_ylabel("Triangle wave voltage (V)")
            axis.set_title("Triangle wave voltage vs. measured phase")
            axis.grid(True, alpha=0.3)
            figure.tight_layout()
            plot_path = os.path.join(
                session_dir,
                f"phase_vs_voltage_{timestamp}.png",
            )
            figure.savefig(plot_path, dpi=150)
            print("Saved phase-vs-voltage plot:", plot_path)
            self.status_ready.emit(f"Saved phase-vs-voltage plot: {plot_path}")
            return plot_path
        except Exception as error:
            print("Phase-vs-voltage plot warning:", error)
            return None

    def run(self):
        session_dir = None
        raw_stack_path = None
        metadata_path = None
        try:
            frames = validate_phase_frame_stack(
                self.frames,
                self.psi_algorithm_id,
            )
            for name, values in (
                ("phase_times", self.phase_times),
                ("frame_timestamps", self.frame_timestamps),
            ):
                if values is not None and len(values) != len(frames):
                    raise ValueError(
                        f"Expected {len(frames)} {name}, got {len(values)}."
                    )

            os.makedirs(self.save_dir, exist_ok=True)
            print("====================================")
            print("Phase save worker started")
            print("Current working directory:", os.getcwd())
            print("Save directory:", self.save_dir)
            print("Number of phase frames:", len(frames))
            print("====================================")

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            session_dir = allocate_capture_session_dir(
                self.save_dir,
                self.session_prefix,
                timestamp,
            )
            print("Session save directory:", session_dir)

            if self.roi_mask is None:
                print("No active ellipse ROI; phase check will use its default region.")
            else:
                roi_mask = np.asarray(self.roi_mask, dtype=bool)
                if roi_mask.shape != frames.shape[1:]:
                    raise ValueError(
                        f"ROI mask shape {roi_mask.shape} does not match "
                        f"frame shape {frames.shape[1:]}"
                    )
                print(
                    "Current ellipse ROI mask supplied to automatic phase check; "
                    "not saved with capture."
                )

            raw_stack_path, stack = save_raw_phase_stack(
                session_dir,
                self.session_prefix,
                timestamp,
                frames,
            )
            print("Saved phase raw stack:", raw_stack_path)
            print("Raw stack shape:", stack.shape)
            print("Raw stack dtype:", stack.dtype)
            print("Raw stack min:", np.min(stack))
            print("Raw stack max:", np.max(stack))
            self.status_ready.emit(f"Saved phase raw stack: {raw_stack_path}")

            png_records = save_phase_pngs(session_dir, timestamp, frames)
            png_paths = [record["path"] for record in png_records]
            for phase_index, record in enumerate(png_records):
                print("Saved phase PNG:", record["path"])
                print(
                    f"Phase {phase_index}: raw min={record['raw_min']}, "
                    f"raw max={record['raw_max']}, png min={record['png_min']}, "
                    f"png max={record['png_max']}"
                )
                self.status_ready.emit(f"Saved phase {phase_index}: {record['path']}")

            metadata_path, _metadata = save_phase_capture_metadata(
                session_dir=session_dir,
                timestamp=timestamp,
                session_prefix=self.session_prefix,
                frames=frames,
                phase_times=self.phase_times,
                frame_timestamps=self.frame_timestamps,
                png_paths=png_paths,
                psi_algorithm_id=self.psi_algorithm_id,
            )
            print("Saved metadata:", metadata_path)
            self.status_ready.emit(f"Saved metadata: {metadata_path}")

            phase_check = self.run_automatic_phase_check(session_dir)
            voltage_plot_path = None
            if phase_check.success:
                try:
                    with open(phase_check.output_json_path, "r", encoding="utf-8") as file:
                        measured_phase_deg = json.load(file).get("measured_phase_deg")
                except Exception as error:
                    print("Read measured phase for plot warning:", error)
                    measured_phase_deg = None
                voltage_plot_path = self.save_phase_vs_voltage_plot(
                    session_dir,
                    timestamp,
                    measured_phase_deg,
                )

            if phase_check.success:
                finish_text = f"Phase capture saved and checked: 6 frames in {session_dir}"
            else:
                finish_text = (
                    f"Phase capture saved: 6 frames in {session_dir}; "
                    "automatic phase check did not complete successfully"
                )
            self.result_ready.emit(PhaseSaveResult(
                save_success=True,
                session_dir=session_dir,
                raw_stack_path=raw_stack_path,
                metadata_path=metadata_path,
                phase_check_success=phase_check.success,
                phase_check_json_path=phase_check.output_json_path,
                phase_check_figure_path=phase_check.output_figure_path,
                phase_voltage_plot_path=voltage_plot_path,
            ))
            self.finished.emit(finish_text)
        except Exception as error:
            finish_text = f"Phase save error: {error}"
            self.result_ready.emit(PhaseSaveResult(
                save_success=False,
                session_dir=session_dir,
                raw_stack_path=raw_stack_path,
                metadata_path=metadata_path,
                error=str(error),
            ))
            self.finished.emit(finish_text)
            print(finish_text)
