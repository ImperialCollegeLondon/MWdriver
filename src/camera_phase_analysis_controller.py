"""Phase-analysis session selection, subprocess launch and result loading."""

import os
import subprocess
import sys

import numpy as np
from qtpy.QtCore import QTimer

from phase_capture_storage import PhaseSaveResult


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(SCRIPT_DIR, "captured_frames")
PHASE_ANALYSIS_SCRIPT = os.path.join(SCRIPT_DIR, "phase_analysis_new.py")


class CameraPhaseAnalysisController:
    def find_latest_phase_analysis_session(self):
        latest_result = self.latest_phase_save_result
        if isinstance(latest_result, PhaseSaveResult) and latest_result.save_success:
            session_dir = latest_result.session_dir
            if (
                session_dir
                and os.path.basename(session_dir).startswith("phase_capture_")
                and os.path.isdir(session_dir)
                and any(
                    filename.endswith("_raw_stack.npy")
                    for filename in os.listdir(session_dir)
                )
            ):
                return os.path.abspath(session_dir)

        candidates = []
        if os.path.isdir(SAVE_DIR):
            for entry in os.scandir(SAVE_DIR):
                if not entry.is_dir() or not entry.name.startswith("phase_capture_"):
                    continue

                raw_stack_paths = [
                    os.path.join(entry.path, filename)
                    for filename in os.listdir(entry.path)
                    if filename.endswith("_raw_stack.npy")
                ]
                if raw_stack_paths:
                    candidates.append((
                        max(os.path.getmtime(path) for path in raw_stack_paths),
                        entry.path,
                    ))

        if not candidates:
            return None

        return os.path.abspath(max(candidates)[1])

    def run_latest_phase_analysis(self):
        session_dir = self.find_latest_phase_analysis_session()
        if session_dir is None:
            self.status_label.setText(
                "Status: No completed phase capture is available for analysis"
            )
            return

        self.run_phase_analysis_session(session_dir, "latest phase capture")

    def run_selected_phase_analysis(self):
        selected_layer = self.viewer.layers.selection.active
        if selected_layer not in self.phase_capture_time_series_layers:
            self.status_label.setText(
                "Status: Select a captured six-frame stack in the layer list first"
            )
            return

        session_dir = selected_layer.metadata.get("session_dir")
        raw_stack_path = selected_layer.metadata.get("raw_stack_path")
        if (
            not session_dir
            or not os.path.isdir(session_dir)
            or not raw_stack_path
            or not os.path.isfile(raw_stack_path)
        ):
            self.status_label.setText(
                "Status: Selected stack is still saving or has no saved session"
            )
            return

        self.run_phase_analysis_session(
            session_dir,
            f"selected stack {selected_layer.name}",
        )

    def run_phase_analysis_session(self, session_dir, description):
        if self.save_thread is not None:
            self.status_label.setText(
                "Status: Phase capture is still being saved; please wait"
            )
            return

        if not os.path.isfile(PHASE_ANALYSIS_SCRIPT):
            self.status_label.setText(
                f"Status: Phase analysis script not found: {PHASE_ANALYSIS_SCRIPT}"
            )
            return

        if (
            self.phase_analysis_process is not None
            and self.phase_analysis_process.poll() is None
        ):
            self.status_label.setText(
                "Status: Phase analysis viewer is already open"
            )
            return

        try:
            raw_stack_paths = sorted(
                os.path.join(session_dir, filename)
                for filename in os.listdir(session_dir)
                if filename.endswith("_raw_stack.npy")
            )
            if not raw_stack_paths:
                self.status_label.setText(
                    f"Status: No raw image stack found in {session_dir}"
                )
                return
            self.phase_analysis_source_stack_name = os.path.basename(
                raw_stack_paths[0]
            )

            self.phase_analysis_output_path = os.path.join(
                session_dir,
                "phase_analysis_results.npz",
            )
            if os.path.exists(self.phase_analysis_output_path):
                os.remove(self.phase_analysis_output_path)

            if self.view_phase_analysis_button is not None:
                self.view_phase_analysis_button.setEnabled(False)

            self.phase_analysis_process = subprocess.Popen(
                [
                    sys.executable,
                    PHASE_ANALYSIS_SCRIPT,
                    session_dir,
                    "--no-viewer",
                    "--export",
                    self.phase_analysis_output_path,
                ],
                cwd=SCRIPT_DIR,
            )
            self.status_label.setText(
                f"Status: Analyzing {description} in background: "
                f"{os.path.basename(session_dir)}"
            )
            print(f"Started phase analysis for {description}:", session_dir)
            QTimer.singleShot(200, self.check_phase_analysis_process)
        except Exception as error:
            self.phase_analysis_process = None
            self.status_label.setText(
                f"Status: Failed to start phase analysis: {error}"
            )

    def check_phase_analysis_process(self):
        process = self.phase_analysis_process
        if process is None:
            return

        return_code = process.poll()
        if return_code is None:
            QTimer.singleShot(200, self.check_phase_analysis_process)
            return

        self.phase_analysis_process = None
        if return_code != 0:
            self.status_label.setText(
                f"Status: Phase analysis failed with code {return_code}"
            )
            if self.view_phase_analysis_button is not None:
                self.view_phase_analysis_button.setEnabled(True)
            return

        output_path = self.phase_analysis_output_path
        if not output_path or not os.path.exists(output_path):
            self.status_label.setText(
                "Status: Phase analysis finished but produced no result file"
            )
            return

        try:
            with np.load(output_path) as result_file:
                self.phase_analysis_results = {
                    key: np.asarray(result_file[key])
                    for key in result_file.files
                }

            self.phase_analysis_view_active = True
            self.create_phase_analysis_controls(selected_key="wrapped")
            self.show_selected_phase_analysis_result(create_new_layer=True)
            self.view_phase_analysis_button.setText("Return to Camera Preview")
            self.status_label.setText(
                "Status: Phase analysis ready; use the controls below"
            )
            print("Phase analysis results loaded:", output_path)
        except Exception as error:
            self.status_label.setText(
                f"Status: Failed to load phase analysis results: {error}"
            )