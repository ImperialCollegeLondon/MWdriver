"""Phase-analysis session selection, subprocess launch and result loading."""

import os
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QFileDialog

from mwdriver.capture.phase_capture_storage import PhaseSaveResult
from mwdriver.domain.phase_psi import get_algorithm_id_from_metadata, get_algorithm
from mwdriver.paths import RESULTS_DIR
from mwdriver.ui.phase_analysis_controls import build_phase_analysis_result_options


SCRIPT_DIR = str(Path(__file__).resolve().parents[2])
SAVE_DIR = str(RESULTS_DIR)
PHASE_ANALYSIS_SCRIPT = os.path.join(SCRIPT_DIR, "phase_analysis_new.py")


class CameraPhaseAnalysisController:
    def reload_stack_from_directory(self):
        if self.phase_capture_active or self.recalibration_active:
            self.status_label.setText("Status: Wait for capture or calibration to finish before loading a stack")
            return
        if getattr(self, "phase_analysis_process", None) is not None:
            self.status_label.setText("Status: Wait for the current analysis to finish before loading a stack")
            return
        directory = QFileDialog.getExistingDirectory(
            self.viewer.window._qt_window, "Reload Stack", SAVE_DIR,
        )
        if not directory:
            return
        session_dir = Path(directory).resolve()
        stacks = sorted(session_dir.glob("*_raw_stack.npy"))
        if len(stacks) != 1:
            self.status_label.setText(
                "Status: Select a dataset directory containing exactly one *_raw_stack.npy file"
            )
            return
        try:
            self.load_saved_dataset(stacks[0])
        except Exception as error:
            self.status_label.setText(f"Status: Could not reload stack: {error}")
            return
        analysis_path = session_dir / "phase_analysis_results.npz"
        if analysis_path.is_file():
            try:
                self.load_saved_analysis(analysis_path)
            except Exception as error:
                self.status_label.setText(f"Status: Stack loaded, but saved analysis could not be loaded: {error}")
                return
            self.status_label.setText(f"Status: Reloaded stack and saved analysis from {session_dir.name}")
        else:
            self.status_label.setText(f"Status: Reloaded stack from {session_dir.name}; no saved analysis found")

    def apply_automatic_analysis_roi(self):
        vertices = self.phase_analysis_results.get("automatic_roi_vertices")
        if vertices is None:
            return False
        current_mask, _, _ = self.get_current_inside_mask()
        if current_mask is not None or (self.roi_layer is not None and len(self.roi_layer.data)):
            return False
        vertices = np.asarray(vertices, dtype=float)
        if vertices.shape != (4, 2) or not np.all(np.isfinite(vertices)):
            raise ValueError("Invalid automatic ROI vertices in analysis results.")
        if self.phase_analysis_results["roi_mask"].shape != (self.height, self.width):
            return False
        self.waiting_for_roi_finish = False
        if self.roi_layer is None or not any(layer is self.roi_layer for layer in self.viewer.layers):
            self.roi_layer = self.viewer.add_shapes(
                name="ROI", ndim=2, shape_type="ellipse", edge_width=3,
                edge_color="red", face_color=[1, 0, 0, 0.15],
            )
            self.roi_layer.events.data.connect(self.on_roi_data_changed)
        self.roi_layer.add(vertices / self.display_downsample, shape_type="ellipse")
        self.roi_layer.selected_data = {0}
        self.roi_layer.visible = True
        self.roi_layer.mode = "select"
        self.last_roi_count = 1
        self.refresh_roi_mask()
        return True

    def reload_analysis_from_file(self):
        if self.phase_capture_active or self.recalibration_active:
            self.status_label.setText("Status: Wait for capture or calibration to finish before loading analysis")
            return
        filename, _ = QFileDialog.getOpenFileName(
            self.viewer.window._qt_window,
            "Reload Analysis",
            SAVE_DIR,
            "Phase-analysis results (*.npz)",
        )
        if not filename:
            return
        try:
            self.load_saved_analysis(filename)
        except Exception as error:
            self.status_label.setText(f"Status: Could not reload analysis: {error}")

    def load_saved_analysis(self, filename):
        output_path = Path(filename).resolve()
        with np.load(output_path, allow_pickle=False) as archive:
            results = {key: np.asarray(archive[key]) for key in archive.files}
        options = build_phase_analysis_result_options(results)
        if not options:
            raise ValueError("The archive contains no recognised analysis images")
        for _, key in options:
            if key not in results:
                continue
            array = results[key]
            if array.dtype.kind not in "uif" or not (
                array.ndim == 2 or (array.ndim == 3 and array.shape[-1] in (3, 4))
            ) or array.size == 0:
                raise ValueError(f"Invalid analysis image: {key}")
        selected_key = "wrapped" if "wrapped" in results else options[0][1]
        self.phase_analysis_results = results
        self.phase_analysis_source_stack_name = output_path.parent.name
        self.phase_analysis_view_active = True
        self.create_phase_analysis_controls(selected_key=selected_key)
        self.show_selected_phase_analysis_result(create_new_layer=True)
        if self.view_phase_analysis_button is not None:
            self.view_phase_analysis_button.setText("Return to Camera Preview")
        self.sync_stream_with_active_layer()
        self.status_label.setText(f"Status: Reloaded analysis {output_path.parent.name}")

    def reload_dataset_from_file(self):
        if self.phase_capture_active or self.recalibration_active:
            self.status_label.setText("Status: Wait for capture or calibration to finish before loading a dataset")
            return
        filename, _ = QFileDialog.getOpenFileName(
            self.viewer.window._qt_window,
            "Reload Dataset",
            SAVE_DIR,
            "Camera raw stacks (*_raw_stack.npy);;NumPy arrays (*.npy)",
        )
        if not filename:
            return
        try:
            self.load_saved_dataset(filename)
        except Exception as error:
            self.status_label.setText(f"Status: Could not reload dataset: {error}")

    def load_saved_dataset(self, filename):
        raw_path = Path(filename).resolve()
        frames = np.load(raw_path, allow_pickle=False)
        if frames.ndim != 3 or frames.shape[0] != 6 or min(frames.shape[1:]) == 0:
            raise ValueError(f"Expected six-frame stack (6, height, width), got {frames.shape}")
        if frames.dtype.kind not in "uif":
            raise ValueError("Camera frames must contain real numeric intensities")
        metadata = {}
        metadata_path = raw_path.with_name(
            raw_path.name.removesuffix("_raw_stack.npy") + "_metadata.json"
        )
        if metadata_path.is_file():
            with metadata_path.open("r", encoding="utf-8") as file:
                metadata = json.load(file)
        algorithm_id = get_algorithm_id_from_metadata(metadata)
        layer = self.viewer.add_image(
            frames,
            name=f"Saved capture: {raw_path.parent.name}",
            axis_labels=("time", "y", "x"),
            colormap="gray",
            metadata={
                "session_dir": str(raw_path.parent),
                "raw_stack_path": str(raw_path),
                "algorithm_id": algorithm_id,
                "expected_phase_deg": list(get_algorithm(algorithm_id)["expected_phase_deg"]),
                "capture_metadata": metadata,
            },
        )
        self.phase_capture_time_series_layers.append(layer)
        self.phase_analysis_view_active = False
        self.apply_layer_visibility_plan("time_series", selected_layer=layer)
        self.viewer.layers.selection.active = layer
        if self.view_phase_analysis_button is not None:
            self.view_phase_analysis_button.setText("View Analysis Results")
        self.sync_stream_with_active_layer()
        self.viewer.reset_view()
        self.status_label.setText(f"Status: Reloaded dataset {raw_path.parent.name}")
        return layer

    def find_latest_phase_analysis_session(self):
        latest_result = self.latest_phase_save_result
        if isinstance(latest_result, PhaseSaveResult) and latest_result.save_success:
            session_dir = latest_result.session_dir
            if (
                session_dir
                and os.path.basename(session_dir).startswith(("measure_", "phase_capture_"))
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
                if not entry.is_dir() or not entry.name.startswith(("measure_", "phase_capture_")):
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

            analysis_command = [
                sys.executable,
                PHASE_ANALYSIS_SCRIPT,
                session_dir,
                "--no-viewer",
                "--export",
                self.phase_analysis_output_path,
            ]
            current_inside_mask, roi_type, roi_info = self.get_current_inside_mask()
            self.phase_analysis_roi_mask_path = None
            if current_inside_mask is not None:
                mask_file = tempfile.NamedTemporaryFile(
                    prefix="mwdriver_phase_roi_",
                    suffix=".npy",
                    delete=False,
                )
                self.phase_analysis_roi_mask_path = mask_file.name
                mask_file.close()
                np.save(self.phase_analysis_roi_mask_path, current_inside_mask)
                analysis_command.extend([
                    "--roi-mask",
                    self.phase_analysis_roi_mask_path,
                    "--pupil-vertices",
                    json.dumps(roi_info["raw_coordinates"]["vertices"]),
                ])

            self.phase_analysis_process = subprocess.Popen(
                analysis_command,
                cwd=SCRIPT_DIR,
            )
            self.status_label.setText(
                f"Status: Analyzing {description} in background: "
                f"{os.path.basename(session_dir)}"
            )
            print(f"Started phase analysis for {description}:", session_dir)
            QTimer.singleShot(200, self.check_phase_analysis_process)
        except Exception as error:
            roi_mask_path = getattr(self, "phase_analysis_roi_mask_path", None)
            if roi_mask_path and os.path.exists(roi_mask_path):
                os.remove(roi_mask_path)
            self.phase_analysis_roi_mask_path = None
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
        roi_mask_path = getattr(self, "phase_analysis_roi_mask_path", None)
        if roi_mask_path and os.path.exists(roi_mask_path):
            os.remove(roi_mask_path)
        self.phase_analysis_roi_mask_path = None
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

            CameraPhaseAnalysisController.apply_automatic_analysis_roi(self)
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