import ast
import json
import sys
import subprocess
import tempfile
import unittest
import time
import ast
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import camera
import phase_capture_worker
from phase_capture_controller import PhaseCaptureController
from application_bootstrap import run_application
import check_measured_vs_expected_phase as phase_check
import phase_analysis_new
from phase_capture_process import PhaseCheckResult, run_phase_check
from phase_capture_storage import (
    PhaseSaveResult,
    allocate_capture_session_dir,
    save_roi_snapshots,
)
from phase_psi import SIX_BUCKET_60_DEG
from recalibration import (
    DEFAULT_RECALIBRATION_CONFIG,
    compute_iterative_refined_positions_ms,
    decide_verification_capture_action,
    select_best_recalibration_candidate,
)
from roi_controls import create_roi_menu
from profile_controls import create_profile_controls
from phase_analysis_controls import (
    build_analysis_layer_presentation,
    build_phase_analysis_result_options,
    choose_analysis_layer_operation,
    create_phase_analysis_controls,
)
from qtpy.QtWidgets import QApplication


class LabelStub:
    def setText(self, value):
        self.value = value


class FakeSerial:
    def __init__(self):
        self.writes = []
        self.flushed = False

    def write(self, value):
        self.writes.append(value)

    def flush(self):
        self.flushed = True


class FakeSignal:
    def __init__(self):
        self.callback = None

    def connect(self, callback):
        self.callback = callback


class CameraContractTests(unittest.TestCase):
    def test_phase_save_worker_rejects_wrong_frame_count_before_allocating_session(self):
        frames = [np.ones((4, 5), dtype=np.uint16) for _ in range(5)]
        with tempfile.TemporaryDirectory() as temporary_directory:
            save_dir = Path(temporary_directory, "captures")
            worker = camera.PhaseSaveWorker(
                frames=frames,
                save_dir=str(save_dir),
            )
            results = []
            worker.result_ready.connect(results.append)

            worker.run()

            result = results[0]
            self.assertFalse(result.save_success)
            self.assertIn("Expected 6 phase frames, got 5", result.error)
            self.assertIsNone(result.session_dir)
            self.assertFalse(save_dir.exists())

    def test_phase_save_worker_rejects_malformed_frame_dimensions_before_allocating_session(self):
        malformed_stacks = {
            "non-2D frames": [np.ones((4, 5, 1), dtype=np.uint16) for _ in range(6)],
            "inconsistent dimensions": [
                np.ones((4, 5), dtype=np.uint16) for _ in range(5)
            ] + [np.ones((3, 5), dtype=np.uint16)],
        }

        for case_name, frames in malformed_stacks.items():
            with self.subTest(case=case_name):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    save_dir = Path(temporary_directory, "captures")
                    worker = camera.PhaseSaveWorker(
                        frames=frames,
                        save_dir=str(save_dir),
                    )
                    results = []
                    worker.result_ready.connect(results.append)

                    worker.run()

                    result = results[0]
                    self.assertFalse(result.save_success)
                    self.assertIsNone(result.session_dir)
                    self.assertFalse(save_dir.exists())
                    self.assertRegex(
                        result.error,
                        "Expected phase frame stack|equally shaped 2D images",
                    )

    def test_phase_save_worker_rejects_misaligned_timing_metadata_before_session(self):
        frames = [np.ones((4, 5), dtype=np.uint16) for _ in range(6)]
        cases = [
            ("phase_times", [0.0] * 5, None),
            ("frame_timestamps", None, [{"phase_index": index} for index in range(5)]),
        ]

        for field_name, phase_times, frame_timestamps in cases:
            with self.subTest(field=field_name):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    save_dir = Path(temporary_directory, "captures")
                    worker = camera.PhaseSaveWorker(
                        frames=frames,
                        phase_times=phase_times,
                        frame_timestamps=frame_timestamps,
                        save_dir=str(save_dir),
                    )
                    results = []
                    worker.result_ready.connect(results.append)

                    worker.run()

                    result = results[0]
                    self.assertFalse(result.save_success)
                    self.assertIn(field_name, result.error)
                    self.assertIsNone(result.session_dir)
                    self.assertFalse(save_dir.exists())

    def test_controller_applies_selected_layer_visibility_plan(self):
        camera_layer = SimpleNamespace(visible=True)
        roi_layer = SimpleNamespace(visible=True)
        analysis_layers = [SimpleNamespace(visible=True), SimpleNamespace(visible=True)]
        time_series_layers = [SimpleNamespace(visible=True), SimpleNamespace(visible=True)]
        viewer = SimpleNamespace(
            layer=camera_layer,
            roi_layer=roi_layer,
            phase_analysis_layer_records=[
                {"layer": layer} for layer in analysis_layers
            ],
            phase_capture_time_series_layers=time_series_layers,
        )

        camera.ThorlabsCameraViewer.apply_layer_visibility_plan(
            viewer,
            "analysis",
            selected_layer=analysis_layers[1],
        )

        self.assertFalse(camera_layer.visible)
        self.assertFalse(roi_layer.visible)
        self.assertEqual(
            [layer.visible for layer in analysis_layers],
            [False, True],
        )
        self.assertEqual(
            [layer.visible for layer in time_series_layers],
            [False, False],
        )

    def test_analysis_layer_operation_covers_add_replace_and_update(self):
        cases = [
            (
                {
                    "create_new_layer": True,
                    "has_current_layer": True,
                    "has_current_record": True,
                    "current_layer_is_rgb": False,
                    "result_is_rgb": False,
                },
                "add",
            ),
            (
                {
                    "create_new_layer": False,
                    "has_current_layer": False,
                    "has_current_record": False,
                    "current_layer_is_rgb": False,
                    "result_is_rgb": False,
                },
                "add",
            ),
            (
                {
                    "create_new_layer": False,
                    "has_current_layer": True,
                    "has_current_record": False,
                    "current_layer_is_rgb": False,
                    "result_is_rgb": False,
                },
                "replace",
            ),
            (
                {
                    "create_new_layer": False,
                    "has_current_layer": True,
                    "has_current_record": True,
                    "current_layer_is_rgb": False,
                    "result_is_rgb": True,
                },
                "replace",
            ),
            (
                {
                    "create_new_layer": False,
                    "has_current_layer": True,
                    "has_current_record": True,
                    "current_layer_is_rgb": True,
                    "result_is_rgb": True,
                },
                "update",
            ),
        ]

        for options, expected in cases:
            with self.subTest(options=options):
                self.assertEqual(
                    choose_analysis_layer_operation(**options),
                    expected,
                )

    def test_phase_analysis_result_options_filter_and_format_metrics(self):
        options = build_phase_analysis_result_options({
            "wrapped": np.zeros((2, 2)),
            "wavefront_aberration_nm": np.zeros((2, 2)),
            "wavefront_aberration_rms_nm": np.float32(12.345),
            "wavefront_aberration_pv_nm": np.float32(67.891),
            "zernike_after_plot_rgb": np.zeros((2, 2, 3)),
        })

        self.assertEqual(
            options,
            [
                ("Wrapped phase", "wrapped"),
                (
                    "Defocus removed: RMS 12.35 nm, PV 67.89 nm",
                    "wavefront_aberration_nm",
                ),
                (
                    "Zernike J1-J15 after piston/tilt removal",
                    "zernike_after_plot_rgb",
                ),
            ],
        )

    def test_phase_analysis_result_options_use_legacy_metric_label(self):
        options = build_phase_analysis_result_options({
            "wavefront_aberration_nm": np.zeros((2, 2)),
            "wavefront_aberration_rms_nm": 12.345,
        })

        self.assertEqual(
            options,
            [("Defocus removed (nm OPD)", "wavefront_aberration_nm")],
        )

    def test_analysis_layer_presentation_selects_name_rgb_cmap_and_limits(self):
        phase = build_analysis_layer_presentation(
            "phase_0",
            np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            "sample",
            "Phase 0",
        )
        self.assertEqual(phase["name"], "Analysis [sample]: Phase 0")
        self.assertFalse(phase["is_rgb"])
        self.assertEqual(phase["colormap"], "gray")
        self.assertEqual(phase["contrast_limits"], (1.0, 4.0))

        wrapped = build_analysis_layer_presentation(
            "wrapped", np.asarray([[-np.pi, np.pi]]), None, "Wrapped"
        )
        self.assertEqual(wrapped["name"], "Analysis [unknown stack]: Wrapped")
        self.assertEqual(wrapped["colormap"], "twilight")

        rgb = build_analysis_layer_presentation(
            "zernike_plot", np.zeros((4, 5, 3)), "sample", "Zernike"
        )
        self.assertTrue(rgb["is_rgb"])
        self.assertIsNone(rgb["contrast_limits"])

        constant = build_analysis_layer_presentation(
            "unwrapped", np.ones((2, 2)), "sample", "Unwrapped"
        )
        self.assertIsNone(constant["contrast_limits"])

    def test_application_bootstrap_installs_cleanup_and_runs_event_loop(self):
        class FakeEvent:
            accepted = False

            def accept(self):
                self.accepted = True

        class FakeViewer:
            def __init__(self):
                self.closed = False
                self.viewer = SimpleNamespace(
                    window=SimpleNamespace(_qt_window=SimpleNamespace())
                )

            def close(self):
                self.closed = True

        viewer = FakeViewer()
        loop_calls = []
        result = run_application(lambda: viewer, lambda: loop_calls.append(True))

        self.assertIs(result, viewer)
        self.assertEqual(loop_calls, [True])
        event = FakeEvent()
        viewer.viewer.window._qt_window.closeEvent(event)
        self.assertTrue(viewer.closed)
        self.assertTrue(event.accepted)

    def test_application_bootstrap_cleans_up_if_event_loop_returns(self):
        viewer = SimpleNamespace(
            closed_count=0,
            viewer=SimpleNamespace(
                window=SimpleNamespace(_qt_window=SimpleNamespace())
            ),
        )
        viewer.close = lambda: setattr(viewer, "closed_count", viewer.closed_count + 1)

        run_application(lambda: viewer, lambda: None)

        self.assertEqual(viewer.closed_count, 1)

    def test_application_bootstrap_cleanup_is_idempotent_after_close_event(self):
        class FakeEvent:
            def accept(self):
                self.accepted = True

        viewer = SimpleNamespace(
            closed_count=0,
            viewer=SimpleNamespace(
                window=SimpleNamespace(_qt_window=SimpleNamespace())
            ),
        )
        viewer.close = lambda: setattr(viewer, "closed_count", viewer.closed_count + 1)

        run_application(
            lambda: viewer,
            lambda: viewer.viewer.window._qt_window.closeEvent(FakeEvent()),
        )

        self.assertEqual(viewer.closed_count, 1)

    def test_profile_controls_widget_has_expected_buttons(self):
        application = QApplication.instance() or QApplication([])
        widget, profile_button, clear_button = create_profile_controls()
        self.addCleanup(widget.deleteLater)
        self.assertIsNotNone(application)
        self.assertEqual(widget.objectName(), "embedded_profile_controls")
        self.assertEqual(profile_button.text(), "Intensity Profile")
        self.assertEqual(clear_button.text(), "Clear Profile")
        self.assertEqual(profile_button.minimumHeight(), 28)
        self.assertEqual(clear_button.minimumHeight(), 28)

    def test_phase_analysis_controls_widget_connects_view_callbacks(self):
        application = QApplication.instance() or QApplication([])
        calls = []
        widget, title, selector, view_button = create_phase_analysis_controls(
            lambda index: calls.append(("selection", index)),
            lambda: calls.append(("view",)),
        )
        self.addCleanup(widget.deleteLater)

        self.assertIsNotNone(application)
        self.assertEqual(widget.objectName(), "embedded_phase_analysis_controls")
        self.assertEqual(view_button.text(), "View Analysis Results")
        title.setText("Review: capture")
        selector.addItem("Wrapped phase", "wrapped")
        selector.setCurrentIndex(0)
        view_button.click()

        self.assertEqual(title.text(), "Review: capture")
        self.assertEqual(selector.currentData(), "wrapped")
        self.assertIn(("selection", 0), calls)
        self.assertIn(("view",), calls)

    def test_roi_menu_actions_preserve_order_labels_and_callbacks(self):
        class FakeAction:
            def __init__(self, label, parent):
                self.label = label
                self.parent = parent
                self.triggered = FakeSignal()
                self.status_tip = None

            def setStatusTip(self, text):
                self.status_tip = text

        class FakeMenu:
            def __init__(self):
                self.items = []

            def addAction(self, action):
                self.items.append(action)

            def addSeparator(self):
                self.items.append(None)

        class FakeMenuBar:
            def __init__(self):
                self.menu = FakeMenu()
                self.title = None

            def addMenu(self, title):
                self.title = title
                return self.menu

        calls = []
        callback_names = (
            "start_rectangle_roi",
            "start_circle_roi",
            "mask_outside_area",
            "mask_inside_area",
            "save_roi",
            "reload_roi",
            "clear_roi",
        )
        callbacks = {
            name: lambda name=name: calls.append(name)
            for name in callback_names
        }
        menu_bar = FakeMenuBar()
        parent = object()

        menu, actions = create_roi_menu(menu_bar, parent, callbacks, FakeAction)

        self.assertIs(menu, menu_bar.menu)
        self.assertEqual(menu_bar.title, "ROI Tools")
        self.assertEqual(
            [item.label if item else None for item in menu.items],
            [
                "Rect ROI",
                "Circle ROI",
                None,
                "Mask Outside Area",
                "Mask Inside Area",
                None,
                "Save ROI",
                "Reload ROI",
                "Clear ROI",
            ],
        )
        self.assertEqual(set(actions), {
            "rect_roi_action",
            "circle_roi_action",
            "mask_outside_action",
            "mask_inside_action",
            "save_roi_action",
            "reload_roi_action",
            "clear_roi_action",
        })
        for action in actions.values():
            action.triggered.callback()
        self.assertEqual(calls, list(callback_names))

    def test_offline_analysis_import_is_safe_and_exposes_callable(self):
        self.assertTrue(callable(phase_analysis_new.analyze_capture))
        self.assertFalse(hasattr(phase_analysis_new, "session_dir"))

    def test_analysis_export_writer_creates_parent_and_preserves_arrays(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            export_path = Path(temporary_directory) / "nested" / "results.npz"
            output_path = phase_analysis_new.write_analysis_export(
                export_path,
                {
                    "wrapped": np.asarray([[0.1, 0.2]], dtype=np.float32),
                    "algorithm_id": np.asarray("six_frame_90deg"),
                },
            )
            self.assertEqual(output_path, export_path.resolve())
            self.assertTrue(output_path.is_file())
            with np.load(output_path) as results:
                self.assertEqual(results["wrapped"].dtype, np.float32)
                np.testing.assert_allclose(results["wrapped"], [[0.1, 0.2]])
                self.assertEqual(results["algorithm_id"].item(), "six_frame_90deg")

    def test_analysis_cli_main_preserves_export_and_no_viewer_options(self):
        session_dir = Path("capture-session")
        stack_path = session_dir / "capture_raw_stack.npy"
        expected_result = object()

        with (
            patch.object(
                phase_analysis_new,
                "find_capture_input",
                return_value=(session_dir, stack_path),
            ),
            patch.object(
                phase_analysis_new,
                "analyze_capture",
                return_value=expected_result,
            ) as analyze,
        ):
            result = phase_analysis_new.main([
                str(session_dir),
                "--no-viewer",
                "--export",
                "results.npz",
            ])

        self.assertIs(result, expected_result)
        analyze.assert_called_once_with(
            session_dir,
            stack_path,
            export_path=Path("results.npz").resolve(),
            show_viewer=False,
        )

    def test_analysis_cli_options_allow_latest_session_mode(self):
        expected_result = object()
        with (
            patch.object(
                phase_analysis_new,
                "find_capture_input",
                return_value=(Path("latest-session"), Path("latest-stack.npy")),
            ) as find_input,
            patch.object(
                phase_analysis_new,
                "analyze_capture",
                return_value=expected_result,
            ) as analyze,
        ):
            result = phase_analysis_new.main(["--no-viewer"])

        self.assertIs(result, expected_result)
        find_input.assert_called_once_with(None)
        analyze.assert_called_once_with(
            Path("latest-session"),
            Path("latest-stack.npy"),
            export_path=None,
            show_viewer=False,
        )

    def test_analysis_cli_latest_mode_selects_newest_capture(self):
        expected_result = object()
        with tempfile.TemporaryDirectory() as temporary_directory:
            capture_root = Path(temporary_directory)
            old_session = capture_root / "phase_capture_old"
            latest_session = capture_root / "phase_capture_latest"
            old_session.mkdir()
            latest_session.mkdir()
            old_stack = old_session / "old_raw_stack.npy"
            latest_stack = latest_session / "latest_raw_stack.npy"
            old_stack.touch()
            latest_stack.touch()
            with patch.object(phase_analysis_new, "CAPTURE_DIR", capture_root):
                with patch.object(
                    phase_analysis_new,
                    "analyze_capture",
                    return_value=expected_result,
                ) as analyze:
                    result = phase_analysis_new.main(["--no-viewer"])

        self.assertIs(result, expected_result)
        analyze.assert_called_once_with(
            latest_session,
            latest_stack,
            export_path=None,
            show_viewer=False,
        )

    def test_capture_directory_allocator_skips_existing_suffixes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            first = allocate_capture_session_dir(
                temporary_directory,
                "phase_capture",
                "20261014_120300",
            )
            Path(temporary_directory, "phase_capture_20261014_120300_01").mkdir()
            Path(temporary_directory, "phase_capture_20261014_120300_02").write_text(
                "occupied",
                encoding="utf-8",
            )
            next_path = allocate_capture_session_dir(
                temporary_directory,
                "phase_capture",
                "20261014_120300",
            )

            self.assertEqual(Path(first).name, "phase_capture_20261014_120300")
            self.assertEqual(Path(next_path).name, "phase_capture_20261014_120300_03")
            self.assertTrue(Path(next_path).is_dir())

    def test_manual_capture_worker_preserves_stack_and_numbered_pngs(self):
        timestamp = "20261014_120500"
        frames = [np.full((4, 5), index + 1, dtype=np.uint16) for index in range(6)]
        with tempfile.TemporaryDirectory() as temporary_directory:
            worker = camera.FrameSaveWorker(
                frames,
                save_dir=temporary_directory,
                prefix="manual_capture",
            )
            finished = []
            statuses = []
            worker.finished.connect(finished.append)
            worker.status_ready.connect(statuses.append)

            with patch.object(camera.time, "strftime", return_value=timestamp):
                worker.run()

            stack = np.load(
                Path(temporary_directory)
                / f"manual_capture_{timestamp}_raw_stack.npy"
            )
            self.assertEqual(stack.shape, (6, 4, 5))
            self.assertTrue(
                (Path(temporary_directory) / f"manual_capture_{timestamp}_01.png").is_file()
            )
            self.assertTrue(
                (Path(temporary_directory) / f"manual_capture_{timestamp}_06.png").is_file()
            )
            self.assertEqual(finished, ["manual_capture saved: 6 frames"])
            self.assertEqual(len(statuses), 7)

    @staticmethod
    def load_controller_method(method_name, namespace):
        if method_name == "wait_for_trigger_update_ack":
            camera_path = ROOT / "src" / "camera_recalibration_controller.py"
            class_name = "CameraRecalibrationController"
        else:
            camera_path = ROOT / "src" / "camera.py"
            class_name = "ThorlabsCameraViewer"

        camera_tree = ast.parse(camera_path.read_text(encoding="utf-8"))
        viewer_class = next(
            node for node in camera_tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == class_name
        )
        method_node = next(
            node for node in viewer_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == method_name
        )
        method_module = ast.Module(body=[method_node], type_ignores=[])
        exec(compile(method_module, str(camera_path), "exec"), namespace)
        return namespace[method_name]

    @staticmethod
    def load_save_result_handler():
        return PhaseCaptureController.on_phase_save_result

    def test_phase_frames_are_assigned_to_requested_slots_and_copied(self):
        viewer = SimpleNamespace(
            phase_capture_active=True,
            phase_capture_frames=[None] * 6,
            phase_capture_times=[None] * 6,
            phase_capture_frame_timestamps=[None] * 6,
            phase_capture_timestamp_clock_hz=None,
            phase_capture_start_time=None,
            phase_capture_count=0,
            pending_phase_indices=[],
            unassigned_triggered_frames=[],
            cam=SimpleNamespace(get_timestamp_clock_frequency=lambda: 1_000_000_000),
            phase_label=LabelStub(),
            status_label=LabelStub(),
            finish_phase_capture_session=lambda: self.fail(
                "A partial phase assignment must not finish the session"
            ),
        )
        first_frame = np.full((2, 2), 17, dtype=np.uint16)
        second_frame = np.full((2, 2), 29, dtype=np.uint16)

        self.assertTrue(camera.ThorlabsCameraViewer.assign_phase_frame(
            viewer, 4, first_frame, frame_meta={"framestamp": 5}
        ))
        first_frame.fill(0)
        self.assertTrue(camera.ThorlabsCameraViewer.assign_phase_frame(
            viewer, 1, second_frame, frame_meta={"framestamp": 2}
        ))
        self.assertFalse(camera.ThorlabsCameraViewer.assign_phase_frame(
            viewer, 4, second_frame
        ))
        self.assertFalse(camera.ThorlabsCameraViewer.assign_phase_frame(
            viewer, 6, second_frame
        ))

        np.testing.assert_array_equal(
            viewer.phase_capture_frames[4], np.full((2, 2), 17, dtype=np.uint16)
        )
        np.testing.assert_array_equal(
            viewer.phase_capture_frames[1], np.full((2, 2), 29, dtype=np.uint16)
        )
        self.assertEqual(viewer.phase_capture_count, 2)
        self.assertEqual(viewer.phase_capture_frame_timestamps[4]["phase_index"], 4)

    def test_native_frame_assignment_uses_acquisition_index_and_rejects_stale_run(self):
        def make_viewer():
            return SimpleNamespace(
                phase_capture_active=True,
                current_camera_mode="external",
                phase_capture_count=0,
                cam=SimpleNamespace(acquisition_id=8),
                assigned=[],
                assign_phase_frame=lambda **kwargs: viewer.assigned.append(kwargs),
            )

        viewer = make_viewer()
        camera.ThorlabsCameraViewer.handle_triggered_phase_frame(
            viewer,
            np.zeros((1, 1)),
            {
                "acquisition_index": 3,
                "acquisition_id": 8,
                "trigger_mode": "ext",
            },
        )
        self.assertEqual(viewer.assigned[0]["phase_index"], 3)

        viewer.assigned.clear()
        camera.ThorlabsCameraViewer.handle_triggered_phase_frame(
            viewer,
            np.zeros((1, 1)),
            {
                "acquisition_index": 2,
                "acquisition_id": 7,
                "trigger_mode": "ext",
            },
        )
        self.assertEqual(viewer.assigned, [])

    def test_pico_worker_queues_and_writes_exact_command_payloads(self):
        worker = camera.PicoSerialWorker(port="COM4")
        worker.request_set_trigger_positions_ms([100.2, 190.5, 280.8, 369, 459, 549])
        worker.request_start_phase_from_gui()

        serial_port = FakeSerial()
        worker.ser = serial_port
        worker.process_pending_commands()

        self.assertEqual(
            serial_port.writes,
            [
                b"SET_TRIGGER_POSITIONS_MS:100,190,281,369,459,549\n",
                b"PC_START_PHASE\n",
            ],
        )
        self.assertTrue(serial_port.flushed)

    def test_pico_worker_dispatches_capture_lifecycle_messages(self):
        worker = camera.PicoSerialWorker(port="COM4")
        started = []
        requested_phases = []
        completed = []
        manual_captures = []
        messages = []
        worker.phase_session_started.connect(lambda: started.append(True))
        worker.phase_requested.connect(requested_phases.append)
        worker.phase_session_done.connect(lambda: completed.append(True))
        worker.capture_requested.connect(lambda: manual_captures.append(True))
        worker.message_ready.connect(messages.append)

        for line in (
            "TRIANGLE_START",
            "CAPTURE_PHASE_0",
            "CAPTURE_PHASE_5",
            "CAPTURE_PHASE_6",
            "CAPTURE_PHASE_invalid",
            "TRIANGLE_DONE",
            "START_CAPTURE",
        ):
            worker.handle_serial_line(line)

        self.assertEqual(started, [True])
        self.assertEqual(requested_phases, [0, 5])
        self.assertEqual(completed, [True])
        self.assertEqual(manual_captures, [True])
        self.assertTrue(any("out of range: 6" in message for message in messages))
        self.assertTrue(any("phase parse error" in message for message in messages))

    def test_pico_ack_parser_records_positions_and_rejections(self):
        viewer = SimpleNamespace(
            last_trigger_update_ack_positions_ms=None,
            last_trigger_update_error_text=None,
        )
        parse_message = self.load_controller_method(
            "print_pico_message",
            {"ast": ast},
        )

        parse_message(viewer, "Pico says: TRIGGER_POSITIONS_UPDATED_MS: [100, 164, 222, 276, 330, 384]")
        self.assertEqual(
            viewer.last_trigger_update_ack_positions_ms,
            [100, 164, 222, 276, 330, 384],
        )
        self.assertIsNone(viewer.last_trigger_update_error_text)

        parse_message(viewer, "Pico says: TRIGGER_POSITIONS_UPDATE_ERROR: invalid count")
        self.assertIn("invalid count", viewer.last_trigger_update_error_text)

    def test_trigger_ack_wait_accepts_matching_rounded_positions(self):
        class FakeApplication:
            @staticmethod
            def processEvents():
                pass

        viewer = SimpleNamespace(
            last_trigger_update_ack_positions_ms=[100, 164, 222, 276, 330, 384],
            last_trigger_update_error_text=None,
        )
        wait_for_ack = self.load_controller_method(
            "wait_for_trigger_update_ack",
            {
                "time": time,
                "QCoreApplication": FakeApplication,
                "RECALIBRATION_TRIGGER_UPDATE_ACK_TIMEOUT_S": 1.5,
            },
        )

        self.assertIsNone(wait_for_ack(
            viewer,
            [100.2, 163.7, 222, 276, 330, 384],
        ))

    def test_trigger_ack_wait_reports_rejection_mismatch_and_timeout(self):
        class FakeApplication:
            @staticmethod
            def processEvents():
                pass

        wait_for_ack = self.load_controller_method(
            "wait_for_trigger_update_ack",
            {
                "time": time,
                "QCoreApplication": FakeApplication,
                "RECALIBRATION_TRIGGER_UPDATE_ACK_TIMEOUT_S": 1.5,
            },
        )
        expected = [100, 164, 222, 276, 330, 384]

        with self.assertRaisesRegex(RuntimeError, "Pico rejected trigger update"):
            wait_for_ack(SimpleNamespace(
                last_trigger_update_ack_positions_ms=None,
                last_trigger_update_error_text="TRIGGER_POSITIONS_UPDATE_ERROR: invalid",
            ), expected)

        with self.assertRaisesRegex(RuntimeError, "ACK mismatch"):
            wait_for_ack(SimpleNamespace(
                last_trigger_update_ack_positions_ms=[100, 164, 222, 276, 330, 385],
                last_trigger_update_error_text=None,
            ), expected)

        class ExpiredClock:
            calls = 0

            @classmethod
            def monotonic(cls):
                cls.calls += 1
                return 0.0 if cls.calls == 1 else 2.0

            @staticmethod
            def sleep(_duration):
                pass

        timeout_wait_for_ack = self.load_controller_method(
            "wait_for_trigger_update_ack",
            {
                "time": ExpiredClock,
                "QCoreApplication": FakeApplication,
                "RECALIBRATION_TRIGGER_UPDATE_ACK_TIMEOUT_S": 1.5,
            },
        )
        with self.assertRaisesRegex(TimeoutError, "Timed out waiting for Pico"):
            timeout_wait_for_ack(
                SimpleNamespace(
                    last_trigger_update_ack_positions_ms=None,
                    last_trigger_update_error_text=None,
                ),
                expected,
            )

    def test_phase_save_worker_preserves_paths_filenames_and_metadata(self):
        timestamp = "20261014_120000"
        frames = [np.full((4, 5), index + 1, dtype=np.uint16) for index in range(6)]

        with tempfile.TemporaryDirectory() as temporary_directory:
            save_root = Path(temporary_directory)

            def complete_phase_check(session_dir):
                check_path = Path(session_dir) / "measured_phase_check.json"
                check_path.write_text(
                    json.dumps({"measured_phase_deg": [0, 60, 120, 180, 240, 300]}),
                    encoding="utf-8",
                )
                (Path(session_dir) / "measured_vs_expected_phase_annotated.png").touch()
                return PhaseCheckResult(
                    True,
                    "checker passed",
                    0,
                    str(Path(session_dir) / "measured_vs_expected_phase_annotated.png"),
                    str(check_path),
                )

            session_dirs = []
            for _ in range(2):
                worker = camera.PhaseSaveWorker(
                    frames=frames,
                    phase_times=[index * 0.1 for index in range(6)],
                    save_dir=str(save_root),
                    session_prefix="phase_calibration",
                    roi_mask=np.ones((4, 5), dtype=bool),
                    roi_params={"shape": "rectangle", "valid_pixels": 20},
                    psi_algorithm_id=SIX_BUCKET_60_DEG,
                )
                worker.run_automatic_phase_check = complete_phase_check
                results = []
                worker.result_ready.connect(results.append)

                with patch.object(camera.time, "strftime", return_value=timestamp):
                    worker.run()

                self.assertTrue(results[0].save_success)
                session_dirs.append(Path(results[0].session_dir))

            self.assertEqual(
                [path.name for path in session_dirs],
                [
                    f"phase_calibration_{timestamp}",
                    f"phase_calibration_{timestamp}_01",
                ],
            )
            for session_dir in session_dirs:
                self.assertTrue(
                    (
                        session_dir
                        / f"phase_calibration_{timestamp}_raw_stack.npy"
                    ).is_file()
                )
                self.assertTrue(
                    (session_dir / f"phase_0_{timestamp}.png").is_file()
                )
                self.assertTrue((session_dir / "roi_mask.npy").is_file())
                saved_roi_params = json.loads(
                    (session_dir / "roi_params.json").read_text(encoding="utf-8")
                )
                self.assertEqual(saved_roi_params["valid_pixels"], 20)
                metadata = json.loads(
                    (
                        session_dir
                        / f"phase_calibration_{timestamp}_metadata.json"
                    ).read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(metadata["phase_count"], 6)
                self.assertIsNone(metadata["analysis_notebook_path"])
                self.assertTrue(metadata["roi"]["roi_available"])
                self.assertEqual(
                    Path(metadata["roi"]["roi_mask_path"]).name,
                    "roi_mask.npy",
                )
                self.assertEqual(
                    metadata["phase_capture_model"]["algorithm_id"],
                    SIX_BUCKET_60_DEG,
                )
                self.assertEqual(
                    metadata["phase_capture_model"]["phase_order"],
                    "phase_0 to phase_5",
                )

    def test_phase_save_worker_reports_checker_failure_without_losing_capture(self):
        frames = [np.full((4, 5), index + 1, dtype=np.uint16) for index in range(6)]
        with tempfile.TemporaryDirectory() as temporary_directory:
            worker = camera.PhaseSaveWorker(
                frames=frames,
                save_dir=temporary_directory,
                session_prefix="phase_calibration",
            )
            worker.run_automatic_phase_check = lambda _session_dir: PhaseCheckResult(
                False,
                "simulated checker failure",
                1,
            )
            results = []
            worker.result_ready.connect(results.append)

            with patch.object(camera.time, "strftime", return_value="20261014_120200"):
                worker.run()
            self.assertTrue(results[0].save_success)
            self.assertFalse(results[0].phase_check_success)
            self.assertTrue(Path(results[0].raw_stack_path).is_file())

    def test_optional_voltage_plot_failure_does_not_fail_capture_save(self):
        timestamp = "20261014_120602"
        frames = [np.full((4, 5), index + 1, dtype=np.uint16) for index in range(6)]

        with tempfile.TemporaryDirectory() as temporary_directory:
            def complete_phase_check(session_dir):
                check_path = Path(session_dir) / "measured_phase_check.json"
                check_path.write_text(
                    json.dumps({"measured_phase_deg": [0, 60, 120, 180, 240, 300]}),
                    encoding="utf-8",
                )
                figure_path = Path(session_dir) / "measured_vs_expected_phase_annotated.png"
                figure_path.touch()
                return PhaseCheckResult(
                    True,
                    "checker passed",
                    0,
                    str(figure_path),
                    str(check_path),
                )

            worker = camera.PhaseSaveWorker(
                frames=frames,
                save_dir=temporary_directory,
                session_prefix="phase_capture",
                trigger_positions_ms=[100, 190, 280, 370, 460, 550],
                voltage_mapper=lambda _position: (_ for _ in ()).throw(
                    OSError("plot device unavailable")
                ),
            )
            worker.run_automatic_phase_check = complete_phase_check
            results = []
            worker.result_ready.connect(results.append)

            with patch.object(camera.time, "strftime", return_value=timestamp):
                worker.run()
            result = results[0]
            self.assertTrue(result.save_success)
            self.assertTrue(result.phase_check_success)
            self.assertIsNone(result.phase_voltage_plot_path)
            self.assertTrue(Path(result.raw_stack_path).is_file())
            self.assertTrue(Path(result.metadata_path).is_file())

    def test_phase_save_worker_reports_storage_failure(self):
        frames = [np.full((4, 5), index + 1, dtype=np.uint16) for index in range(6)]
        with tempfile.TemporaryDirectory() as temporary_directory:
            worker = camera.PhaseSaveWorker(
                frames=frames,
                save_dir=temporary_directory,
                session_prefix="phase_calibration",
            )
            results = []
            worker.result_ready.connect(results.append)

            with (
                patch.object(camera.time, "strftime", return_value="20261014_120201"),
                patch.object(camera.np, "save", side_effect=OSError("disk full")),
            ):
                worker.run()

        self.assertFalse(results[0].save_success)
        self.assertIn("disk full", results[0].error)

    def test_optional_roi_mask_write_failure_is_reported_without_blocking_params(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch(
                "phase_capture_storage.np.save",
                side_effect=OSError("mask device unavailable"),
            ):
                result = save_roi_snapshots(
                    temporary_directory,
                    np.ones((3, 4), dtype=bool),
                    {"roi_type": "rectangle"},
                )

            self.assertIsNone(result["roi_mask_path"])
            self.assertTrue(Path(result["roi_params_path"]).is_file())
            self.assertEqual(len(result["warnings"]), 1)
            self.assertIn("mask device unavailable", result["warnings"][0])

    def test_roi_mask_write_failure_marks_metadata_roi_unavailable(self):
        timestamp = "20261014_120605"
        frames = [np.full((4, 5), index + 1, dtype=np.uint16) for index in range(6)]
        original_save = np.save

        def fail_mask_save(path, array, *args, **kwargs):
            if Path(path).name == "roi_mask.npy":
                raise OSError("mask device unavailable")
            return original_save(path, array, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary_directory:
            def complete_phase_check(session_dir):
                check_path = Path(session_dir) / "measured_phase_check.json"
                check_path.write_text(
                    json.dumps({"measured_phase_deg": [0, 60, 120, 180, 240, 300]}),
                    encoding="utf-8",
                )
                figure_path = Path(session_dir) / "measured_vs_expected_phase_annotated.png"
                figure_path.touch()
                return PhaseCheckResult(
                    True,
                    "checker passed",
                    0,
                    str(figure_path),
                    str(check_path),
                )

            worker = camera.PhaseSaveWorker(
                frames=frames,
                save_dir=temporary_directory,
                roi_mask=np.ones((4, 5), dtype=bool),
                session_prefix="phase_capture",
            )
            worker.run_automatic_phase_check = complete_phase_check
            results = []
            worker.result_ready.connect(results.append)

            with (
                patch.object(camera.time, "strftime", return_value=timestamp),
                patch("phase_capture_storage.np.save", side_effect=fail_mask_save),
            ):
                worker.run()

            result = results[0]
            metadata = json.loads(Path(result.metadata_path).read_text(encoding="utf-8"))
            self.assertTrue(result.save_success)
            self.assertTrue(result.phase_check_success)
            self.assertFalse(metadata["roi"]["roi_available"])
            self.assertIsNone(metadata["roi"]["roi_mask_path"])

    def test_mismatched_roi_mask_shape_is_skipped_without_failing_capture(self):
        timestamp = "20261014_120606"
        frames = [np.full((4, 5), index + 1, dtype=np.uint16) for index in range(6)]

        with tempfile.TemporaryDirectory() as temporary_directory:
            def complete_phase_check(session_dir):
                check_path = Path(session_dir) / "measured_phase_check.json"
                check_path.write_text(
                    json.dumps({"measured_phase_deg": [0, 60, 120, 180, 240, 300]}),
                    encoding="utf-8",
                )
                figure_path = Path(session_dir) / "measured_vs_expected_phase_annotated.png"
                figure_path.touch()
                return PhaseCheckResult(
                    True,
                    "checker passed",
                    0,
                    str(figure_path),
                    str(check_path),
                )

            worker = camera.PhaseSaveWorker(
                frames=frames,
                save_dir=temporary_directory,
                roi_mask=np.ones((3, 5), dtype=bool),
                roi_params={"shape": "rectangle"},
                session_prefix="phase_capture",
            )
            worker.run_automatic_phase_check = complete_phase_check
            results = []
            worker.result_ready.connect(results.append)

            with patch.object(camera.time, "strftime", return_value=timestamp):
                worker.run()

            result = results[0]
            metadata = json.loads(Path(result.metadata_path).read_text(encoding="utf-8"))
            self.assertTrue(result.save_success)
            self.assertTrue(result.phase_check_success)
            self.assertFalse(metadata["roi"]["roi_available"])
            self.assertIsNone(metadata["roi"]["roi_mask_path"])
            self.assertIsNotNone(metadata["roi"]["roi_params_path"])

    def test_optional_roi_params_write_failure_is_reported_without_blocking_mask(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch(
                "phase_capture_storage.json.dump",
                side_effect=OSError("parameter device unavailable"),
            ):
                result = save_roi_snapshots(
                    temporary_directory,
                    np.ones((3, 4), dtype=bool),
                    {"roi_type": "rectangle"},
                )

            self.assertTrue(Path(result["roi_mask_path"]).is_file())
            self.assertIsNone(result["roi_params_path"])
            self.assertEqual(len(result["warnings"]), 1)
            self.assertIn("parameter device unavailable", result["warnings"][0])

    def test_png_write_failure_returns_failed_result_but_keeps_raw_stack(self):
        frames = [np.full((4, 5), index + 1, dtype=np.uint16) for index in range(6)]
        with tempfile.TemporaryDirectory() as temporary_directory:
            worker = camera.PhaseSaveWorker(
                frames=frames,
                save_dir=temporary_directory,
                session_prefix="phase_capture",
            )
            results = []
            worker.result_ready.connect(results.append)
            with (
                patch.object(camera.time, "strftime", return_value="20261014_120600"),
                patch.object(
                    phase_capture_worker,
                    "save_phase_pngs",
                    side_effect=OSError("png disk failure"),
                ),
            ):
                worker.run()

            result = results[0]
            self.assertFalse(result.save_success)
            self.assertIn("png disk failure", result.error)
            self.assertTrue(Path(result.raw_stack_path).is_file())

    def test_partial_png_write_failure_preserves_completed_png_and_raw_stack(self):
        timestamp = "20261014_120603"
        frames = [np.full((4, 5), index + 1, dtype=np.uint16) for index in range(6)]
        write_count = 0

        def fail_after_first_png(path, image):
            nonlocal write_count
            write_count += 1
            if write_count == 2:
                raise OSError("second PNG disk failure")
            Path(path).write_bytes(b"first PNG placeholder")

        with tempfile.TemporaryDirectory() as temporary_directory:
            worker = camera.PhaseSaveWorker(
                frames=frames,
                save_dir=temporary_directory,
                session_prefix="phase_capture",
            )
            results = []
            worker.result_ready.connect(results.append)
            with (
                patch.object(camera.time, "strftime", return_value=timestamp),
                patch(
                    "phase_capture_storage.imageio.imwrite",
                    side_effect=fail_after_first_png,
                ),
                patch.object(worker, "run_automatic_phase_check") as phase_check,
            ):
                worker.run()

            result = results[0]
            session_dir = Path(result.session_dir)
            self.assertFalse(result.save_success)
            self.assertIn("second PNG disk failure", result.error)
            self.assertTrue(Path(result.raw_stack_path).is_file())
            self.assertTrue((session_dir / f"phase_0_{timestamp}.png").is_file())
            self.assertFalse((session_dir / f"phase_1_{timestamp}.png").exists())
            self.assertIsNone(result.metadata_path)
            phase_check.assert_not_called()

    def test_metadata_write_failure_returns_failed_result_after_required_artifacts(self):
        frames = [np.full((4, 5), index + 1, dtype=np.uint16) for index in range(6)]
        with tempfile.TemporaryDirectory() as temporary_directory:
            worker = camera.PhaseSaveWorker(
                frames=frames,
                save_dir=temporary_directory,
                session_prefix="phase_capture",
            )
            results = []
            worker.result_ready.connect(results.append)
            with (
                patch.object(camera.time, "strftime", return_value="20261014_120601"),
                patch.object(
                    phase_capture_worker,
                    "save_phase_capture_metadata",
                    side_effect=OSError("metadata disk failure"),
                ),
                patch.object(worker, "run_automatic_phase_check") as phase_check,
            ):
                worker.run()

            result = results[0]
            self.assertFalse(result.save_success)
            self.assertIn("metadata disk failure", result.error)
            self.assertTrue(Path(result.raw_stack_path).is_file())
            self.assertTrue(list(Path(result.session_dir).glob("phase_*.png")))
            self.assertIsNone(result.metadata_path)
            phase_check.assert_not_called()

    def test_phase_check_process_returns_success_only_with_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checker_path = Path(temporary_directory, "checker.py")
            checker_path.touch()
            session_dir = Path(temporary_directory, "session")
            session_dir.mkdir()

            def successful_runner(command, **kwargs):
                self.assertEqual(command[0], "fake-python.exe")
                self.assertEqual(command[1], str(checker_path))
                self.assertEqual(kwargs["cwd"], temporary_directory)
                self.assertEqual(kwargs["timeout"], 12)
                (session_dir / "measured_vs_expected_phase_annotated.png").touch()
                (session_dir / "measured_phase_check.json").touch()
                return SimpleNamespace(returncode=0, stdout="ok", stderr="")

            result = run_phase_check(
                str(checker_path),
                str(session_dir),
                temporary_directory,
                12,
                python_executable="fake-python.exe",
                runner=successful_runner,
            )

        self.assertTrue(result.success)
        self.assertEqual(result.return_code, 0)
        self.assertEqual(Path(result.output_json_path).name, "measured_phase_check.json")
        self.assertEqual(result.stdout, "ok")

    def test_phase_check_process_reports_nonzero_or_missing_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checker_path = Path(temporary_directory, "checker.py")
            checker_path.touch()
            session_dir = Path(temporary_directory, "session")
            session_dir.mkdir()
            result = run_phase_check(
                str(checker_path),
                str(session_dir),
                temporary_directory,
                12,
                runner=lambda *_args, **_kwargs: SimpleNamespace(
                    returncode=2,
                    stdout="partial",
                    stderr="failure",
                ),
            )

        self.assertFalse(result.success)
        self.assertEqual(result.return_code, 2)
        self.assertIn("Return code: 2", result.message)

    def test_phase_check_process_reports_missing_script_and_timeout(self):
        missing = run_phase_check(
            "missing_checker.py",
            ".",
            ".",
            12,
            runner=lambda *_args, **_kwargs: self.fail("Runner must not execute"),
        )
        self.assertFalse(missing.success)
        self.assertIsNone(missing.return_code)
        self.assertIn("script not found", missing.message)

        with tempfile.TemporaryDirectory() as temporary_directory:
            checker_path = Path(temporary_directory, "checker.py")
            checker_path.touch()
            timed_out = run_phase_check(
                str(checker_path),
                temporary_directory,
                temporary_directory,
                0.01,
                runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    subprocess.TimeoutExpired("checker", 0.01, output="partial")
                ),
            )

        self.assertFalse(timed_out.success)
        self.assertIsNone(timed_out.return_code)
        self.assertEqual(timed_out.stdout, "partial")
        self.assertIn("timed out", timed_out.message)

    def test_phase_checker_writes_expected_json_contract(self):
        timestamp = "20261014_120100"
        measured = np.asarray([0, 65.1, 120, 180, 240, 300], dtype=np.float64)

        with tempfile.TemporaryDirectory() as temporary_directory:
            session_dir = Path(temporary_directory)
            (session_dir / "capture_metadata.json").write_text(
                json.dumps({"phase_capture_model": {"algorithm_id": SIX_BUCKET_60_DEG}}),
                encoding="utf-8",
            )
            raw_stack_path = session_dir / "capture_raw_stack.npy"

            with (
                patch.object(phase_check, "find_capture_folder", return_value=session_dir),
                patch.object(phase_check, "find_raw_stack", return_value=raw_stack_path),
                patch.object(phase_check.np, "load", return_value=np.zeros((6, 2, 2))),
                patch.object(
                    phase_check,
                    "estimate_phase_curve",
                    return_value=(
                        measured,
                        {
                            "peak_x": 10,
                            "peak_y": 11,
                            "mask_radius": 18,
                            "peak_distance": 5.0,
                            "spectrum": np.ones((2, 2)),
                        },
                    ),
                ),
                patch.object(phase_check, "create_annotated_plot") as create_plot,
            ):
                phase_check.main()

            report_path = session_dir / "measured_phase_check.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["algorithm_id"], SIX_BUCKET_60_DEG)
            self.assertEqual(report["expected_phase_deg"], [0, 60, 120, 180, 240, 300])
            self.assertFalse(report["within_plus_minus_5_deg"])
            self.assertAlmostEqual(report["phase_error_deg"][1], 5.1)
            self.assertNotIn("spectrum", report["fft_and_phase_diagnostics"])
            create_plot.assert_called_once()

    def test_iterative_refinement_moves_timings_toward_phase_targets(self):
        current_positions = [100, 180, 260, 340, 420, 500]
        measured_phase = [0, 65, 130, 195, 260, 325]

        refined = compute_iterative_refined_positions_ms(
            current_positions,
            measured_phase,
            [0, 90, 180, 270, 360, 450],
            config=DEFAULT_RECALIBRATION_CONFIG,
            camera_required_spacing_ms=44,
        )

        self.assertNotEqual(refined, current_positions)
        self.assertEqual(refined[0], current_positions[0])
        self.assertEqual(refined, sorted(refined))
        self.assertTrue(all(
            abs(new - old) <= DEFAULT_RECALIBRATION_CONFIG.large_error_time_step_ms
            for old, new in zip(current_positions, refined)
        ))

    def test_nonpassing_verification_queues_next_refined_capture(self):
        next_positions = [100, 190, 280, 370, 460, 550]
        candidate = {
            "phase_error_deg": [0, 8, 12, 20, 30, 40],
            "max_abs_error": 40.0,
            "count_error_over_5": 5,
            "count_error_over_10": 4,
            "score": (4, 40.0, 5, 22.0, 18.0),
            "session_dir": "session",
            "trigger_positions_ms": [100, 190, 280, 370, 460, 550],
        }
        queued = []
        refined = []

        with tempfile.TemporaryDirectory() as temporary_directory:
            check_path = Path(temporary_directory) / "measured_phase_check.json"
            check_path.write_text(
                json.dumps({"measured_phase_deg": [0, 98, 192, 290, 390, 490]}),
                encoding="utf-8",
            )
            viewer = SimpleNamespace(
                latest_phase_save_result=None,
                pending_phase_capture_layer=None,
                recalibration_active=True,
                recalibration_stage="verification_capture",
                recalibration_candidates=[],
                recalibration_incomplete_verification_retries=0,
                recalibration_verification_capture_count=0,
                recalibration_pending_pass_candidate=None,
                recalibration_current_positions_ms=[100, 190, 280, 370, 460, 550],
                recalibration_target_positions_ms=[100, 190, 280, 370, 460, 550],
                recalibration_refinement_round=0,
                recalibration_coarse_positions_ms=[100, 190, 280, 370, 460, 550],
                recalibration_coarse_session_dir="coarse",
                phase_label=LabelStub(),
                status_label=LabelStub(),
                build_recalibration_candidate=lambda **kwargs: candidate,
                compute_iterative_refined_positions_ms=lambda **kwargs: refined.append(kwargs) or next_positions,
                save_recalibrated_trigger_positions=lambda **kwargs: None,
                queue_or_defer_recalibration_phase_capture=lambda positions, status: queued.append((positions, status)),
            )

            self.load_save_result_handler()(viewer, PhaseSaveResult(
                save_success=True,
                session_dir=temporary_directory,
                raw_stack_path=None,
                metadata_path=None,
                phase_check_json_path=str(check_path),
            ))

        self.assertEqual(viewer.recalibration_verification_capture_count, 1)
        self.assertEqual(viewer.recalibration_refinement_round, 1)
        self.assertEqual(viewer.recalibration_current_positions_ms, next_positions)
        np.testing.assert_array_equal(
            refined[0]["measured_phase_deg"],
            [0, 98, 192, 290, 390, 490],
        )
        self.assertEqual(queued[0][0], next_positions)
        self.assertIn("refine 1/3", queued[0][1])

    def test_nonpassing_verification_terminates_at_capture_limit(self):
        candidate = {
            "phase_error_deg": [0, 8, 12, 20, 30, 40],
            "max_abs_error": 40.0,
            "count_error_over_5": 5,
            "count_error_over_10": 4,
            "score": (4, 40.0, 5, 22.0, 18.0),
            "session_dir": "session",
            "trigger_positions_ms": [100, 190, 280, 370, 460, 550],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            check_path = Path(temporary_directory) / "measured_phase_check.json"
            check_path.write_text(
                json.dumps({"measured_phase_deg": [0, 98, 192, 290, 390, 490]}),
                encoding="utf-8",
            )
            finalized = []
            viewer = SimpleNamespace(
                latest_phase_save_result=None,
                pending_phase_capture_layer=None,
                recalibration_active=True,
                recalibration_stage="verification_capture",
                recalibration_candidates=[candidate],
                recalibration_incomplete_verification_retries=0,
                recalibration_verification_capture_count=3,
                recalibration_pending_pass_candidate=None,
                recalibration_current_positions_ms=[100, 190, 280, 370, 460, 550],
                recalibration_target_positions_ms=[100, 190, 280, 370, 460, 550],
                recalibration_refinement_round=2,
                recalibration_coarse_positions_ms=[100, 190, 280, 370, 460, 550],
                recalibration_coarse_session_dir="coarse",
                phase_label=LabelStub(),
                status_label=LabelStub(),
                build_recalibration_candidate=lambda **kwargs: candidate,
                should_auto_restart_recalibration=lambda best: False,
                finalize_recalibration_candidate=lambda **kwargs: finalized.append(kwargs),
            )

            self.load_save_result_handler()(viewer, PhaseSaveResult(
                save_success=True,
                session_dir=temporary_directory,
                raw_stack_path=None,
                metadata_path=None,
                phase_check_json_path=str(check_path),
            ))

        self.assertEqual(len(finalized), 1)
        self.assertTrue(finalized[0]["fallback_used"])
        self.assertIn("after 4 verification captures", viewer.status_label.value)


if __name__ == "__main__":
    unittest.main()