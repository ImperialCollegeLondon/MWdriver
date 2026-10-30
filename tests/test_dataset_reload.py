import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mwdriver.ui.camera_phase_analysis_controller import CameraPhaseAnalysisController


class DatasetReloadTests(unittest.TestCase):
    def test_directory_picker_loads_stack_and_optional_saved_analysis(self):
        for has_analysis in (True, False):
            with self.subTest(has_analysis=has_analysis), tempfile.TemporaryDirectory() as directory:
                controller = self.make_controller()
                controller.create_phase_analysis_controls = MagicMock()
                controller.show_selected_phase_analysis_result = MagicMock()
                session = Path(directory)
                frames = np.ones((6, 4, 5), dtype=np.uint16)
                np.save(session / "measure_test_raw_stack.npy", frames)
                if has_analysis:
                    np.savez(session / "phase_analysis_results.npz", wrapped=np.ones((4, 5)))
                with patch("mwdriver.ui.camera_phase_analysis_controller.QFileDialog.getExistingDirectory", return_value=directory):
                    controller.reload_stack_from_directory()
                np.testing.assert_array_equal(controller.viewer.add_image.call_args.args[0], frames)
                self.assertEqual(len(controller.phase_capture_time_series_layers), 1)
                self.assertEqual(controller.show_selected_phase_analysis_result.called, has_analysis)
                if has_analysis:
                    self.assertIn("wrapped", controller.phase_analysis_results)
                else:
                    self.assertIn("no saved analysis", controller.status_label.setText.call_args.args[0])

    def test_directory_picker_rejects_missing_or_ambiguous_stacks(self):
        for count in (0, 2):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                controller = self.make_controller()
                for index in range(count):
                    np.save(Path(directory) / f"capture_{index}_raw_stack.npy", np.ones((6, 4, 5)))
                with patch("mwdriver.ui.camera_phase_analysis_controller.QFileDialog.getExistingDirectory", return_value=directory):
                    controller.reload_stack_from_directory()
                controller.viewer.add_image.assert_not_called()

    def test_directory_picker_cancel_or_capture_does_not_load(self):
        controller = self.make_controller()
        with patch("mwdriver.ui.camera_phase_analysis_controller.QFileDialog.getExistingDirectory", return_value="") as picker:
            controller.reload_stack_from_directory()
            picker.assert_called_once()
            picker.reset_mock()
            controller.phase_capture_active = True
            controller.reload_stack_from_directory()
            picker.assert_not_called()
        controller.viewer.add_image.assert_not_called()

    def test_corrupt_saved_analysis_does_not_discard_loaded_stack(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = self.make_controller()
            np.save(Path(directory) / "capture_raw_stack.npy", np.ones((6, 4, 5)))
            np.savez(Path(directory) / "phase_analysis_results.npz", unrelated=np.ones(3))
            with patch("mwdriver.ui.camera_phase_analysis_controller.QFileDialog.getExistingDirectory", return_value=directory):
                controller.reload_stack_from_directory()
            self.assertEqual(len(controller.phase_capture_time_series_layers), 1)
            self.assertIn("Stack loaded, but", controller.status_label.setText.call_args.args[0])

    def test_reload_analysis_restores_results_without_recalculation(self):
        controller = self.make_controller()
        controller.create_phase_analysis_controls = MagicMock()
        controller.show_selected_phase_analysis_result = MagicMock()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "phase_analysis_results.npz"
            wrapped = np.ones((4, 5), dtype=np.float32)
            np.savez(path, wrapped=wrapped, roi_mask=np.ones((4, 5), dtype=bool))
            controller.load_saved_analysis(path)
            np.testing.assert_array_equal(controller.phase_analysis_results["wrapped"], wrapped)
            self.assertEqual(controller.phase_analysis_source_stack_name, path.parent.name)
        controller.create_phase_analysis_controls.assert_called_once_with(selected_key="wrapped")
        controller.show_selected_phase_analysis_result.assert_called_once_with(create_new_layer=True)
        self.assertTrue(controller.phase_analysis_view_active)

    def test_invalid_analysis_archive_preserves_existing_results(self):
        controller = self.make_controller()
        previous = {"wrapped": np.ones((2, 2))}
        controller.phase_analysis_results = previous
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.npz"
            np.savez(path, unrelated=np.ones(3))
            with self.assertRaisesRegex(ValueError, "no recognised"):
                controller.load_saved_analysis(path)
            np.savez(path, wrapped=np.ones(3))
            with self.assertRaisesRegex(ValueError, "Invalid analysis image"):
                controller.load_saved_analysis(path)
        self.assertIs(controller.phase_analysis_results, previous)

    def test_cancelled_analysis_picker_leaves_results_unchanged(self):
        controller = self.make_controller()
        controller.load_saved_analysis = MagicMock()
        with patch("mwdriver.ui.camera_phase_analysis_controller.QFileDialog.getOpenFileName", return_value=("", "")):
            controller.reload_analysis_from_file()
        controller.load_saved_analysis.assert_not_called()

    def make_controller(self):
        controller = CameraPhaseAnalysisController()
        controller.viewer = SimpleNamespace(
            add_image=MagicMock(return_value=SimpleNamespace(metadata={})),
            layers=SimpleNamespace(selection=SimpleNamespace(active=None)),
            reset_view=MagicMock(),
            window=SimpleNamespace(_qt_window=None),
        )
        controller.phase_capture_time_series_layers = []
        controller.view_phase_analysis_button = None
        controller.apply_layer_visibility_plan = MagicMock()
        controller.sync_stream_with_active_layer = MagicMock()
        controller.status_label = MagicMock()
        controller.phase_capture_active = False
        controller.recalibration_active = False
        return controller

    def test_load_restores_stack_and_capture_metadata(self):
        controller = self.make_controller()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "measure_test_raw_stack.npy"
            frames = np.arange(120, dtype=np.uint16).reshape(6, 4, 5)
            np.save(path, frames)
            metadata = {"phase_capture_model": {"algorithm_id": "six_bucket_60deg"}}
            path.with_name("measure_test_metadata.json").write_text(json.dumps(metadata))
            layer = controller.load_saved_dataset(path)
            call = controller.viewer.add_image.call_args
            np.testing.assert_array_equal(call.args[0], frames)
            self.assertEqual(call.kwargs["metadata"]["algorithm_id"], "six_bucket_60deg")
            self.assertEqual(call.kwargs["metadata"]["raw_stack_path"], str(path.resolve()))
            self.assertEqual(controller.phase_capture_time_series_layers, [layer])
            controller.apply_layer_visibility_plan.assert_called_once_with("time_series", selected_layer=layer)
            self.assertIs(controller.viewer.layers.selection.active, layer)

    def test_legacy_stack_defaults_to_90_degrees(self):
        controller = self.make_controller()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy_raw_stack.npy"
            np.save(path, np.ones((6, 4, 5)))
            controller.load_saved_dataset(path)
            self.assertEqual(controller.viewer.add_image.call_args.kwargs["metadata"]["algorithm_id"], "six_frame_90deg")

    def test_invalid_stack_does_not_add_a_layer(self):
        controller = self.make_controller()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.npy"
            np.save(path, np.ones((5, 4, 5)))
            with self.assertRaisesRegex(ValueError, "six-frame"):
                controller.load_saved_dataset(path)
        controller.viewer.add_image.assert_not_called()

    def test_cancelled_picker_leaves_view_unchanged(self):
        controller = self.make_controller()
        with patch("mwdriver.ui.camera_phase_analysis_controller.QFileDialog.getOpenFileName", return_value=("", "")):
            controller.reload_dataset_from_file()
        controller.viewer.add_image.assert_not_called()