import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mwdriver import paths
from mwdriver.analysis import phase_analysis_new, check_measured_vs_expected_phase
from mwdriver.capture import phase_capture_controller, phase_capture_worker
from mwdriver.ui import camera_phase_analysis_controller
from mwdriver.ui import camera_recalibration_controller, camera_roi_controller


class ResultPathTests(unittest.TestCase):
    def test_calibration_cleanup_retries_readonly_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory) / "calibrate_readonly"
            session.mkdir()
            operation = MagicMock()

            def simulate_readonly(path, *, onexc):
                self.assertEqual(Path(path), session)
                onexc(operation, str(session), PermissionError("Read-only directory"))

            with patch.object(camera_recalibration_controller.shutil, "rmtree", side_effect=simulate_readonly):
                with patch.object(camera_recalibration_controller.os, "chmod") as chmod:
                    camera_recalibration_controller.remove_calibration_session(session)
            chmod.assert_called_once_with(str(session), stat.S_IWRITE)
            operation.assert_called_once_with(str(session))

    def test_calibration_cleanup_removes_readonly_files_and_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory) / "calibrate_readonly"
            session.mkdir()
            data = session / "capture.npy"
            data.touch()
            data.chmod(stat.S_IREAD)
            session.chmod(stat.S_IREAD | stat.S_IEXEC)
            camera_recalibration_controller.remove_calibration_session(session)
            self.assertFalse(session.exists())

    def test_final_calibration_retains_only_selected_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            coarse = root / "calibrate_coarse"
            rejected = root / "calibrate_rejected"
            selected = root / "calibrate_selected"
            unrelated = root / "measure_unrelated"
            for session in (coarse, rejected, selected, unrelated):
                session.mkdir()
                (session / "capture_raw_stack.npy").touch()
            selected_candidate = {"session_dir": str(selected)}
            viewer = SimpleNamespace(
                recalibration_coarse_session_dir=str(coarse),
                recalibration_candidates=[
                    {"session_dir": str(rejected)}, selected_candidate,
                ],
            )
            camera_recalibration_controller.CameraRecalibrationController.keep_only_recalibration_candidate(
                viewer, selected_candidate
            )
            self.assertFalse(coarse.exists())
            self.assertFalse(rejected.exists())
            self.assertTrue((selected / "capture_raw_stack.npy").exists())
            self.assertTrue(unrelated.exists())
            self.assertIsNone(viewer.recalibration_coarse_session_dir)
            self.assertEqual(viewer.recalibration_candidates, [selected_candidate])

    def test_default_paths_share_user_documents_directory(self):
        root = paths.RESULTS_DIR
        self.assertTrue(root.is_absolute())
        self.assertEqual(root.name, "MWDriver")
        for directory in (
            phase_capture_controller.SAVE_DIR,
            phase_capture_worker.DEFAULT_SAVE_DIR,
            camera_phase_analysis_controller.SAVE_DIR,
            phase_analysis_new.CAPTURE_DIR,
            check_measured_vs_expected_phase.CAPTURE_DIR,
            camera_roi_controller.RESULTS_DIR,
        ):
            self.assertEqual(Path(directory), root)
        self.assertEqual(
            Path(camera_recalibration_controller.RECALIBRATION_TRIGGER_JSON_PATH).parent,
            root,
        )

    @unittest.skipUnless(os.name == "nt", "Windows Documents folder lookup")
    def test_redirected_documents_directory_is_respected(self):
        def locate_documents(window, folder, token, flags, buffer):
            self.assertEqual(folder, 5)
            buffer.value = r"C:\Redirected\Documents"
            return 0

        with patch.object(paths.ctypes.windll.shell32, "SHGetFolderPathW", side_effect=locate_documents):
            self.assertEqual(paths.get_results_directory(), Path(r"C:\Redirected\Documents\MWDriver"))

    def test_calibration_and_verification_use_calibrate_prefix(self):
        for active, stage, expected in (
            (False, None, "measure"),
            (True, "coarse_capture", "calibrate"),
            (True, "verify_capture", "calibrate"),
        ):
            viewer = SimpleNamespace(
                save_thread=None, roi_mask=None,
                recalibration_active=active, recalibration_stage=stage,
                last_capture_trigger_positions_ms=[], last_capture_algorithm_id="six_frame_90deg",
            )
            for callback in ("update_status", "on_phase_save_result", "on_phase_save_finished", "cleanup_save_worker"):
                setattr(viewer, callback, MagicMock())
            with patch.object(phase_capture_controller, "QThread"):
                with patch.object(phase_capture_controller, "PhaseSaveWorker") as worker:
                    phase_capture_controller.PhaseCaptureController.start_save_phase_frames_worker(viewer, [], [])
            self.assertEqual(worker.call_args.kwargs["session_prefix"], expected)

    def test_latest_discovery_includes_measure_and_excludes_calibrate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "phase_capture_old"
            measured = root / "measure_20261019_120000"
            calibration = root / "calibrate_20261019_130000"
            for timestamp, session in enumerate((legacy, measured, calibration), start=1):
                session.mkdir()
                stack = session / "capture_raw_stack.npy"
                stack.touch()
                os.utime(stack, (timestamp, timestamp))
                os.utime(session, (timestamp, timestamp))
            with patch.object(phase_analysis_new, "CAPTURE_DIR", root):
                self.assertEqual(phase_analysis_new.find_capture_input()[0], measured)
            with patch.object(check_measured_vs_expected_phase, "CAPTURE_DIR", root):
                self.assertEqual(check_measured_vs_expected_phase.find_capture_folder(), measured)
            viewer = SimpleNamespace(latest_phase_save_result=None)
            with patch.object(camera_phase_analysis_controller, "SAVE_DIR", str(root)):
                latest = camera_phase_analysis_controller.CameraPhaseAnalysisController.find_latest_phase_analysis_session(viewer)
            self.assertEqual(Path(latest), measured)


if __name__ == "__main__":
    unittest.main()