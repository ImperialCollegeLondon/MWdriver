import sys
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mwdriver.ui.camera_phase_analysis_controller as analysis_controller
from mwdriver.ui.camera_phase_analysis_controller import CameraPhaseAnalysisController
from mwdriver.capture.phase_capture_storage import PhaseSaveResult


class CameraPhaseAnalysisControllerTests(unittest.TestCase):
    def test_latest_successful_save_is_preferred_when_its_stack_exists(self):
        with tempfile.TemporaryDirectory() as save_root:
            session_dir = Path(save_root) / "phase_capture_latest"
            session_dir.mkdir()
            raw_stack_path = session_dir / "phase_capture_latest_raw_stack.npy"
            raw_stack_path.touch()
            viewer = SimpleNamespace(
                latest_phase_save_result=PhaseSaveResult(
                    save_success=True,
                    session_dir=str(session_dir),
                    raw_stack_path=str(raw_stack_path),
                    metadata_path=None,
                )
            )

            with patch.object(analysis_controller, "SAVE_DIR", save_root):
                selected = CameraPhaseAnalysisController.find_latest_phase_analysis_session(
                    viewer
                )

        self.assertEqual(selected, str(session_dir.resolve()))

    def test_fallback_selects_newest_persisted_session_with_raw_stack(self):
        with tempfile.TemporaryDirectory() as save_root:
            older = Path(save_root) / "phase_capture_older"
            newer = Path(save_root) / "phase_capture_newer"
            incomplete = Path(save_root) / "phase_capture_incomplete"
            older.mkdir()
            newer.mkdir()
            incomplete.mkdir()
            older_stack = older / "phase_capture_older_raw_stack.npy"
            newer_stack = newer / "phase_capture_newer_raw_stack.npy"
            older_stack.touch()
            newer_stack.touch()
            (incomplete / "metadata.json").touch()
            older_time = 1_000_000_000
            newer_time = 1_000_000_100
            os.utime(older_stack, (older_time, older_time))
            os.utime(newer_stack, (newer_time, newer_time))
            viewer = SimpleNamespace(latest_phase_save_result=None)

            with patch.object(analysis_controller, "SAVE_DIR", save_root):
                selected = CameraPhaseAnalysisController.find_latest_phase_analysis_session(
                    viewer
                )

        self.assertEqual(selected, str(newer.resolve()))


if __name__ == "__main__":
    unittest.main()