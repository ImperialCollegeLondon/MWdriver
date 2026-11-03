import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, "src")

from mwdriver import launcher
from mwdriver.runtime import build_worker_command


class StandaloneLauncherTests(unittest.TestCase):
    def test_worker_commands_select_source_or_frozen_dispatch(self):
        with patch("mwdriver.runtime.sys.frozen", False, create=True):
            self.assertEqual(
                build_worker_command("phase-check", "checker.py", ["run"]),
                [sys.executable, "checker.py", "run"],
            )
        with patch("mwdriver.runtime.sys.frozen", True, create=True), patch(
            "mwdriver.runtime.sys.executable", "C:/Program Files/MWdriver/MWdriver.exe"
        ):
            self.assertEqual(
                build_worker_command("phase-check", "checker.py", ["run", "--roi-mask", "mask.npy"]),
                ["C:/Program Files/MWdriver/MWdriver.exe", "--mwdriver-phase-check", "run", "--roi-mask", "mask.npy"],
            )

    def test_phase_checker_mode_does_not_start_camera(self):
        with patch(
            "mwdriver.analysis.check_measured_vs_expected_phase.main",
            return_value=object(),
        ) as phase_check, patch("camera.main") as camera_main:
            result = launcher.main([
                "--mwdriver-phase-check", "C:/captures/run", "--no-diagnostic-plots",
            ])

        self.assertEqual(result, 0)
        phase_check.assert_called_once_with([
            "C:/captures/run", "--no-diagnostic-plots",
        ])
        camera_main.assert_not_called()

    def test_phase_analysis_mode_forwards_cli_options(self):
        with patch(
            "mwdriver.analysis.phase_analysis_new.main",
            return_value=object(),
        ) as phase_analysis, patch("camera.main") as camera_main:
            result = launcher.main([
                "--mwdriver-phase-analysis", "C:/captures/run", "--no-viewer",
            ])

        self.assertEqual(result, 0)
        phase_analysis.assert_called_once_with([
            "C:/captures/run", "--no-viewer",
        ])
        camera_main.assert_not_called()

    def test_default_mode_starts_camera_application(self):
        with patch("camera.main") as camera_main:
            self.assertEqual(launcher.main([]), 0)
        camera_main.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()