import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class DiagnosticsTests(unittest.TestCase):
    def test_log_and_fault_trace_are_written_in_child_process(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            code = (
                "import sys, faulthandler\n"
                "from pathlib import Path\n"
                f"sys.path.insert(0, {str(root / 'src')!r})\n"
                "from mwdriver.diagnostics import configure_diagnostics, LOGGER\n"
                f"configure_diagnostics(Path({directory!r}))\n"
                "LOGGER.info('surface transition checkpoint')\n"
                "import mwdriver.diagnostics as diagnostics\n"
                "faulthandler.dump_traceback(file=diagnostics._fault_file)\n"
                "raise RuntimeError('diagnostic exception probe')\n"
            )
            result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            logs = list(Path(directory).glob("*.log"))
            self.assertEqual(len(logs), 2)
            regular = next(path for path in logs if not path.name.endswith("_fault.log"))
            contents = regular.read_text(encoding="utf-8")
            self.assertIn("surface transition checkpoint", contents)
            self.assertIn("diagnostic exception probe", contents)
            fault = next(path for path in logs if path.name.endswith("_fault.log"))
            self.assertIn("most recent call first", fault.read_text())