"""Helpers for dispatching subprocess work in source and frozen builds."""

import sys


def build_worker_command(mode, script_path, arguments, python_executable=None):
    if getattr(sys, "frozen", False):
        return [sys.executable, f"--mwdriver-{mode}", *map(str, arguments)]
    return [python_executable or sys.executable, str(script_path), *map(str, arguments)]