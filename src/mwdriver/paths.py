"""Shared locations for user-owned measurements and settings."""

import ctypes
import os
from pathlib import Path


def get_results_directory():
    documents = Path.home() / "Documents"
    if os.name == "nt":
        buffer = ctypes.create_unicode_buffer(260)
        result = ctypes.windll.shell32.SHGetFolderPathW(None, 5, None, 0, buffer)
        if result != 0:
            raise OSError(f"Could not locate the Documents folder (HRESULT {result}).")
        documents = Path(buffer.value)
    return documents / "MWDriver"


RESULTS_DIR = get_results_directory()