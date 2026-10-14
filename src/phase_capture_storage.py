"""Filesystem path allocation helpers for phase-capture sessions."""

import os


def allocate_capture_session_dir(save_dir, session_prefix, timestamp):
    os.makedirs(save_dir, exist_ok=True)

    base_name = f"{session_prefix}_{timestamp}"
    suffix = 0
    while True:
        session_name = (
            base_name if suffix == 0 else f"{base_name}_{suffix:02d}"
        )
        session_dir = os.path.join(save_dir, session_name)
        try:
            os.mkdir(session_dir)
            return session_dir
        except FileExistsError:
            suffix += 1