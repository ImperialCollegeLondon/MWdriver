"""Persistent desktop diagnostics, including native Python fault traces."""

import faulthandler
import logging
import os
import platform
import sys
from datetime import datetime

from mwdriver.paths import RESULTS_DIR


LOGGER = logging.getLogger("mwdriver")
_fault_file = None
_qt_handler = None


def configure_diagnostics(directory=None):
    global _fault_file, _qt_handler
    if _fault_file is not None:
        return
    directory = RESULTS_DIR / "logs" if directory is None else directory
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"camera_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}"
    path = directory / f"{stem}.log"
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s"
    ))
    LOGGER.setLevel(logging.DEBUG)
    LOGGER.addHandler(handler)
    _fault_file = (directory / f"{stem}_fault.log").open("a", buffering=1)
    faulthandler.enable(file=_fault_file, all_threads=True)
    previous_hook = sys.excepthook

    def exception_hook(exception_type, exception, traceback):
        LOGGER.critical("Unhandled exception", exc_info=(exception_type, exception, traceback))
        previous_hook(exception_type, exception, traceback)

    sys.excepthook = exception_hook
    from qtpy.QtCore import qInstallMessageHandler

    def qt_message(message_type, context, message):
        LOGGER.warning("Qt [%s] %s (%s:%s)", message_type, message, context.file, context.line)

    _qt_handler = qt_message
    qInstallMessageHandler(_qt_handler)
    LOGGER.info("Startup Python=%s platform=%s executable=%s", sys.version, platform.platform(), sys.executable)
    for name in ("napari", "vispy"):
        module = sys.modules.get(name)
        LOGGER.info("%s version=%s", name, getattr(module, "__version__", "not loaded"))
    print(f"Diagnostic log: {path}", flush=True)