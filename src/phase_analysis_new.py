"""Compatibility CLI for the packaged phase-analysis application."""

import sys

from mwdriver.analysis import phase_analysis_new as _implementation

if __name__ == "__main__":
    _implementation.main()

sys.modules[__name__] = _implementation
