"""Compatibility CLI for the packaged phase-capture checker."""

import sys

from mwdriver.analysis import check_measured_vs_expected_phase as _implementation

if __name__ == "__main__":
    raise SystemExit(_implementation.main())

sys.modules[__name__] = _implementation
