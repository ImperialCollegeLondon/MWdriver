# MWdriver Maintainability Plan

## Goal

Keep the hardware-driven six-frame phase-shifting interferometry workflow behaviourally stable while organising implementation by responsibility and testing pure policies independently.

## Package Layout

- `src/mwdriver/domain/`: PSI definitions, recalibration policies, ROI/profile geometry, stream selection, layer visibility, and display-frame preparation.
- `src/mwdriver/hardware/`: native Thorlabs camera adapter and Pico serial worker.
- `src/mwdriver/capture/`: phase-capture lifecycle, frame/timestamp policies, save workers, persistence, and phase-check subprocess execution.
- `src/mwdriver/analysis/`: phase-check CLI implementation, phase-analysis CLI implementation, reconstruction helpers, aberration analysis, plots, and result viewer.
- `src/mwdriver/ui/`: Napari/Qt controller mixins, controls, application bootstrap, and worker lifecycle.
- `src/camera.py`: desktop launcher, compatibility facade, and UI composition root.
- `src/check_measured_vs_expected_phase.py` and `src/phase_analysis_new.py`: retained CLI entry points that delegate to packaged implementations.
- `pypico/main.py`: independently deployed MicroPython firmware; it remains outside the desktop package.

There are no longer top-level mirrors for package implementation modules. Internal code and tests import package paths directly.

## Verified

- The complete test suite passes on Windows with Python 3.14.7 (146 tests).
- Focused tests cover phase assignment, stale-frame rejection, acquisition metadata, trigger ACKs, recalibration policies, save schemas/failure handling, ROI geometry/storage, profile sampling/dragging, controller behaviour, and CLI analysis contracts.
- Explicit-session analysis smoke tests preserve NPZ arrays, generated PNGs, Fourier diagnostics, and RMS/PV results against the established baseline.
- The user has confirmed the physical test is okay. Automated tests do not independently verify camera/Pico wiring or trigger timing.

## Remaining Work

- Decide whether to add packaging metadata for installation as a Python distribution. Current supported usage adds `src` to the import path and launches the documented scripts directly.
- Review whether the two ROI artifacts (`roi_params.json` and `roi_mask.npy`) need atomic paired replacement/recovery. This remains deliberately deferred.
- Keep `camera.py` as the compatibility facade unless a separate API migration is explicitly undertaken.

## Contracts To Preserve

- Six-frame acquisition order, frame/timestamp association, and internal/external/internal trigger transitions.
- Pico serial command protocol and trigger timing behaviour.
- Capture directory naming, filenames, metadata keys, and raw-stack shape.
- Existing CLI options and legacy 90-degree fallback for captures without algorithm metadata.
- Recalibration candidate cleanup and calibration-save semantics.
- The Pico firmware remains independently deployable.
