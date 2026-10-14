# MWdriver Maintainability Plan

## Goal

Restructure the hardware-driven six-frame phase-shifting interferometry workflow around testable ownership boundaries, without changing camera/Pico timing, capture-session formats, offline command-line behavior, or compatibility with older captures.

## Current Progress

- **Implemented:** value-only trigger-position validation and coarse-measurement quality checks live in [`src/recalibration.py`](../src/recalibration.py), with the desktop controller delegating to them from [`src/camera.py`](../src/camera.py).
- **Implemented:** focused unit coverage for trigger rounding, bounds, ordering, spacing, measurement shape, finite values, span, backtracking, strict acceptance tolerance, and the coarse solver's list-to-array validation boundary in [`tests/test_recalibration.py`](../tests/test_recalibration.py).
- **Implemented:** fake-driven coverage for stale-frame acceptance boundaries, timestamp normalization and fallback, and native frame acquisition metadata in [`tests/test_phase_capture_helpers.py`](../tests/test_phase_capture_helpers.py) and [`tests/test_thorlabs_native_camera.py`](../tests/test_thorlabs_native_camera.py).
- **Implemented:** controller contract tests cover phase-slot assignment, stale acquisition rejection, Pico serial command payloads and capture-lifecycle messages, trigger-update ACK parsing/match/rejection/mismatch/timeout, saved session paths/filenames and metadata, phase-check JSON output, save/check failure reporting, iterative refinement scheduling, and verification-limit termination in [`tests/test_camera_contracts.py`](../tests/test_camera_contracts.py).
- **Implemented:** value-only coarse trigger generation, camera-safe spacing enforcement, coarse phase interpolation, damped iterative refinement, candidate scoring, strict acceptance, and auto-restart eligibility now live in [`src/recalibration.py`](../src/recalibration.py). The GUI supplies PSI targets, camera-required spacing, bounds, tolerances, and damping settings explicitly.
- **Implemented:** direct unit tests cover coarse/refined timing bounds and spacing, score ordering, and auto-restart thresholds in [`tests/test_recalibration.py`](../tests/test_recalibration.py).
- **Implemented:** calibration captures continue to save files and run phase checks, but no longer add six-frame stacks as Napari layers.
- **Implemented:** unique phase-capture session directory allocation now lives in [`src/phase_capture_storage.py`](../src/phase_capture_storage.py) and preserves the existing timestamp and numeric suffix convention.
- **Fixed:** hardware log evidence exposed a list/NumPy-array mismatch in the coarse recalibration solver, which stopped the workflow before the first verification capture. The validated phase list is now converted back to an array before interpolation.
- **Fixed:** only verification candidates with every point inside +/-5 degrees count as a pass; candidates outside that band continue through iterative refinement instead of going straight to same-timing confirmation.
- **Verified:** the complete unit suite passes on the Windows target using Python 3.14.7 (49 tests).
- **Not yet verified:** a successful post-fix end-to-end camera/Pico calibration cycle. The Pico is visible on COM4, but this session cannot drive the Napari GUI to complete calibration; unit tests use simulated hardware and cannot validate physical trigger timing or wiring.

## Work Plan

1. **Characterise existing contracts (in progress).** Coverage now includes frame-to-phase assignment, timestamps and stale-frame boundaries, native acquisition metadata, capture paths and filenames, phase-check output and failures, Pico command payloads, capture-lifecycle messages and trigger-update acknowledgements, plus recalibration iteration/termination. Remaining verification is the successful end-to-end hardware workflow.
2. **Finish extracting recalibration logic (in progress).** Value-only trigger generation/spacing, phase interpolation, iterative refinement, candidate scoring, and acceptance/restart decisions now live in Qt-free helpers using an immutable [`RecalibrationConfig`](../src/recalibration.py). Camera-required spacing remains an explicit runtime input. A final review of remaining validation/measurement policy is outstanding; capture scheduling, Pico acknowledgements, UI state, and persistence remain in the controller.
3. **Separate capture persistence from orchestration (in progress).** Unique session-directory allocation is extracted and tested for same-timestamp collisions. Next centralise writes for raw stacks, PNGs, ROI data and metadata; isolate checker subprocess execution and optional notebook generation; then return a defined result object instead of an informal dictionary.
4. **Make offline analysis import-safe.** Put reconstruction and aberration calculations behind callable functions that return a result bundle. Keep CLI parsing, session discovery, exports, plotting, and optional viewing at explicit entry points. Preserve current command-line options and the legacy 90-degree default for captures without algorithm metadata.
5. **Split the GUI after service boundaries stabilise.** Migrate Qt workers, capture/recalibration orchestration, ROI/profile controls, analysis review, and application bootstrap in small slices. Keep the current launcher usable during migration.

## Target Boundaries

A possible eventual package layout is:

- `mwdriver/domain/`: pure phase and calibration rules.
- `mwdriver/hardware/`: camera SDK and Pico serial adapters.
- `mwdriver/capture/`: session paths, persistence, and capture workflow services.
- `mwdriver/analysis/`: callable phase reconstruction and analysis.
- `mwdriver/ui/`: Napari/Qt workers, widgets, and orchestration.

Do not move code mechanically just to match this layout. Extract a boundary when it has a coherent responsibility and can be tested independently. Keep `src/camera.py` as a launcher during migration if that helps preserve existing usage.

## Contracts To Preserve

- Six-frame acquisition order and existing frame/timestamp association.
- Pico serial command protocol and hardware trigger timing behavior.
- Capture directory naming, file names, metadata keys, and raw-stack shape.
- Existing phase-check and analysis command-line options.
- Captures without algorithm metadata continue to use the legacy 90-degree method.
- Existing recalibration artifact cleanup and calibration-save semantics remain unchanged unless explicitly tested and intentionally revised.
- Keep [`pypico/main.py`](../pypico/main.py) independently deployable; do not combine firmware and desktop restructuring.

## Verification

- Run `python -m unittest discover -s tests -v` in the supported project environment after each extracted boundary.
- Add focused fake-driven tests for each newly isolated contract, including analysis imports and storage schema.
- Smoke-test documented analysis commands for explicit sessions, latest-session discovery, `--no-viewer`, and `--export`.
- On the supported Windows hardware setup, verify Pico acknowledgements, six external-trigger frame order, camera trigger-mode restoration, and a successful post-fix complete capture/recalibration cycle.

## Risks And Open Questions

- Hardware timing and native SDK frame metadata cannot be fully certified by unit tests or on a machine without the supported camera/Pico setup.
- The optional `phase_shifts.ipynb` template is documented as absent. Confirm whether generated per-capture notebooks remain a supported behavior before changing that path.
- Recalibration can delete rejected session directories and writes trigger positions during the workflow; preserve these semantics while extracting code.
- Napari integration uses private Qt APIs and a custom close handler. Treat compatibility work as a separate concern unless a module move requires it.
