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
- **Implemented:** ROI snapshot, raw-stack, phase PNG and capture-metadata writes are now storage helpers; the GUI worker retains ordering, notebook/checker orchestration and UI progress signals.
- **Implemented:** phase-check subprocess execution now lives in [`src/phase_capture_process.py`](../src/phase_capture_process.py) and returns a typed `PhaseCheckResult`; success requires exit code zero plus both expected output files.
- **Implemented:** optional per-session notebook generation now lives in [`src/phase_capture_storage.py`](../src/phase_capture_storage.py) and returns a typed `NotebookGenerationResult`.
- **Implemented:** `PhaseSaveWorker` now returns a typed `PhaseSaveResult` for successful and failed saves; the GUI and recalibration result consumer use its fields directly.
- **Implemented:** Pico serial protocol and thread work now live in [`src/pico_serial_worker.py`](../src/pico_serial_worker.py); [`src/camera.py`](../src/camera.py) re-exports the worker for launcher compatibility.
- **Implemented:** manual frame saving now lives in [`src/manual_capture_worker.py`](../src/manual_capture_worker.py); `camera.FrameSaveWorker` remains available through a compatibility import.
- **Implemented:** phase-capture saving and progress reporting now live in [`src/phase_capture_worker.py`](../src/phase_capture_worker.py); the GUI supplies its voltage mapper and keeps the existing `camera.PhaseSaveWorker` import path.
- **Implemented:** ROI vertex scaling, clipping, rectangle/ellipse masks and ROI metadata construction now live in the Qt-free [`src/roi_geometry.py`](../src/roi_geometry.py), with direct tests for shape and boundary behavior.
- **Implemented:** ROI menu action construction and callback wiring now live in [`src/roi_controls.py`](../src/roi_controls.py); the controller retains behavior callbacks and its existing fallback dock panel.
- **Implemented:** intensity profile widget construction now lives in [`src/profile_controls.py`](../src/profile_controls.py); the controller retains profile behavior, signal targets and dock placement.
- **Implemented:** raw-frame sampling along the display-space intensity profile line now lives in the pure [`src/intensity_profile.py`](../src/intensity_profile.py) helper, with tests for downsampling, clipping, and point limits.
- **Implemented:** phase-analysis title/selector/view-button construction now lives in [`src/phase_analysis_controls.py`](../src/phase_analysis_controls.py); option population and Napari layer behavior remain in the controller.
- **Implemented:** desktop viewer creation, Napari event-loop startup and close-event cleanup wiring now live in [`src/application_bootstrap.py`](../src/application_bootstrap.py); `camera.main()` remains the launcher entry point.
- **Implemented:** offline phase analysis is import-safe and exposes `analyze_capture(...)` with a typed `PhaseAnalysisResult`; CLI session discovery, `--export`, and viewer selection remain in `main()`.
- **Fixed:** hardware log evidence exposed a list/NumPy-array mismatch in the coarse recalibration solver, which stopped the workflow before the first verification capture. The validated phase list is now converted back to an array before interpolation.
- **Fixed:** only verification candidates with every point inside +/-5 degrees count as a pass; candidates outside that band continue through iterative refinement instead of going straight to same-timing confirmation.
- **Verified:** the complete unit suite passes on the Windows target using Python 3.14.7 (69 tests). A real explicit-session CLI smoke test on a temporary copy completed with `--no-viewer --export`, producing the NPZ and PNG outputs; temporary-tree tests cover latest-session selection. Legacy missing-algorithm fallback remains covered in [`tests/test_phase_psi.py`](../tests/test_phase_psi.py).
- **Not yet verified:** a successful post-fix end-to-end camera/Pico calibration cycle. The Pico is visible on COM4, but this session cannot drive the Napari GUI to complete calibration; unit tests use simulated hardware and cannot validate physical trigger timing or wiring.

## Work Plan

1. **Characterise existing contracts (in progress).** Coverage now includes frame-to-phase assignment, timestamps and stale-frame boundaries, native acquisition metadata, capture paths and filenames, phase-check output and failures, Pico command payloads, capture-lifecycle messages and trigger-update acknowledgements, plus recalibration iteration/termination. Remaining verification is the successful end-to-end hardware workflow.
2. **Finish extracting recalibration logic (in progress).** Value-only trigger generation/spacing, phase interpolation, iterative refinement, candidate scoring, and acceptance/restart decisions now live in Qt-free helpers using an immutable [`RecalibrationConfig`](../src/recalibration.py). Camera-required spacing remains an explicit runtime input. A final review of remaining validation/measurement policy is outstanding; capture scheduling, Pico acknowledgements, UI state, and persistence remain in the controller.
3. **Separate capture persistence from orchestration (in progress).** Unique session-directory allocation, ROI/raw-stack/PNG/metadata writes, checker subprocess execution, and optional notebook generation now have focused storage/process boundaries. Notebook generation, phase checking, and overall save outcomes use typed result objects. Remaining work is further review of filesystem failure semantics and storage schemas.
4. **Make offline analysis import-safe (in progress).** The module imports without session processing, `analyze_capture(...)` returns a result bundle, and an `argparse` CLI preserves explicit/latest session selection, `--no-viewer`, and `--export`. A real explicit-session `--no-viewer --export` smoke test passed on a temporary capture copy; tests cover latest-session selection and the legacy missing-algorithm default. Remaining: consider moving more plotting/export orchestration behind explicit functions.
5. **Split the GUI after service boundaries stabilise (in progress).** Pico serial handling, manual frame saving and phase-capture saving are extracted as separate workers; ROI geometry/menu wiring, profile and phase-analysis widget construction, and application bootstrap are separate modules. Compatibility imports remain in `camera.py`. Remaining work is moving deeper analysis/ROI behavior in small slices while keeping `camera.py` as the launcher.

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
