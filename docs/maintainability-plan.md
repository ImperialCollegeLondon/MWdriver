# MWdriver Maintainability Plan

## Goal

Restructure the hardware-driven six-frame phase-shifting interferometry workflow around testable ownership boundaries, without changing camera/Pico timing, capture-session formats, offline command-line behavior, or compatibility with older captures.

## Current Progress

- **Implemented:** value-only trigger-position validation and coarse-measurement quality checks live in [`src/recalibration.py`](../src/recalibration.py), with the desktop controller delegating to them from [`src/camera.py`](../src/camera.py).
- **Implemented:** focused unit coverage for trigger rounding, bounds, ordering, spacing, measurement shape, finite values, span, and backtracking in [`tests/test_recalibration.py`](../tests/test_recalibration.py).
- **Verified:** the focused recalibration tests pass, and the changed Python files compile with the available macOS Python.
- **Not yet verified:** the full project test suite and hardware workflow. The current macOS environment lacks project dependencies; the Windows camera/Pico setup remains necessary for hardware validation.

## Work Plan

1. **Characterise existing contracts.** Add tests for frame-to-phase assignment, timestamps and metadata, capture paths and filenames, phase-check outputs, Pico command payloads, and recalibration acceptance/termination. Use fakes for camera, Pico, process, and filesystem boundaries.
2. **Finish extracting recalibration logic.** Move trigger spacing enforcement, phase-to-trigger interpolation, refinement, candidate scoring, and acceptance rules into Qt-free functions. Pass the camera-required trigger spacing and configuration explicitly. Keep capture scheduling, Pico acknowledgements, UI state, and persistence in the controller initially.
3. **Separate capture persistence from orchestration.** Centralise session path allocation and writes for raw stacks, PNGs, ROI data, and metadata. Isolate checker subprocess execution and optional notebook generation, and return a defined result object instead of an informal dictionary.
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
- On the supported Windows hardware setup, verify Pico acknowledgements, six external-trigger frame order, camera trigger-mode restoration, and a complete capture/recalibration cycle.

## Risks And Open Questions

- Hardware timing and native SDK frame metadata cannot be fully certified by unit tests or on a machine without the supported camera/Pico setup.
- The optional `phase_shifts.ipynb` template is documented as absent. Confirm whether generated per-capture notebooks remain a supported behavior before changing that path.
- Recalibration can delete rejected session directories and writes trigger positions during the workflow; preserve these semantics while extracting code.
- Napari integration uses private Qt APIs and a custom close handler. Treat compatibility work as a separate concern unless a module move requires it.
