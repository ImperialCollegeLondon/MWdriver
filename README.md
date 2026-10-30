# MWdriver

A Thorlabs camera + Raspberry Pi Pico workflow for six-frame phase capture,
phase validation, and wavefront/aberration analysis. The Pico generates a
PWM-based triangle waveform for the phase shifter and hardware camera triggers;
the desktop application coordinates acquisition and saves captures for analysis.

## Files

| File | Role |
| --- | --- |
| `src/camera.py` | Desktop launcher, compatibility facade, and composition root for the Napari/Qt viewer. |
| `src/check_measured_vs_expected_phase.py` | Phase-check CLI entry point used by saved-capture workflows. |
| `src/phase_analysis_new.py` | Offline phase-analysis CLI entry point. |
| `src/mwdriver/domain/` | Hardware-independent PSI, calibration, ROI geometry, streaming, visibility, and frame policies. |
| `src/mwdriver/hardware/` | Native Thorlabs camera adapter and Pico serial worker. |
| `src/mwdriver/capture/` | Phase-capture lifecycle, frame/timestamp policy, workers, persistence, and checker process. |
| `src/mwdriver/analysis/` | Phase reconstruction, unwrapping, Fourier, aberration, plotting, and result-viewing implementations. |
| `src/mwdriver/ui/` | Napari/Qt controllers, widget builders, application bootstrap, and worker lifecycle. |
| `pypico/main.py` | Pico/MicroPython firmware: triangle-wave timing, six trigger pulses, and USB serial commands/status. |

## Setup

### 1. Install Thorlabs Camera Drivers

Use 64-bit Windows and 64-bit Python for the setup below. Camera acquisition
uses Thorlabs' official `thorlabs_tsi_sdk` interface. The Windows camera
drivers, native SDK DLLs, and Python wrapper are separate requirements;
`pip install -r requirements.txt` does not install the drivers or SDK wrapper.

1. Go to your camera's product page on [Thorlabs](https://www.thorlabs.com/)
   and open its **Software** tab. Download the ThorCam installer appropriate
   for your camera model.
2. Run the installer and install the camera drivers and scientific camera
   support/SDK components provided for that model. Accept Windows driver
   installation prompts and restart if the installer requests it.
3. Connect the camera, open ThorCam, and confirm that it is detected and can
   display live images. If ThorCam cannot acquire images, resolve the USB or
   driver issue before continuing with Python.
4. Close ThorCam before running MWdriver. Only one application should have
   the camera open at a time.

The native backend has been tested with a CS165MU1 camera. Use the vendor's
SDK documentation to check support for other camera models.

### 2. Extract the Native SDK and Python Package

With the standard ThorCam installation, the SDK archive is located at:

```text
C:\Program Files\Thorlabs\Scientific Imaging\Scientific Camera Support\Scientific_Camera_Interfaces.zip
```

Extract it to a writable directory, for example `C:\ThorlabsSDK`. The archive
contains a `Scientific Camera Interfaces` folder. The following examples
assume this extracted layout:

```text
C:\ThorlabsSDK\Scientific Camera Interfaces\
    Python_README.txt
    SDK\
        Python Toolkit\
            thorlabs_tsi_camera_python_sdk_package.zip
        Native Toolkit\
            dlls\
                Native_64_lib\
                    thorlabs_tsi_camera_sdk.dll
                    ... companion DLLs ...
```

If your installer uses a different location or layout, follow its
`Python_README.txt` and adjust the paths below. Keep the Python package ZIP
intact for installation with pip; do not install an unrelated similarly
named package from PyPI.

### 3. Install the Python Environment

Run these PowerShell commands from the repository root:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install "C:\ThorlabsSDK\Scientific Camera Interfaces\SDK\Python Toolkit\thorlabs_tsi_camera_python_sdk_package.zip"
.\.venv\Scripts\python.exe -m pip check
```

The requirements install NumPy, Matplotlib, ImageIO, pySerial, Napari,
QtPy, and the PySide6 Qt binding. The vendor ZIP installs `thorlabs_tsi_sdk`
into the same virtual environment. No environment activation is needed for
these commands, avoiding PowerShell activation-policy issues.

MWdriver no longer needs pylablib, PyQt5, pyqtgraph, pyft232, pyvisa, pyusb,
rpyc, plumbum, numba, or llvmlite. Installing the current requirements into
an existing environment does not uninstall old packages. A fresh environment
is the simplest way to avoid legacy dependencies. When cleaning an existing
environment, retain any packages needed by other software: NumPy, SciPy,
pandas and pySerial are still required by the current application stack.

### 4. Configure Native DLLs and Launch

The backend automatically searches these standard installation directories,
in order, for `thorlabs_tsi_camera_sdk.dll`:

```text
C:\Program Files\Thorlabs\Scientific Imaging\ThorCam
C:\Program Files\Thorlabs\ThorImageCAM\Bin
```

If the native DLLs are available there, launch directly:

```powershell
.\.venv\Scripts\python.exe src\camera.py
```

Otherwise, or to select a specific SDK version, set `THORLABS_TSI_DLL_DIR`
to the extracted native DLL directory before launching:

```powershell
$env:THORLABS_TSI_DLL_DIR = "C:\ThorlabsSDK\Scientific Camera Interfaces\SDK\Native Toolkit\dlls\Native_64_lib"
.\.venv\Scripts\python.exe src\camera.py
```

This variable overrides automatic discovery and applies to the current
PowerShell session. Point it at the directory containing the DLLs, not the
SDK root or the DLL file itself. Keep all companion DLLs together; copying
only `thorlabs_tsi_camera_sdk.dll` is not sufficient. DLL architecture must
match Python: `Native_64_lib` for 64-bit Python, `Native_32_lib` for 32-bit
Python. Do not mix DLLs from different SDK releases or copy them into Windows
system directories.

### Troubleshooting

Persistent diagnostic logging is disabled during normal desktop launches.
The diagnostic helper remains available for troubleshooting; when explicitly
enabled with `configure_diagnostics()`, it writes to `Documents/MWDriver/logs`.
The `camera_*.log` file records layer transitions, Qt warnings and unhandled
Python exceptions. Its companion `camera_*_fault.log` records Python thread
stacks for native faults where Python's fault handler can run; an empty fault
file is normal when no fault was captured. For a silent crash, retain both
files from the affected run. Not every graphics-driver or forced process exit
can produce a fault trace.

| Symptom | Check |
| --- | --- |
| `No module named 'thorlabs_tsi_sdk'` or a request to install the official SDK | Install the vendor Python package ZIP using `.venv\Scripts\python.exe -m pip`, not a different Python interpreter. |
| `Thorlabs native camera DLLs were not found` | Install ThorCam or set `THORLABS_TSI_DLL_DIR` to the native DLL directory. An incorrect override prevents automatic discovery. |
| DLL load failure, `WinError 126`, or `WinError 193` | Check that all companion DLLs are present and that SDK DLL architecture matches Python. Follow the vendor installer instructions for any required runtime components. |
| No camera detected or camera cannot be opened | Close ThorCam and other camera applications, check USB and power, and confirm acquisition works in ThorCam first. |

Check the software environment without starting camera acquisition:

```powershell
.\.venv\Scripts\python.exe -c "from thorlabs_tsi_sdk.tl_camera import TLCameraSDK; print('Thorlabs Python wrapper import OK')"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The import check confirms the Python wrapper is installed; it does not prove
that native DLLs load or that a camera is connected. Backend tests use a
simulated camera. Live view and live/external/live switching have also been
checked on the CS165MU1; a full six-pulse capture must be verified with your
Pico and trigger wiring.

### Acquisition and Hardware Notes

Live view uses unlimited software-triggered acquisition; phase capture uses
one frame per hardware trigger. Native frame buffers are copied before the
next poll. Saved camera timestamps are SDK-relative nanoseconds, with
`timestamp_clock_frequency_hz` set to 1,000,000,000 to match those units;
they are not host wall-clock timestamps.

Offline analysis needs only NumPy and Matplotlib.
- The optional notebook template `src/phase_shifts.ipynb` is not included.
- **Hardware:** a supported Thorlabs camera with external triggering, a Pico
   running MicroPython, and the phase-shifter drive/interface circuitry.
   Firmware pins are GP15 (PWM waveform), GP12 (camera trigger), and GP27 (buzzer).
   Use appropriate signal conditioning and compatible voltage levels; GP15 is
   PWM, not a direct analogue output.
- Set `PICO_PORT` in `camera.py` to the Pico's serial port (default `COM4`);
  `PICO_BAUDRATE` defaults to 115200.

## Workflow

1. Copy `pypico/main.py` to the Pico using a MicroPython-capable tool, such as
   Thonny. Run it on the Pico, not with desktop Python. The source file is
   already named `main.py`, so placing it on the device as `main.py` makes it
   run automatically after reset.
2. Connect the waveform/trigger circuitry and camera. Close any serial monitor
   so the controller can open the Pico port, then launch the desktop GUI:

   ```bash
   python src\camera.py
   ```

   MWdriver controls appear in a separate, scrollable **MWdriver Controls**
   dock on the right; Napari's layer list remains separate on the left.

   Use **Camera > Start Stream** to allow live preview streaming and
   **Stop Stream** to disable it. The camera reads and displays live frames only
   while the **Camera Preview** layer is visible and streaming is allowed.
   During normal phase capture, preview redraws are suspended while the
   controller collects six externally triggered frames. Recalibration displays
   each acquired frame, but keeps live streaming disabled between capture sets.

3. Set the camera exposure and ROI, then choose **Start Phase Capture**.
   The controller temporarily uses external triggering, saves six frames under
   `Documents/MWDriver/measure_YYYYMMDD_HHMMSS`, and automatically runs the phase checker.
   Windows' actual Documents location is used, including redirected or OneDrive
   Documents folders. Calibration captures, including verification runs, use
   `calibrate_YYYYMMDD_HHMMSS`.
   On calibration completion, the final completed verification capture is retained,
   rather than an earlier best-scoring capture;
   the coarse capture and rejected verification captures are removed. If
   calibration aborts before selecting a verification capture, diagnostic data
   remains available.
   Enable **Analysis > Extra Diagnostic Plots** to save the optional plots below.
   This checkable session setting is off by default and applies to subsequent
   captures. It controls combined calibration plots, Fourier spectrum/mask plots,
   field amplitude/relative-phase maps, and phase-versus-voltage plots. The
   standard measured-versus-expected plot and phase-check JSON are always saved.
   Disabling the option does not delete existing diagnostic files.
   With diagnostics enabled during recalibration, `recalibration_iterations.png` is updated in each capture
   folder with the coarse and all verification curves on shared measured-versus-
   expected phase axes. Curves are labelled by attempt and verification number,
   including automatic restarts. The final retained calibration folder contains
   the combined plot even though earlier capture folders are removed.
   A companion `recalibration_iteration_errors.png` shows measured minus expected
   phase for the same iterations, with matching curve colours and a zero-error
   reference. Both plots reference measured phases to the first frame.
   `recalibration_iterations_time_vs_phase.png` overlays all calibration scans
   with measured phase on the horizontal axis and requested trigger time in
   milliseconds from the falling-edge start on the vertical axis. These are the
   requested timing positions, not camera timestamps or estimated voltages.
   Saved ROI files and `phase_trigger_positions.json` also live in
   `Documents/MWDriver`. The directory is created when results are first saved.
   Sessions include a `*_raw_stack.npy`, PNG frames, and ROI/session metadata.
   Each completed stack is also added to Napari as its own `time, y, x` image
   layer, with a time slider for stepping through the six frames. Select a
   captured layer to inspect it; use **Camera > Return to Camera Preview**
   to resume the live preview. Captured layers remain available for the current
   Napari session while the original files continue to be saved to disk.
   To analyse a particular saved stack, select its layer and choose
   **Analysis > Analyze Selected Stack**. **Analyze Latest Capture** remains
   available when you want the most recently saved phase capture instead.
   Use **Analysis > Reload Stack...** to choose an existing acquired dataset
   directory. It loads the directory's single `*_raw_stack.npy` with capture
   metadata and, if present, `phase_analysis_results.npz` without recalculation.
   Saved analysis opens for review while the raw stack remains in the layer list.
   Without saved analysis, the raw stack is shown ready for analysis. Current
   camera settings and ROI are preserved.
   To reopen a dataset from an earlier run, choose **File > Reload Dataset...**
   and select its `*_raw_stack.npy` file. The six-frame stack and adjacent capture
   metadata are restored as a new layer; choose **Analyze Selected Stack** to
   analyse it using the current ROI. Loading does not change camera settings or
   replace the current ROI. Older stacks without metadata default to the
   90-degree phase-step method.
   To review existing analysis without recalculating, choose **File > Reload
   Analysis...** and select `phase_analysis_results.npz`. This creates a new
   analysis view with the saved images, charts and available interactive 3D
   surfaces. It does not reload the raw capture stack or change the current ROI
   or camera settings; use **Reload Dataset...** separately when needed.
   If the managed ROI layer has been deleted, **ROI > Reload ROI** recreates it
   and restores the last separately saved ellipse from `Documents/MWDriver`.
   The restored layer reconnects mask updates and opens in 2D selection mode.
   This action does not import an ROI from a capture or analysis archive.
   Each completed analysis creates a separate results layer. Select an analysis
   layer to make the review controls refer to that run, then choose among its
   reconstructed phase and aberration results. The six captured phase images
   are available in the separate capture-stack layer, not in the analysis review
   selector.
   RGB analysis plots use `spline36` interpolation automatically, with the
   interpolation control hidden. Scalar phase and aberration image layers keep
   their existing interpolation defaults.
   The three saved 3D wavefront panels share vertical and colour limits from
   the 0.1st and 99.1st percentiles of the tilt-removed wavefront in nm OPD.
   Interactive 3D surfaces use the same limits for colour and the same span
   for height scaling, while retaining their original mesh heights. Masked
   and non-finite values are excluded from the percentile calculation.
   Choose the phase-step method in the Camera Status controls. The default
   remains the 90-degree six-frame method; the alternative is the 60-degree
   six-bucket method. Recalibrate separately after switching methods because
   each method has its own target phase positions and saved trigger timings.
   Coarse recalibration starts at 100 ms for both methods. The 90-degree method
   samples at `100, 190, 280, 369, 459, 549 ms`; the 60-degree method uses
   75% of that spacing: `100, 167, 235, 302, 369, 437 ms`.
   Verification uses 50% of each calculated timing adjustment, with the existing
   per-step timing limits and camera-safe spacing. Up to six verification
   captures are allowed before the restart decision, with at most one automatic
   restart. Two consecutive captures within +/-5 degrees confirm calibration;
   at the limit, the final completed capture is retained even if it is not a pass.
   Captures record the selected method in their metadata, which analysis uses
   automatically. Older captures without method metadata are treated as 90-degree.
4. Inspect a saved session manually or run full analysis:

   ```powershell
   $results = Join-Path ([Environment]::GetFolderPath('MyDocuments')) 'MWDriver'
   $session = Join-Path $results 'measure_YYYYMMDD_HHMMSS'
   python src\check_measured_vs_expected_phase.py "$session"
   python src\phase_analysis_new.py "$session"
   ```

   With no arguments, either script discovers the latest measurement in
   `Documents/MWDriver`, excluding calibration sessions. For
   non-interactive analysis and a NumPy export, supply the session first:

   ```powershell
   python src\phase_analysis_new.py "$session" --no-viewer --export (Join-Path $session 'phase_analysis_results.npz')
   ```

The standalone phase checker enables diagnostic plots by default. Pass
`--no-diagnostic-plots` to suppress the optional Fourier and field-map outputs.
The application's **Analysis > Extra Diagnostic Plots** setting controls those
outputs for automatic capture checks and is off by default.

The checker extracts the complex field using the Fourier half-plane containing
the detected sideband. Before the FFT, each frame has its mean ROI intensity
subtracted inside the supplied ROI, and all pixels outside the ROI are set to
zero. A full-frame ROI uses one mean. Without a supplied ROI, the checker creates
a central 400-by-400-pixel ROI (smaller for small images) and applies exactly
the same mean subtraction and outside-zeroing before the FFT.
The same ROI also selects pixels for complex-field phase correlation after
filtering. The hard ROI edge can introduce spectral leakage at the boundary.

The Fourier filter retains the half-plane containing
the detected sideband. The boundary passes through DC and is perpendicular to
the DC-to-sideband direction. A Gaussian high-pass factor,
`1 - exp(-r^2 / (2 * sigma^2))`, suppresses the zero order within that half-plane.
Its sigma equals the carrier detector's DC exclusion radius in Fourier bins;
there is no longer a narrow Gaussian passband centred on the sideband.
The wider passband also admits more noise and any other structure on that side.

With diagnostics enabled, the checker also saves `fourier_spectrum_and_mask.png`: frame 0's log Fourier
spectrum, a zoom with filter-weight contours, the full actual filter weights,
and the filtered spectrum. DC and the selected sideband are marked, with
colourbars and the DC-suppression sigma in Fourier bins. The phase-check JSON
records `filter_type` and `dc_suppression_sigma` in its Fourier diagnostics.

The checker always writes `measured_phase_check.json`. With diagnostics enabled,
it also writes `field_amplitude_maps.png` and `relative_phase_maps.png` into the session folder. These six-frame montages
use shared colourbars: FFT sideband
amplitude in camera intensity units, and wrapped phase relative to frame 0 in
degrees (-180 to 180). Maps cover the full frame; phase in low-amplitude
background regions is unreliable. The phase-map sign follows the measured
phase curve. The existing annotated plot is saved as
`measured_vs_expected_phase_annotated.png` into the session folder. Full analysis
saves intermediate PNGs and `phase_analysis_result_images/` there. Review
analysis constants for your optical setup, particularly `WAVELENGTH_NM`
(currently 632.8) in `src/mwdriver/analysis/phase_analysis_new.py`.

Existing captures and settings in the repository are not moved automatically.
Old captures can still be opened with an explicit session path. To retain a
saved ROI or trigger calibration, copy `roi_params.json`, `roi_mask.npy`, and
`phase_trigger_positions.json` into `Documents/MWDriver` before using them.
Legacy `phase_capture_*` folders copied into that directory remain discoverable.

## Saved Results

The desktop application saves user data outside the source tree:

```text
Documents/MWDriver/
   roi_params.json
   roi_mask.npy
   phase_trigger_positions.json
   calibrate_YYYYMMDD_HHMMSS/
   measure_YYYYMMDD_HHMMSS/
      measure_YYYYMMDD_HHMMSS_raw_stack.npy
      measure_YYYYMMDD_HHMMSS_metadata.json
      phase_0_YYYYMMDD_HHMMSS.png
      ...
      measured_phase_check.json
      measured_vs_expected_phase_annotated.png
      phase_analysis_results.npz
      phase_intermediate_*.png
      phase_analysis_result_images/
```

Analysis outputs appear after analysis is run. The GUI writes
`phase_analysis_results.npz` automatically; the CLI writes an NPZ only when
`--export` is supplied. Explicit CLI export paths are honoured. Intermediate
diagnostics include phase numerator, denominator, magnitude and unwrapping
stages; duplicate `phase_intermediate_01_input_*` images are no longer generated.
The original six intensity frames remain in the raw stack and capture PNGs,
not in the results NPZ. There is no separate manual-frame export workflow.

### Phase and Aberration Analysis

Scalar phase images and interactive 3D surfaces show a colourbar in the upper
right of the Napari canvas. It follows the selected layer's colormap and contrast
limits. Wrapped and unwrapped phase images display waves; tilt-removed images,
wavefront results and interactive surfaces display nanometres (OPD). A matching
unit label appears at the lower right, and units are also included in the layer
name. Stored wrapped, unwrapped and tilt-removed arrays remain in radians;
conversion is display-only and uses the exported wavelength (632.8 nm for older
archives without a wavelength). RGB charts keep their embedded scales; returning to
camera preview hides analysis colourbars.

When analysis runs without a defined ROI, it derives a circular pupil enclosing
the largest 8-connected region of valid PSI-data pixels. Detached noise patches
do not expand the circle, and valid pixels outside it are excluded from analysis.
The circle is centred on the main region's centroid, with its
radius reaching the furthest valid pixel plus a half-pixel margin. This is an
enclosing circle, not a minimum-enclosing-circle optimisation. Zernike fitting
uses this circle immediately, while invalid data pixels remain excluded.
The export contains its raw-pixel `automatic_roi_vertices` and circular
`roi_mask`. When background analysis completes, the camera viewer adds the circle
to the editable ROI layer if no ROI has been drawn in the meantime. Subsequent
analysis uses that ROI and pupil geometry. Use **ROI > Save ROI** to persist it;
existing user-defined ROIs and archive-only reload behaviour are unchanged.

The Zernike fit includes **J1-J11**, ending at primary spherical. The composition
plot reports astigmatism, coma, trefoil and primary spherical after piston,
tilt and defocus removal. Coefficients are expressed in waves at the configured
wavelength, currently **632.8 nm**.

Analysis charts and intermediate diagnostic figures use a **1440 x 1080**
canvas with antialiased text and larger horizontal coefficient labels. Scalar
result images retain the input camera dimensions.

`phase_analysis_result_images/wavefront_surfaces.png` compares tilt removed,
tilt and defocus removed, and residual in three opaque 3D surface panels.
All panels use wavefront OPD in nanometres, identical z-axis limits and one
shared `viridis` colour bar. The axes are restricted to valid-pixel bounds;
masked and non-finite data are excluded. Mesh faces crossing invalid pixels
are omitted, and the display mesh is sampled to at most 100 rows and columns.
The numerical result arrays are not downsampled.

For interactive viewing in Napari, select **Interactive 3D: Tilt removed**,
**Interactive 3D: Tilt and defocus removed**, or **Interactive 3D: Residual
wavefront** in the analysis review selector. These choices switch a Surface
layer between the three correction stages; drag to rotate and scroll to zoom.
The surfaces use `viridis` with a shared OPD colour range in nanometres and
exclude triangles crossing invalid pixels. Height is visually scaled relative
to the pixel coordinates for inspection, using the same height scale across
the three stages. The interactive mesh uses full-resolution pixels within the
valid-data bounds, with no display downsampling; large ROIs require more graphics
memory and may rotate more slowly. Selecting an image result or returning to camera
preview restores 2D viewing. New NPZ exports also record `wavelength_nm`; older
exports default to 632.8 nm when converting tilt-removed phase to OPD.

Image, RGB plot and Surface layers are retained separately for each analysis
run and reused when switching views. Inactive layers are hidden rather than
deleted, avoiding an OpenGL texture-deletion crash during repeated 2D/3D
transitions. The selected analysis layer is placed above the ROI layer; the
ROI is hidden in surface views and brought forward again for ROI editing.
Selecting a captured six-frame stack restores a time slider limited to frames
0-5, independent of the height range of retained surfaces.

Switching analysis views exclusively selects the target layer and refreshes
its controls. Surface views expose shading, opacity and blending as well as
colormap and contrast limits. Explicitly deleting a surface first returns the
viewer to 2D and makes its OpenGL canvas current before releasing resources.

The GUI passes the current ROI to analysis. For CLI analysis, pass
`--roi-mask` explicitly to apply a saved Boolean ROI; without it, the spatial
ROI defaults to the full frame. Pixels must also pass the PSI signal-validity
filter, so the valid analysis area may be smaller than the ROI.

### Results NPZ Contents

| Key | Contents and units |
| --- | --- |
| `roi_mask` | Boolean original spatial ROI; `True` means inside. This is not the signal-filtered analysis mask. |
| `algorithm_id` | PSI algorithm identifier. |
| `expected_phase_deg` | Six expected phase positions in degrees. |
| `wavelength_nm` | Analysis wavelength in nanometres. |
| `wrapped` | Wrapped phase in radians. |
| `unwrapped` | Unwrapped phase in radians. |
| `tilt_removed` | Phase after piston and tilt removal, in radians. |
| `wavefront_aberration_nm` | Wavefront OPD after piston, tilt and defocus removal, in nanometres. |
| `wavefront_aberration_rms_nm` | RMS of that wavefront, in nanometres. |
| `wavefront_aberration_pv_nm` | Robust PV: 99.9th minus 0.1st percentile, in nanometres. |
| `residual_nm` | Residual surface error after additionally removing astigmatism and primary spherical, in nanometres. |
| `zernike_coefficients_before_correction_waves` | JSON dictionary of signed J1-J11 coefficients before correction, in waves. |
| `zernike_coefficients_after_correction_waves` | JSON dictionary of signed J1-J11 coefficients after correction, in waves. |
| `aberration_component_waves` | JSON dictionary of non-negative astigmatism, coma, trefoil and primary spherical magnitudes, in waves. |
| `zernike_before_plot_rgb`, `zernike_after_plot_rgb` | Rendered coefficient charts. |
| `aberration_composition_plot_rgb` | Rendered aberration composition chart. |
| `wavefront_surfaces_plot_rgb` | Rendered three-panel 3D wavefront comparison. |

Numerical image arrays are `float32`, with excluded pixels represented by
`NaN`. Rendered charts are `uint8` RGB arrays of shape `(1080, 1440, 3)`.
The residual array uses surface error, whereas the 3D comparison converts it
to wavefront OPD by multiplying by two.

Coefficient dictionaries are stored as JSON strings rather than pickled Python
objects, so the archive can be opened with `allow_pickle=False`:

```python
import json
from pathlib import Path

import numpy as np

session = Path(r"C:\path\to\Documents\MWDriver\measure_YYYYMMDD_HHMMSS")
with np.load(session / "phase_analysis_results.npz", allow_pickle=False) as archive:
   results = {name: archive[name] for name in archive.files}

coefficients = json.loads(
   results["zernike_coefficients_after_correction_waves"].item()
)
print("Primary spherical (waves):", coefficients["J11 Primary Spherical"])
roi_mask = results["roi_mask"]
wavefront_nm = np.ma.masked_invalid(results["wavefront_aberration_nm"])
```

Older NPZ files may contain positional coefficient arrays instead of JSON
dictionaries and may include `phase_0` through `phase_5` rather than an ROI
mask. Rerun analysis to generate the current export format.
