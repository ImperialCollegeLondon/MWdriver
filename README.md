# MWdriver

A Thorlabs camera + Raspberry Pi Pico workflow for six-frame phase capture,
phase validation, and wavefront/aberration analysis. The Pico generates a
PWM-based triangle waveform for the phase shifter and hardware camera triggers;
the desktop application coordinates acquisition and saves captures for analysis.

## Files

| File | Role |
| --- | --- |
| `src/camera.py` | Napari/Qt desktop controller: live view, ROI selection, six-frame capture, trigger recalibration, and analysis launching. |
| `src/phase_capture_timestamps.py` | Timestamp payload and camera frame-timestamp normalization helpers used by the capture controller. |
| `src/phase_capture_frame_filter.py` | Frame acceptance helper used by the capture controller. |
| `src/phase_psi.py` | Shared six-frame PSI definitions and 90-degree / 60-degree wrapped-phase estimators. |
| `src/check_measured_vs_expected_phase.py` | Non-interactive FFT-based check against phase positions recorded in each capture session; saves JSON diagnostics and an annotated plot. |
| `src/phase_analysis_new.py` | Mode-selected six-frame phase reconstruction, unwrapping, and Zernike/aberration analysis, with saved images and an optional viewer/export. |
| `pypico/main.py` | Pico/MicroPython firmware: triangle-wave timing, six trigger pulses, and USB serial commands/status. |

## Setup

Create and install the desktop environment from the repository root:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python src\camera.py
```

The requirements install NumPy, Matplotlib, ImageIO, pySerial, pylablib,
Napari, QtPy, and the PySide6 Qt binding. Install the Thorlabs camera
drivers/SDK required by pylablib's `ThorlabsTLCamera` backend separately.
Offline analysis needs only NumPy and Matplotlib.
- The timestamp and frame-filter helper modules are included in `src/`.
   The optional notebook template `src/phase_shifts.ipynb` is not included.
- **Hardware:** a supported Thorlabs camera with external triggering, a Pico
  running MicroPython, and the phase-shifter drive/interface circuitry.
  Firmware pins are GP15 (PWM waveform), GP12 (camera trigger), and GP27 (buzzer).
  Use appropriate signal conditioning and compatible voltage levels; GP15 is
  PWM, not a direct analog output.
- Set `PICO_PORT` in `camera.py` to the Pico's serial port (default `COM3`);
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

3. Set the camera exposure and ROI, then choose **Start Phase Capture**.
   The controller temporarily uses external triggering, saves six frames under
   `src/captured_frames/phase_capture_*`, and automatically runs the phase checker.
   Sessions include a `*_raw_stack.npy`, PNG frames, and ROI/session metadata.
   Choose the phase-step method in the Camera Status controls. The default
   remains the 90-degree six-frame method; the alternative is the 60-degree
   six-bucket method. Recalibrate separately after switching methods because
   each method has its own target phase positions and saved trigger timings.
   Captures record the selected method in their metadata, which analysis uses
   automatically. Older captures without method metadata are treated as 90-degree.
4. Inspect a saved session manually or run full analysis:

   ```bash
   python src\check_measured_vs_expected_phase.py captured_frames\phase_capture_YYYYMMDD_HHMMSS
   python src\phase_analysis_new.py captured_frames\phase_capture_YYYYMMDD_HHMMSS
   ```

   With no arguments, either script discovers the latest capture. For
   non-interactive analysis and a NumPy export, supply the session first:

   ```bash
   python src\phase_analysis_new.py captured_frames\phase_capture_YYYYMMDD_HHMMSS --no-viewer --export results.npz
   ```

The checker writes `measured_phase_check.json` and
`measured_vs_expected_phase_annotated.png` into the session folder. Full analysis
saves intermediate PNGs and `phase_analysis_result_images/` there. Review
analysis constants for your optical setup, particularly `WAVELENGTH_NM`
(currently 632.8) in `phase_analysis_new.py`.
