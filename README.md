# MWdriver

A Thorlabs camera + Raspberry Pi Pico workflow for six-frame phase capture,
phase validation, and wavefront/aberration analysis. The Pico generates a
PWM-based triangle waveform for the phase shifter and hardware camera triggers;
the desktop application coordinates acquisition and saves captures for analysis.

## Files

| File | Role |
| --- | --- |
| `camera.py` | Napari/Qt desktop controller: live view, ROI selection, six-frame capture, trigger recalibration, and analysis launching. |
| `check_measured_vs_expected_phase.py` | Non-interactive FFT-based check against expected phases of 0, 90, 180, 270, 360, and 450 degrees; saves JSON diagnostics and an annotated plot. |
| `phase_analysis_new.py` | Six-frame phase reconstruction, unwrapping, and Zernike/aberration analysis, with saved images and an optional viewer/export. |
| `pypico.py` | Pico/MicroPython firmware: triangle-wave timing, six trigger pulses, and USB serial commands/status. |

## Setup

- **Desktop:** Python with `numpy`, `matplotlib`, `imageio`, `pyserial`,
  `pylablib`, `napari`, and `qtpy`, plus a compatible Qt binding for Napari.
  Install the Thorlabs camera drivers/SDK required by pylablib's
  `ThorlabsTLCamera` backend. Offline analysis needs only NumPy and Matplotlib.
- **Checkout limitation:** `camera.py` imports `phase_capture_timestamps.py`
  and `phase_capture_frame_filter.py`, which are not included in this repository.
  Supply these project modules on the Python import path before running the GUI.
  The optional notebook template `phase_shifts.ipynb` is also not included.
- **Hardware:** a supported Thorlabs camera with external triggering, a Pico
  running MicroPython, and the phase-shifter drive/interface circuitry.
  Firmware pins are GP15 (PWM waveform), GP12 (camera trigger), and GP27 (buzzer).
  Use appropriate signal conditioning and compatible voltage levels; GP15 is
  PWM, not a direct analog output.
- Set `PICO_PORT` in `camera.py` to the Pico's serial port (default `COM3`);
  `PICO_BAUDRATE` defaults to 115200.

## Workflow

1. Copy `pypico.py` to the Pico using a MicroPython-capable tool, such as Thonny.
   Run it on the Pico, not with desktop Python. For automatic startup after
   reset, create a **device-side** `main.py` containing `import pypico`.
   The repository firmware entrypoint remains `pypico.py`.
2. Connect the waveform/trigger circuitry and camera. Close any serial monitor
   so the controller can open the Pico port, then launch the desktop GUI:

   ```bash
   python camera.py
   ```

3. Set the camera exposure and ROI, then choose **Start Phase Capture**.
   The controller temporarily uses external triggering, saves six frames under
   `captured_frames/phase_capture_*`, and automatically runs the phase checker.
   Sessions include a `*_raw_stack.npy`, PNG frames, and ROI/session metadata.
4. Inspect a saved session manually or run full analysis:

   ```bash
   python check_measured_vs_expected_phase.py captured_frames/phase_capture_YYYYMMDD_HHMMSS
   python phase_analysis_new.py captured_frames/phase_capture_YYYYMMDD_HHMMSS
   ```

   With no arguments, either script discovers the latest capture. For
   non-interactive analysis and a NumPy export, supply the session first:

   ```bash
   python phase_analysis_new.py captured_frames/phase_capture_YYYYMMDD_HHMMSS --no-viewer --export results.npz
   ```

The checker writes `measured_phase_check.json` and
`measured_vs_expected_phase_annotated.png` into the session folder. Full analysis
saves intermediate PNGs and `phase_analysis_result_images/` there. Review
analysis constants for your optical setup, particularly `WAVELENGTH_NM`
(currently 632.8) in `phase_analysis_new.py`.
