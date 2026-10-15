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
| `src/mwdriver/domain/` | Hardware-independent PSI, calibration, ROI/profile geometry, streaming, visibility, and frame policies. |
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

   Use **Camera Tools > Start Stream** to allow live preview streaming and
   **Stop Stream** to disable it. The camera reads and displays live frames only
   while the **Camera Preview** layer is visible and streaming is allowed.
   During phase capture and recalibration, preview updates are suspended while
   the controller collects the six externally triggered frames; recalibration
   keeps preview streaming disabled between capture sets.

3. Set the camera exposure and ROI, then choose **Start Phase Capture**.
   The controller temporarily uses external triggering, saves six frames under
   `src/captured_frames/phase_capture_*`, and automatically runs the phase checker.
   Sessions include a `*_raw_stack.npy`, PNG frames, and ROI/session metadata.
   Each completed stack is also added to Napari as its own `time, y, x` image
   layer, with a time slider for stepping through the six frames. Select a
   captured layer to inspect it; use **Camera Tools > Return to Camera Preview**
   to resume the live preview. Captured layers remain available for the current
   Napari session while the original files continue to be saved to disk.
   To analyse a particular saved stack, select its layer and choose
   **Camera Tools > Analyze Selected Stack**. **Analyze Latest Capture** remains
   available when you want the most recently saved phase capture instead.
   Each completed analysis creates a separate results layer. Select an analysis
   layer to make the review controls refer to that run, then choose among its
   reconstructed phase and aberration results. The six captured phase images
   are available in the separate capture-stack layer, not in the analysis review
   selector.
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
(currently 632.8) in `src/mwdriver/analysis/phase_analysis_new.py`.
