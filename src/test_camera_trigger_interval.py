"""Measure six-pulse hardware capture reliability using the existing Pico firmware."""

import ast
import json
import re
import time
from pathlib import Path

import serial

from mwdriver.hardware.thorlabs_native_camera import NativeThorlabsCamera


def send_command(port, command):
    port.write((command + "\n").encode("ascii"))
    port.flush()


def set_positions(port, positions):
    send_command(port, "SET_TRIGGER_POSITIONS_MS:" + ",".join(map(str, positions)))
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        line = port.readline().decode(errors="replace").strip()
        if line.startswith("TRIGGER_POSITIONS_UPDATED_MS:"):
            actual = ast.literal_eval(line.split(":", 1)[1].strip())
            if actual != positions:
                raise RuntimeError(f"Pico acknowledged different positions: {actual}")
            return
        if "TRIGGER_POSITIONS_UPDATE_ERROR:" in line:
            raise RuntimeError(line)
    raise TimeoutError("Pico did not acknowledge trigger positions.")


def capture_run(camera, port):
    camera.start_acquisition(1)
    send_command(port, "PC_START_PHASE")
    deadline = time.monotonic() + 12.0
    done_time = None
    metadata = []
    messages = []
    try:
        while time.monotonic() < deadline:
            image_range = camera.get_new_images_range()
            if image_range is not None:
                _, infos = camera.read_multiple_images(image_range, return_info=True)
                metadata.extend(infos)
            line = port.readline().decode(errors="replace").strip()
            if line:
                messages.append(line)
                if "PHASE_CAPTURE_ERROR:" in line or "ignored:" in line:
                    raise RuntimeError(line)
                if line == "TRIANGLE_DONE":
                    done_time = time.monotonic()
            if done_time is not None and time.monotonic() - done_time >= 0.75:
                break
        if done_time is None:
            raise TimeoutError("Pico did not finish its triangle cycle.")
    finally:
        camera.stop_acquisition()
    positions = None
    for line in messages:
        if line.startswith("trigger_positions_ms:"):
            positions = ast.literal_eval(line.split(":", 1)[1].strip())
    pulse_lateness_us = [
        int(match.group(1))
        for line in messages
        if (match := re.search(r"actual_lateness_us:(-?\d+)$", line))
    ]
    timestamps = [info["camera_timestamp"] for info in metadata]
    intervals_ms = [
        (current - previous) / 1_000_000.0
        for previous, current in zip(timestamps, timestamps[1:])
        if previous is not None and current is not None
    ]
    return {
        "frame_count": len(metadata),
        "frames": metadata,
        "camera_intervals_ms": intervals_ms,
        "trigger_positions_ms": positions,
        "pulse_lateness_us": pulse_lateness_us,
        "serial_messages": messages,
    }


def main():
    results = []
    original_positions = None
    report_path = Path(__file__).resolve().parents[1] / "trigger_interval_results.json"
    with serial.Serial("COM4", 115200, timeout=0.005) as port:
        camera = NativeThorlabsCamera()
        native = camera._camera
        original_settings = {
            "exposure_time_us": native.exposure_time_us,
            "operation_mode": native.operation_mode,
            "trigger_polarity": native.trigger_polarity,
            "frames_per_trigger_zero_for_unlimited": native.frames_per_trigger_zero_for_unlimited,
        }
        report = {
            "camera": camera.get_device_info(),
            "sensor_readout_time_ns": native.sensor_readout_time_ns,
            "original_exposure_time_us": original_settings["exposure_time_us"],
            "runs": results,
        }
        try:
            camera.set_trigger_mode("ext")
            camera.setup_ext_trigger("rise")
            baseline = capture_run(camera, port)
            original_positions = baseline["trigger_positions_ms"]
            report["baseline"] = baseline
            if original_positions is None or len(baseline["pulse_lateness_us"]) != 6:
                raise RuntimeError("Cannot verify original Pico configuration and six pulses.")
            print(f"Baseline: {baseline['frame_count']}/6, original positions {original_positions}", flush=True)
            for exposure_ms in (10, 40):
                camera.set_exposure(exposure_ms / 1000.0)
                for gap_ms in (30, 35, 40, 45, 55, 75, 90):
                    positions = [100 + index * gap_ms for index in range(6)]
                    set_positions(port, positions)
                    for repeat in range(2):
                        result = capture_run(camera, port)
                        result.update({
                            "exposure_ms": exposure_ms,
                            "actual_exposure_us": native.exposure_time_us,
                            "gap_ms": gap_ms,
                            "repeat": repeat + 1,
                        })
                        results.append(result)
                        print(
                            f"Exposure {exposure_ms} ms, gap {gap_ms} ms, "
                            f"run {repeat + 1}: {result['frame_count']}/6; "
                            f"camera intervals {result['camera_intervals_ms']}",
                            flush=True,
                        )
                        if len(result["pulse_lateness_us"]) != 6:
                            raise RuntimeError("Pico did not report all six actual trigger pulses.")
        finally:
            try:
                if original_positions is not None:
                    set_positions(port, original_positions)
                    report["pico_positions_restored"] = True
            finally:
                try:
                    camera.stop_acquisition()
                    for name, value in original_settings.items():
                        setattr(native, name, value)
                    report["camera_settings_restored"] = True
                finally:
                    camera.close()
                    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
                    print(f"Report: {report_path}", flush=True)


if __name__ == "__main__":
    main()