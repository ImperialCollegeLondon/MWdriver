"""Qt worker for Pico serial commands and capture lifecycle messages."""

import queue
import time

import serial
from qtpy.QtCore import QObject, Signal

from recalibration import DEFAULT_RECALIBRATION_CONFIG


PHASE_CAPTURE_COUNT = DEFAULT_RECALIBRATION_CONFIG.frame_count


class PicoSerialWorker(QObject):
    phase_session_started = Signal()
    phase_requested = Signal(int)
    phase_session_done = Signal()

    capture_requested = Signal()

    status_ready = Signal(str)
    message_ready = Signal(str)

    def __init__(self, port="COM3", baudrate=115200, reboot_pico=True):
        super().__init__()

        self.port = port
        self.baudrate = baudrate
        self.reboot_pico = reboot_pico
        self.running = False
        self.ser = None
        self.command_queue = queue.Queue()

    def request_start_phase_from_gui(self):
        try:
            self.command_queue.put("PC_START_PHASE")
            self.message_ready.emit("Queued command to Pico: PC_START_PHASE")
        except Exception as error:
            self.message_ready.emit(f"Queue Pico command error: {error}")

    def request_set_trigger_positions_ms(self, trigger_positions_ms):
        try:
            serialized = ",".join(
                str(int(round(value)))
                for value in trigger_positions_ms
            )
            command = f"SET_TRIGGER_POSITIONS_MS:{serialized}"
            self.command_queue.put(command)
            self.message_ready.emit(
                f"Queued trigger positions to Pico: {serialized}"
            )
        except Exception as error:
            self.message_ready.emit(
                f"Queue trigger positions command error: {error}"
            )

    def process_pending_commands(self):
        if self.ser is None:
            return

        while True:
            try:
                command = self.command_queue.get_nowait()
            except queue.Empty:
                break

            try:
                self.ser.write((command + "\n").encode("utf-8"))
                self.ser.flush()
                self.message_ready.emit(f"Sent to Pico: {command}")
            except Exception as error:
                self.message_ready.emit(f"Pico command send error: {error}")

    def handle_serial_line(self, line):
        if line == "TRIANGLE_START":
            self.phase_session_started.emit()
        elif line.startswith("CAPTURE_PHASE_"):
            try:
                phase_index = int(line.split("_")[-1])
                if 0 <= phase_index < PHASE_CAPTURE_COUNT:
                    self.phase_requested.emit(phase_index)
                else:
                    self.message_ready.emit(
                        f"Pico phase index out of range: {phase_index}"
                    )
            except Exception as error:
                self.message_ready.emit(f"Pico phase parse error: {error}")
        elif line == "TRIANGLE_DONE":
            self.phase_session_done.emit()
        elif line == "START_CAPTURE":
            self.capture_requested.emit()

    def run(self):
        self.running = True

        try:
            self.status_ready.emit(f"Pico: Opening {self.port}...")
            with serial.Serial(self.port, self.baudrate, timeout=0.2) as serial_port:
                self.ser = serial_port
                self.status_ready.emit(f"Pico: Connected on {self.port}")

                if self.reboot_pico:
                    self.message_ready.emit("Pico: soft reboot...")
                    try:
                        serial_port.write(b"\x03")
                        time.sleep(0.3)
                        serial_port.write(b"\x04")
                        time.sleep(2.0)
                    except Exception as error:
                        self.message_ready.emit(f"Pico reboot warning: {error}")

                start_time = time.time()
                while time.time() - start_time < 2.0 and self.running:
                    line = serial_port.readline().decode(errors="ignore").strip()
                    if line:
                        self.message_ready.emit(f"Pico startup: {line}")

                self.status_ready.emit(
                    "Pico: Ready - use Start Phase Capture button"
                )
                while self.running:
                    self.process_pending_commands()
                    line = serial_port.readline().decode(errors="ignore").strip()
                    if not line:
                        continue
                    self.message_ready.emit(f"Pico says: {line}")
                    self.handle_serial_line(line)
        except Exception as error:
            self.status_ready.emit(f"Pico: Error - {error}")
            self.message_ready.emit(f"Pico listener error: {error}")
        finally:
            self.ser = None
            self.status_ready.emit("Pico: Disconnected")

    def stop(self):
        self.running = False
        try:
            if self.ser is not None:
                self.ser.close()
        except Exception:
            pass
