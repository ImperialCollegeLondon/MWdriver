"""Manual frame capture and save-worker orchestration for the camera viewer."""

import os
from pathlib import Path

from qtpy.QtCore import QThread, QTimer

from mwdriver.capture.manual_capture_worker import FrameSaveWorker


MANUAL_CAPTURE_COUNT = 6
MANUAL_CAPTURE_INTERVAL_MS = 150
SCRIPT_DIR = str(Path(__file__).resolve().parents[2])
SAVE_DIR = os.path.join(SCRIPT_DIR, "captured_frames")


class CameraManualCaptureController:
    def initialize_manual_capture_state(self):
        self.manual_capture_in_progress = False
        self.manual_capture_frames = []
        self.manual_capture_index = 0

    def request_manual_capture(self):
        print("====================================")
        print("request_manual_capture() called")
        print("is_streaming:", self.is_streaming)
        print("latest_raw_frame is None:", self.latest_raw_frame is None)
        print("manual_capture_in_progress:", self.manual_capture_in_progress)
        print("====================================")

        if self.manual_capture_in_progress:
            print("Manual capture ignored: previous capture still running.")
            self.status_label.setText("Status: Manual capture already running")
            return

        if self.latest_raw_frame is None:
            print("No frame available for manual capture.")
            self.status_label.setText("Status: No frame available for manual capture")
            return

        self.manual_capture_in_progress = True
        self.manual_capture_frames = []
        self.manual_capture_index = 0

        self.status_label.setText("Status: Manual capture started")
        print("Manual capture started.")

        self.collect_manual_frame()

    def collect_manual_frame(self):
        if self.latest_raw_frame is None:
            print("No latest_raw_frame during manual capture.")
            self.finish_manual_capture()
            return

        frame = self.latest_raw_frame.copy()
        self.manual_capture_frames.append(frame)

        self.manual_capture_index += 1

        self.status_label.setText(
            f"Status: Manual captured {self.manual_capture_index}/{MANUAL_CAPTURE_COUNT}"
        )

        print(
            f"Manual captured frame "
            f"{self.manual_capture_index}/{MANUAL_CAPTURE_COUNT}"
        )

        if self.manual_capture_index < MANUAL_CAPTURE_COUNT:
            QTimer.singleShot(
                MANUAL_CAPTURE_INTERVAL_MS,
                self.collect_manual_frame
            )
        else:
            self.finish_manual_capture()

    def finish_manual_capture(self):
        frames = self.manual_capture_frames.copy()

        self.manual_capture_frames = []
        self.manual_capture_index = 0
        self.manual_capture_in_progress = False

        if len(frames) == 0:
            self.status_label.setText("Status: Manual capture failed - no frames")
            return

        self.status_label.setText("Status: Saving manual frames...")
        print(f"Manual capture finished. Saving {len(frames)} frames...")

        self.start_save_manual_frames_worker(frames)

    def start_save_manual_frames_worker(self, frames):
        if self.save_thread is not None:
            print("Save worker is already running.")
            self.status_label.setText("Status: Save worker already running")
            return

        self.save_thread = QThread()
        self.save_worker = FrameSaveWorker(
            frames=frames,
            save_dir=SAVE_DIR,
            prefix="manual_capture"
        )

        self.save_worker.moveToThread(self.save_thread)

        self.save_thread.started.connect(self.save_worker.run)
        self.save_worker.status_ready.connect(self.update_status)
        self.save_worker.finished.connect(self.on_manual_save_finished)
        self.save_worker.finished.connect(self.save_thread.quit)
        self.save_thread.finished.connect(self.cleanup_save_worker)

        self.save_thread.start()

    def on_manual_save_finished(self, text):
        print(text)
        self.status_label.setText(f"Status: {text}")
