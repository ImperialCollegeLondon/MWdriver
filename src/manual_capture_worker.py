"""Qt worker for saving manually triggered camera frame stacks."""

import os
import time

import imageio.v2 as imageio
import numpy as np
from qtpy.QtCore import QObject, Signal

from phase_capture_storage import raw_to_viewable_uint8


class FrameSaveWorker(QObject):
    status_ready = Signal(str)
    finished = Signal(str)

    def __init__(self, frames, save_dir="captured_frames", prefix="manual_capture"):
        super().__init__()
        self.frames = frames
        self.save_dir = save_dir
        self.prefix = prefix

    def run(self):
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            print("====================================")
            print("Frame save worker started")
            print("Current working directory:", os.getcwd())
            print("Save directory:", self.save_dir)
            print("Number of frames:", len(self.frames))
            print("====================================")

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            stack = np.asarray(self.frames)
            raw_stack_path = os.path.join(
                self.save_dir,
                f"{self.prefix}_{timestamp}_raw_stack.npy",
            )
            np.save(raw_stack_path, stack)

            print("Saved raw stack:", raw_stack_path)
            print("Raw stack shape:", stack.shape)
            print("Raw stack dtype:", stack.dtype)
            print("Raw stack min:", np.min(stack))
            print("Raw stack max:", np.max(stack))
            self.status_ready.emit(f"Saved raw stack: {raw_stack_path}")

            for index, frame in enumerate(self.frames):
                viewable_frame = raw_to_viewable_uint8(frame)
                filepath = os.path.join(
                    self.save_dir,
                    f"{self.prefix}_{timestamp}_{index + 1:02d}.png",
                )
                imageio.imwrite(filepath, viewable_frame)
                print("Saved viewable PNG:", filepath)
                print(
                    f"PNG {index + 1}: "
                    f"raw min={np.min(frame)}, raw max={np.max(frame)}, "
                    f"png min={np.min(viewable_frame)}, png max={np.max(viewable_frame)}"
                )
                self.status_ready.emit(f"Saved PNG: {filepath}")

            self.finished.emit(f"{self.prefix} saved: {len(self.frames)} frames")
        except Exception as error:
            error_text = f"{self.prefix} save error: {error}"
            self.finished.emit(error_text)
            print(error_text)
