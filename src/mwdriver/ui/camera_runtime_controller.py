"""Camera acquisition and Pico worker lifecycle for the desktop viewer."""

import time

import numpy as np
from qtpy.QtCore import QObject, QThread, Signal

from mwdriver.domain.camera_frames import prepare_display_frame
from mwdriver.hardware.pico_serial_worker import PicoSerialWorker
from mwdriver.domain.stream_policy import get_camera_stream_mode
from mwdriver.ui.worker_lifecycle import stop_worker_thread


class CameraWorker(QObject):
    frame_ready = Signal(object, object, object)
    fps_ready = Signal(float)
    status_ready = Signal(str)

    def __init__(self, cam, display_downsample=1, phase_mode_getter=None):
        super().__init__()

        self.cam = cam
        self.display_downsample = display_downsample
        self.phase_mode_getter = phase_mode_getter

        self.running = False

        self.n_frames = 0
        self.t_fps = time.monotonic()
        self.last_fps_update = time.monotonic()
        self.fps_update_interval = 1.0

    def run(self):
        self.running = True

        self.n_frames = 0
        self.t_fps = time.monotonic()
        self.last_fps_update = time.monotonic()

        self.status_ready.emit("Status: Running...")

        while self.running:
            try:
                image_range = self.cam.get_new_images_range()

                if image_range is None:
                    QThread.msleep(1)
                    continue

                in_phase_external_mode = False
                try:
                    if callable(self.phase_mode_getter):
                        in_phase_external_mode = bool(self.phase_mode_getter())
                except Exception:
                    in_phase_external_mode = False

                raw_frames = []
                raw_frame_infos = []

                if in_phase_external_mode:
                    try:
                        frames_result = self.cam.read_multiple_images(
                            image_range,
                            return_info=True,
                        )

                        if frames_result is None:
                            raw_frames = []
                            raw_frame_infos = []
                        elif isinstance(frames_result, tuple):
                            frames = frames_result[0]
                            infos = frames_result[1] if len(frames_result) > 1 else None
                        else:
                            frames = frames_result
                            infos = None

                        if isinstance(frames, np.ndarray):
                            if frames.ndim == 2:
                                raw_frames = [frames]
                            elif frames.ndim >= 3:
                                raw_frames = [frames[i] for i in range(frames.shape[0])]
                        elif isinstance(frames, (list, tuple)):
                            raw_frames = [np.asarray(frame) for frame in frames]

                        if isinstance(infos, (list, tuple)):
                            raw_frame_infos = list(infos)
                        elif infos is not None:
                            raw_frame_infos = [infos] * max(1, len(raw_frames))
                        else:
                            raw_frame_infos = [None] * len(raw_frames)

                    except Exception as e:
                        self.status_ready.emit(
                            f"Status: read_multiple_images warning: {e}; using newest frame fallback"
                        )
                        raw_frame = self.cam.read_newest_image()
                        if raw_frame is not None:
                            raw_frames = [raw_frame]
                            raw_frame_infos = [None]
                else:
                    try:
                        raw_frame = self.cam.read_newest_image(return_info=True)
                        if raw_frame is not None:
                            if isinstance(raw_frame, tuple) and len(raw_frame) == 2:
                                raw_frame, frame_info = raw_frame
                                raw_frames = [raw_frame]
                                raw_frame_infos = [frame_info]
                            else:
                                raw_frames = [raw_frame]
                                raw_frame_infos = [None]
                        else:
                            raw_frames = []
                            raw_frame_infos = []
                    except Exception as e:
                        self.status_ready.emit(
                            f"Status: read_newest_image warning: {e}; using fallback"
                        )
                        raw_frame = self.cam.read_newest_image()
                        if raw_frame is not None:
                            raw_frames = [raw_frame]
                            raw_frame_infos = [None]

                if len(raw_frames) == 0:
                    QThread.msleep(1)
                    continue

                for index, raw_frame in enumerate(raw_frames):
                    if raw_frame is None:
                        continue

                    frame_meta = raw_frame_infos[index] if index < len(raw_frame_infos) else None

                    display_frame = prepare_display_frame(
                        raw_frame,
                        self.display_downsample,
                    )

                    self.frame_ready.emit(raw_frame, display_frame, frame_meta)

                    self.n_frames += 1

                now = time.monotonic()

                if now - self.last_fps_update >= self.fps_update_interval:
                    fps = self.n_frames / (now - self.t_fps)
                    self.fps_ready.emit(fps)
                    self.last_fps_update = now

            except Exception as e:
                self.status_ready.emit(f"Status: Worker error: {e}")
                QThread.msleep(10)

        self.status_ready.emit("Status: Stopped")

    def stop(self):
        self.running = False


class CameraRuntimeController:
    def start_stream(self):
        if self.is_streaming:
            return

        stream_mode = get_camera_stream_mode(
            streaming_allowed=self.streaming_allowed,
            preview_layer_visible=(
                self.layer is not None and self.layer.visible
            ),
            camera_mode=self.current_camera_mode,
            phase_capture_active=self.phase_capture_active,
            recalibration_active=self.recalibration_active,
        )
        if stream_mode == "stopped":
            return

        self.is_streaming = True

        self.worker_thread = QThread()
        self.worker = CameraWorker(
            cam=self.cam,
            display_downsample=self.display_downsample,
            phase_mode_getter=lambda: (
                self.phase_capture_active
                and self.current_camera_mode == "external"
            )
        )

        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)

        self.worker.frame_ready.connect(self.update_frame)
        self.worker.fps_ready.connect(self.update_fps)
        self.worker.status_ready.connect(self.update_status)

        self.worker_thread.start()

        self.update_stream_menu_actions()

        self.status_label.setText("Status: Starting...")

    def stop_stream(self):
        if not self.is_streaming:
            return

        self.is_streaming = False

        stop_worker_thread(self.worker, self.worker_thread)

        self.worker = None
        self.worker_thread = None

        self.update_stream_menu_actions()

        self.status_label.setText("Status: Stopped")
        self.fps_label.setText("FPS: --")

    def update_frame(self, raw_frame, display_frame, frame_meta=None):
        self.latest_raw_frame = raw_frame
        self.latest_display_frame = display_frame

        if (
            self.phase_capture_active
            and self.current_camera_mode == "external"
        ):
            self.handle_triggered_phase_frame(raw_frame, frame_meta=frame_meta)

            # Do not let GUI redraw work delay closely spaced triggered frames.
            return

        self.layer.data = display_frame

    def update_fps(self, fps):
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def update_status(self, text):
        self.status_label.setText(text)

    def start_pico_listener(self):
        if self.pico_thread is not None:
            return

        self.pico_thread = QThread()
        self.pico_worker = PicoSerialWorker(
            port=self.pico_port,
            baudrate=self.pico_baudrate,
            reboot_pico=True
        )

        self.pico_worker.moveToThread(self.pico_thread)

        self.pico_thread.started.connect(self.pico_worker.run)

        self.pico_worker.phase_session_started.connect(self.start_phase_capture_session)
        self.pico_worker.phase_requested.connect(self.capture_phase_frame)
        self.pico_worker.phase_session_done.connect(self.on_phase_session_done)

        self.pico_worker.capture_requested.connect(self.request_manual_capture)

        self.pico_worker.status_ready.connect(self.update_pico_status)
        self.pico_worker.message_ready.connect(self.print_pico_message)

        self.pico_thread.start()

        print("Pico listener thread started.")

    def stop_pico_listener(self):
        try:
            if self.pico_worker is not None:
                self.pico_worker.stop()
        except Exception as e:
            print("Error while stopping Pico worker:", e)

        try:
            if self.pico_thread is not None:
                self.pico_thread.quit()
                self.pico_thread.wait(2000)
        except Exception as e:
            print("Error while stopping Pico thread:", e)

        self.pico_worker = None
        self.pico_thread = None

        print("Pico listener stopped.")

    def update_pico_status(self, text):
        print(text)

        try:
            self.pico_label.setText(text)
        except Exception:
            pass