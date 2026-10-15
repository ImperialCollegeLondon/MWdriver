"""Camera menu, status and exposure/gain controls."""

from qtpy.QtCore import QTimer, Qt
from qtpy.QtGui import QAction
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSlider,
    QDoubleSpinBox,
    QComboBox,
)

from mwdriver.domain.phase_psi import ALGORITHMS as PSI_ALGORITHMS, get_algorithm as get_psi_algorithm


STANDARD_PHASE_EXPOSURE_MS = {60.0: 20.0, 90.0: 30.0}


class CameraControlsController:
    def update_stream_menu_actions(self):
        if self.start_button is not None:
            self.start_button.setEnabled(not self.streaming_allowed)
        if self.stop_button is not None:
            self.stop_button.setEnabled(self.streaming_allowed)

    def on_psi_algorithm_changed(self, index):
        algorithm_id = self.psi_algorithm_selector.itemData(index)
        if algorithm_id is None:
            return

        if self.recalibration_active:
            previous_index = self.psi_algorithm_selector.findData(
                self.last_selected_psi_algorithm_id
            )
            self.psi_algorithm_selector.blockSignals(True)
            self.psi_algorithm_selector.setCurrentIndex(previous_index)
            self.psi_algorithm_selector.blockSignals(False)
            return

        self.selected_psi_algorithm_id = algorithm_id
        self.last_selected_psi_algorithm_id = algorithm_id
        exposure_ms = STANDARD_PHASE_EXPOSURE_MS[
            get_psi_algorithm(algorithm_id)["phase_step_deg"]
        ]
        self.exposure_slider.blockSignals(True)
        self.exposure_slider.setValue(int(exposure_ms * 10))
        self.exposure_slider.blockSignals(False)
        self.exposure_spinbox.blockSignals(True)
        self.exposure_spinbox.setValue(exposure_ms)
        self.exposure_spinbox.blockSignals(False)
        self.set_camera_exposure(exposure_ms)
        self.active_trigger_positions_ms = None
        positions = self.load_active_trigger_positions_from_json()
        if positions is None:
            self.status_label.setText(
                "Status: No calibration saved for "
                f"{get_psi_algorithm(algorithm_id)['label']}"
            )
        else:
            self.status_label.setText(
                "Status: Loaded calibrated triggers for "
                f"{get_psi_algorithm(algorithm_id)['label']}"
            )

    def place_camera_adjust_controls_in_layer_controls(self):
        self.add_widget_to_mwdriver_dock(self.camera_adjust_widget)

    def place_camera_status_controls_in_layer_controls(self):
        self.add_widget_to_mwdriver_dock(self.camera_status_widget)

    def init_controls(self):
        """
        Camera buttons are in the top menu bar.
        Exposure/Gain sliders are moved into the left layer-controls panel.
        Status information is also moved into the left panel, so no full-width bottom dock is used.
        """

        # ============================================================
        # 1) Camera Tools menu on the top menu bar
        # ============================================================
        qt_window = self.viewer.window._qt_window
        menubar = qt_window.menuBar()

        self.camera_menu = menubar.addMenu("Camera Tools")

        self.start_button = QAction("Start Stream", qt_window)
        self.stop_button = QAction("Stop Stream", qt_window)
        self.start_phase_capture_button = QAction("Start Phase Capture", qt_window)
        self.recalibrate_button = QAction("Recalibrate", qt_window)
        self.analyze_latest_phase_button = QAction(
            "Analyze Latest Capture",
            qt_window,
        )
        self.analyze_selected_phase_button = QAction(
            "Analyze Selected Stack",
            qt_window,
        )
        self.camera_preview_button = QAction("Return to Camera Preview", qt_window)

        self.start_button.setStatusTip("Start live camera stream")
        self.stop_button.setStatusTip("Stop live camera stream")
        self.start_phase_capture_button.setStatusTip(
            "Start one Pico triangle cycle and hardware-trigger 6 phase images"
        )
        self.recalibrate_button.setStatusTip(
            "Run two-stage 6-point phase recalibration and verification"
        )
        self.analyze_latest_phase_button.setStatusTip(
            "Analyze the latest completed six-frame phase capture"
        )
        self.analyze_selected_phase_button.setStatusTip(
            "Analyze the currently selected captured six-frame stack"
        )

        self.camera_menu.addAction(self.start_button)
        self.camera_menu.addAction(self.stop_button)
        self.camera_menu.addSeparator()
        self.camera_menu.addAction(self.start_phase_capture_button)
        self.camera_menu.addAction(self.recalibrate_button)
        self.camera_menu.addAction(self.analyze_selected_phase_button)
        self.camera_menu.addAction(self.analyze_latest_phase_button)
        self.camera_menu.addAction(self.camera_preview_button)

        self.update_stream_menu_actions()

        self.start_button.triggered.connect(self.allow_streaming)
        self.stop_button.triggered.connect(self.disallow_streaming)
        self.start_phase_capture_button.triggered.connect(
            self.request_pico_phase_capture_from_button
        )
        self.recalibrate_button.triggered.connect(
            self.start_recalibration_from_button
        )
        self.analyze_latest_phase_button.triggered.connect(
            self.run_latest_phase_analysis
        )
        self.analyze_selected_phase_button.triggered.connect(
            self.run_selected_phase_analysis
        )
        self.camera_preview_button.triggered.connect(self.show_camera_preview)

        # ============================================================
        # 2) Compact status widget in the left layer-controls panel
        #    No full-width bottom dock is created, so the camera preview
        #    can use more vertical space.
        # ============================================================
        self.camera_status_widget = QWidget()
        self.camera_status_widget.setObjectName("embedded_camera_status_controls")

        status_layout = QVBoxLayout()
        status_layout.setContentsMargins(4, 4, 4, 4)
        status_layout.setSpacing(4)

        status_title_label = QLabel("Camera Status:")
        status_title_label.setStyleSheet("font-weight: bold;")

        self.psi_algorithm_label = QLabel("Phase-step method:")
        self.psi_algorithm_selector = QComboBox()
        for algorithm_id, algorithm in PSI_ALGORITHMS.items():
            self.psi_algorithm_selector.addItem(
                algorithm["label"],
                algorithm_id,
            )
        self.psi_algorithm_selector.setCurrentIndex(
            self.psi_algorithm_selector.findData(self.selected_psi_algorithm_id)
        )
        self.psi_algorithm_selector.currentIndexChanged.connect(
            self.on_psi_algorithm_changed
        )

        self.status_label = QLabel("Status: Stopped")
        self.fps_label = QLabel("FPS: --")
        self.roi_label = QLabel("ROI: Not selected")
        self.profile_label = QLabel("Profile: Off")
        self.pico_label = QLabel("Pico: Not connected")
        self.phase_label = QLabel("Phase Capture: Idle")

        self.status_label.setWordWrap(True)
        self.fps_label.setWordWrap(True)
        self.roi_label.setWordWrap(True)
        self.profile_label.setWordWrap(True)
        self.pico_label.setWordWrap(True)
        self.phase_label.setWordWrap(True)

        status_layout.addWidget(status_title_label)
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.fps_label)
        status_layout.addWidget(self.roi_label)
        status_layout.addWidget(self.profile_label)
        status_layout.addWidget(self.pico_label)
        status_layout.addWidget(self.phase_label)

        self.camera_status_widget.setLayout(status_layout)

        QTimer.singleShot(350, self.place_camera_status_controls_in_layer_controls)
        QTimer.singleShot(900, self.place_camera_status_controls_in_layer_controls)

        # ============================================================
        # 3) Compact Exposure/Gain widget in the left layer-controls panel
        # ============================================================
        self.camera_adjust_widget = QWidget()
        self.camera_adjust_widget.setObjectName("embedded_camera_adjust_controls")

        adjust_layout = QVBoxLayout()
        adjust_layout.setContentsMargins(4, 4, 4, 4)
        adjust_layout.setSpacing(6)

        title_label = QLabel("Camera Adjustment:")
        title_label.setStyleSheet("font-weight: bold;")

        # ---------- Exposure ----------
        exposure_layout = QHBoxLayout()
        exposure_layout.setContentsMargins(0, 0, 0, 0)
        exposure_layout.setSpacing(6)

        self.exposure_text_label = QLabel("Exp:")
        self.exposure_text_label.setFixedWidth(55)

        self.exposure_slider = QSlider(Qt.Horizontal)
        self.exposure_slider.setMinimum(1)
        self.exposure_slider.setMaximum(1000)
        self.exposure_slider.setValue(int(self.exposure_s * 1000 * 10))
        self.exposure_slider.setMinimumWidth(80)

        self.exposure_spinbox = QDoubleSpinBox()
        self.exposure_spinbox.setMinimum(0.1)
        self.exposure_spinbox.setMaximum(100.0)
        self.exposure_spinbox.setSingleStep(0.1)
        self.exposure_spinbox.setDecimals(1)
        self.exposure_spinbox.setValue(self.exposure_s * 1000)
        self.exposure_spinbox.setFixedWidth(70)

        exposure_layout.addWidget(self.exposure_text_label)
        exposure_layout.addWidget(self.exposure_slider)
        exposure_layout.addWidget(self.exposure_spinbox)

        # ---------- Gain ----------
        gain_layout = QHBoxLayout()
        gain_layout.setContentsMargins(0, 0, 0, 0)
        gain_layout.setSpacing(6)

        self.gain_text_label = QLabel("Gain:")
        self.gain_text_label.setFixedWidth(55)

        self.gain_slider = QSlider(Qt.Horizontal)
        self.gain_slider.setMinimum(0)
        self.gain_slider.setMaximum(240)
        self.gain_slider.setValue(int(self.gain_value * 10))
        self.gain_slider.setMinimumWidth(80)

        self.gain_spinbox = QDoubleSpinBox()
        self.gain_spinbox.setMinimum(0.0)
        self.gain_spinbox.setMaximum(24.0)
        self.gain_spinbox.setSingleStep(0.1)
        self.gain_spinbox.setDecimals(1)
        self.gain_spinbox.setValue(self.gain_value)
        self.gain_spinbox.setFixedWidth(70)

        gain_layout.addWidget(self.gain_text_label)
        gain_layout.addWidget(self.gain_slider)
        gain_layout.addWidget(self.gain_spinbox)

        adjust_layout.addWidget(title_label)
        adjust_layout.addWidget(self.psi_algorithm_label)
        adjust_layout.addWidget(self.psi_algorithm_selector)
        adjust_layout.addLayout(exposure_layout)
        adjust_layout.addLayout(gain_layout)

        self.camera_adjust_widget.setLayout(adjust_layout)

        self.exposure_slider.valueChanged.connect(self.on_exposure_slider_changed)
        self.exposure_spinbox.valueChanged.connect(self.on_exposure_spinbox_changed)

        self.gain_slider.valueChanged.connect(self.on_gain_slider_changed)
        self.gain_spinbox.valueChanged.connect(self.on_gain_spinbox_changed)

        QTimer.singleShot(300, self.place_camera_adjust_controls_in_layer_controls)
        QTimer.singleShot(350, self.place_camera_status_controls_in_layer_controls)
        QTimer.singleShot(800, self.place_camera_adjust_controls_in_layer_controls)
        QTimer.singleShot(900, self.place_camera_status_controls_in_layer_controls)

        print("Camera menu, status panel, and compact exposure/gain controls ready.")

    def on_exposure_slider_changed(self, value):
        exposure_ms = value / 10.0

        self.exposure_spinbox.blockSignals(True)
        self.exposure_spinbox.setValue(exposure_ms)
        self.exposure_spinbox.blockSignals(False)

        self.set_camera_exposure(exposure_ms)

    def on_exposure_spinbox_changed(self, value):
        slider_value = int(value * 10)

        self.exposure_slider.blockSignals(True)
        self.exposure_slider.setValue(slider_value)
        self.exposure_slider.blockSignals(False)

        self.set_camera_exposure(value)

    def set_camera_exposure(self, exposure_ms):
        self.exposure_s = exposure_ms / 1000.0

        try:
            if self.cam is not None:
                self.cam.set_exposure(self.exposure_s)

            self.status_label.setText(
                f"Status: Exposure set to {exposure_ms:.1f} ms"
            )

        except Exception as e:
            self.status_label.setText(f"Status: Exposure set error: {e}")
            print("Exposure set error:", e)

    def on_gain_slider_changed(self, value):
        gain = value / 10.0

        self.gain_spinbox.blockSignals(True)
        self.gain_spinbox.setValue(gain)
        self.gain_spinbox.blockSignals(False)

        self.set_camera_gain(gain)

    def on_gain_spinbox_changed(self, value):
        slider_value = int(value * 10)

        self.gain_slider.blockSignals(True)
        self.gain_slider.setValue(slider_value)
        self.gain_slider.blockSignals(False)

        self.set_camera_gain(value)

    def set_camera_gain(self, gain):
        self.gain_value = gain

        try:
            if self.cam is not None:
                self.cam.set_gain(self.gain_value)

            self.status_label.setText(
                f"Status: Gain set to {self.gain_value:.1f}"
            )

        except Exception as e:
            self.status_label.setText(f"Status: Gain set error: {e}")
            print("Gain set error:", e)
