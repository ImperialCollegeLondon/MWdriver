"""Widget construction for selecting and viewing phase-analysis results."""

import numpy as np
from qtpy.QtWidgets import QComboBox, QLabel, QPushButton, QVBoxLayout, QWidget


def build_analysis_layer_presentation(result_key, result_data, source_name, result_name):
    result_data = np.asarray(result_data)
    is_rgb = result_data.ndim == 3 and result_data.shape[-1] in (3, 4)
    colormap = "gray" if result_key.startswith("phase_") else "viridis"
    if result_key == "wrapped":
        colormap = "twilight"

    contrast_limits = None
    if not is_rgb:
        finite_values = result_data[np.isfinite(result_data)]
        if finite_values.size:
            limits = (float(np.min(finite_values)), float(np.max(finite_values)))
            if limits[0] != limits[1]:
                contrast_limits = limits

    return {
        "name": f"Analysis [{source_name or 'unknown stack'}]: {result_name}",
        "is_rgb": is_rgb,
        "colormap": colormap,
        "contrast_limits": contrast_limits,
    }


def create_phase_analysis_controls(selection_changed, view_clicked):
    widget = QWidget()
    widget.setObjectName("embedded_phase_analysis_controls")

    layout = QVBoxLayout()
    layout.setContentsMargins(4, 4, 4, 4)
    layout.setSpacing(5)

    title_label = QLabel()
    title_label.setStyleSheet("font-weight: bold;")
    selector = QComboBox()
    view_button = QPushButton("View Analysis Results")

    layout.addWidget(title_label)
    layout.addWidget(selector)
    layout.addWidget(view_button)
    widget.setLayout(layout)

    selector.currentIndexChanged.connect(selection_changed)
    view_button.clicked.connect(view_clicked)
    return widget, title_label, selector, view_button
