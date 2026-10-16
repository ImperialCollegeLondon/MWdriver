"""Widget construction for selecting and viewing phase-analysis results."""

import numpy as np
from qtpy.QtWidgets import QComboBox, QLabel, QPushButton, QVBoxLayout, QWidget
from mwdriver.domain.display_range import percentile_display_limits


def choose_analysis_layer_operation(
    create_new_layer,
    has_current_layer,
    has_current_record,
    current_layer_is_rgb,
    result_is_rgb,
):
    if create_new_layer or not has_current_layer:
        return "add"
    if not has_current_record or current_layer_is_rgb != result_is_rgb:
        return "replace"
    return "update"


def build_phase_analysis_result_options(phase_analysis_results):
    aberration_label = "Primary defocus removed; higher modes retained (nm OPD)"
    aberration_rms = phase_analysis_results.get("wavefront_aberration_rms_nm")
    aberration_pv = phase_analysis_results.get("wavefront_aberration_pv_nm")
    if aberration_rms is not None and aberration_pv is not None:
        aberration_label = (
            f"Primary defocus removed; higher modes retained: "
            f"RMS {float(aberration_rms):.2f} nm, "
            f"PV (0.1-99.9%) {float(aberration_pv):.2f} nm"
        )

    result_options = [
        ("Wrapped phase", "wrapped"),
        ("Unwrapped phase", "unwrapped"),
        ("Tilt removed", "tilt_removed"),
        (aberration_label, "wavefront_aberration_nm"),
        ("Residual surface (nm)", "residual_nm"),
        (
            "Zernike J1-J28 before piston/tilt removal",
            "zernike_before_plot_rgb",
        ),
        (
            "Zernike J1-J28 residual after piston/tilt/defocus removal",
            "zernike_after_plot_rgb",
        ),
        (
            "ROI system aberration composition",
            "aberration_composition_plot_rgb",
        ),
    ]
    return [
        (label, key)
        for label, key in result_options
        if key in phase_analysis_results
    ]


def build_analysis_layer_presentation(result_key, result_data, source_name, result_name):
    result_data = np.asarray(result_data)
    is_rgb = result_data.ndim == 3 and result_data.shape[-1] in (3, 4)
    colormap = "gray" if result_key.startswith("phase_") else "viridis"
    if result_key == "wrapped":
        colormap = "twilight"

    contrast_limits = None
    if not is_rgb:
        limits = percentile_display_limits(result_data)
        if limits is not None and not np.isclose(limits[0], limits[1]):
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
