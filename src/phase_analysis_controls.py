"""Widget construction for selecting and viewing phase-analysis results."""

from qtpy.QtWidgets import QComboBox, QLabel, QPushButton, QVBoxLayout, QWidget


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
