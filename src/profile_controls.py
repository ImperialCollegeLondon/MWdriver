"""Widget construction for the camera intensity-profile controls."""

from qtpy.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget


def create_profile_controls():
    widget = QWidget()
    widget.setObjectName("embedded_profile_controls")

    profile_layout = QVBoxLayout()
    profile_layout.setContentsMargins(0, 0, 0, 0)
    profile_layout.setSpacing(6)
    title_label = QLabel("Intensity Profile Control:")
    title_label.setStyleSheet("font-weight: bold;")

    button_layout = QHBoxLayout()
    button_layout.setContentsMargins(0, 0, 0, 0)
    button_layout.setSpacing(6)

    profile_button = QPushButton("Intensity Profile")
    clear_profile_button = QPushButton("Clear Profile")
    profile_button.setMinimumHeight(28)
    clear_profile_button.setMinimumHeight(28)
    button_layout.addWidget(profile_button)
    button_layout.addWidget(clear_profile_button)

    info_label = QLabel(
        "Draw a yellow line, then drag the line or its endpoints to update the curve."
    )
    info_label.setWordWrap(True)

    profile_layout.addWidget(title_label)
    profile_layout.addLayout(button_layout)
    profile_layout.addWidget(info_label)
    widget.setLayout(profile_layout)
    return widget, profile_button, clear_profile_button
