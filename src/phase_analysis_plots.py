"""Rendering helpers for offline phase-analysis diagnostics."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class AberrationPlotImages:
    zernike_bar_labels: list[str]
    zernike_before_rgb: np.ndarray
    zernike_after_rgb: np.ndarray
    composition_rgb: np.ndarray


def figure_to_rgb(figure):
    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba())
    rgb = np.asarray(rgba[..., :3], dtype=np.uint8).copy()
    plt.close(figure)
    return rgb


def render_aberration_plots(
    zernike_modes,
    coefficients_before,
    coefficients_after,
    component_waves,
):
    zernike_bar_labels = [
        f"J{mode_index:02d}\n{mode_name}"
        for mode_index, _, _, mode_name in zernike_modes
    ]

    zernike_figure, zernike_axis = plt.subplots(figsize=(12, 7))
    bar_positions = zernike_axis.bar(
        zernike_bar_labels,
        coefficients_before,
    )
    for bar, coefficient in zip(bar_positions, coefficients_before):
        zernike_axis.text(
            bar.get_x() + bar.get_width() / 2,
            coefficient,
            f"{coefficient:+.3f}",
            ha="center",
            va="bottom" if coefficient >= 0 else "top",
            rotation=90,
            fontsize=7,
        )
    zernike_axis.axhline(0, linewidth=1)
    zernike_axis.set_title(
        "Zernike Coefficients Before Piston and Tilt Removal"
    )
    zernike_axis.set_xlabel("Noll index and aberration")
    zernike_axis.set_ylabel("Coefficient / waves")
    zernike_axis.tick_params(axis="x", labelrotation=45)
    zernike_figure.tight_layout()
    zernike_before_rgb = figure_to_rgb(zernike_figure)

    aberration_component_labels = [
        "Astigmatism\nJ5/J6",
        "Coma\nJ7/J8",
        "Trefoil\nJ9/J10",
        "Spherical\nJ11",
        "Secondary astig.\nJ12/J13",
        "Quadrafoil\nJ14/J15",
    ]
    composition_figure, composition_axis = plt.subplots(figsize=(12, 7))
    composition_bars = composition_axis.bar(
        aberration_component_labels,
        component_waves,
    )
    composition_axis.bar_label(
        composition_bars,
        fmt="%.4f",
        padding=3,
    )
    composition_axis.set_title(
        "ROI System Aberration Composition (Piston/Tilt/Defocus Excluded)"
    )
    composition_axis.set_xlabel("Zernike aberration group")
    composition_axis.set_ylabel("Combined coefficient / waves")
    composition_axis.set_ylim(
        0,
        max(float(np.max(component_waves)) * 1.18, 0.01),
    )
    composition_figure.tight_layout()
    composition_rgb = figure_to_rgb(composition_figure)

    zernike_after_figure, zernike_after_axis = plt.subplots(figsize=(12, 7))
    zernike_after_bar_positions = zernike_after_axis.bar(
        zernike_bar_labels,
        coefficients_after,
    )
    for bar, coefficient in zip(
        zernike_after_bar_positions,
        coefficients_after,
    ):
        zernike_after_axis.text(
            bar.get_x() + bar.get_width() / 2,
            coefficient,
            f"{coefficient:+.3f}",
            ha="center",
            va="bottom" if coefficient >= 0 else "top",
            rotation=90,
            fontsize=7,
        )
    zernike_after_axis.axhline(0, linewidth=1)
    zernike_after_axis.set_title(
        "Zernike Coefficients After Piston and Tilt Removal"
    )
    zernike_after_axis.set_xlabel("Noll index and aberration")
    zernike_after_axis.set_ylabel("Coefficient / waves")
    zernike_after_axis.tick_params(axis="x", labelrotation=45)
    zernike_after_figure.tight_layout()
    zernike_after_rgb = figure_to_rgb(zernike_after_figure)

    return AberrationPlotImages(
        zernike_bar_labels=zernike_bar_labels,
        zernike_before_rgb=zernike_before_rgb,
        zernike_after_rgb=zernike_after_rgb,
        composition_rgb=composition_rgb,
    )