"""Rendering helpers for offline phase-analysis diagnostics."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.text import Text
from matplotlib.tri import Triangulation
import numpy as np

from mwdriver.domain.wavefront_surface import wavefront_surface_limits


PLOT_DPI = 100
PLOT_FIGSIZE = (14.4, 10.8)


def style_plot_text(figure):
    for axis in figure.axes:
        axis.title.set_fontsize(16)
        axis.xaxis.label.set_fontsize(13)
        axis.yaxis.label.set_fontsize(13)
        axis.tick_params(axis="both", labelsize=11)
    for text in figure.findobj(match=Text):
        text.set_fontfamily("DejaVu Sans")
        text.set_antialiased(True)


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


def render_wavefront_surfaces(tilt_removed_nm, defocus_removed_nm, residual_nm):
    surfaces = [
        np.ma.masked_invalid(np.ma.asarray(surface, dtype=float))
        for surface in (tilt_removed_nm, defocus_removed_nm, residual_nm)
    ]
    if any(surface.ndim != 2 or surface.shape != surfaces[0].shape for surface in surfaces):
        raise ValueError("Wavefront surfaces must be matching 2D arrays.")
    limits = wavefront_surface_limits(surfaces[0])
    normalization = Normalize(*limits)
    height, width = surfaces[0].shape
    if min(height, width) < 2:
        raise ValueError("Wavefront surfaces must have at least two rows and columns.")
    valid_pixels = np.logical_or.reduce([
        ~np.ma.getmaskarray(surface) for surface in surfaces
    ])
    valid_y, valid_x = np.nonzero(valid_pixels)
    first_row, last_row = int(valid_y.min()), int(valid_y.max())
    first_column, last_column = int(valid_x.min()), int(valid_x.max())
    valid_height = last_row - first_row + 1
    valid_width = last_column - first_column + 1
    if min(valid_height, valid_width) < 2:
        raise ValueError("Valid wavefront data must span at least two rows and columns.")
    rows = np.unique(np.linspace(first_row, last_row, min(valid_height, 100), dtype=int))
    columns = np.unique(np.linspace(first_column, last_column, min(valid_width, 100), dtype=int))
    grid_x, grid_y = np.meshgrid(columns, rows)
    vertices = np.arange(grid_x.size).reshape(grid_x.shape)
    top_left = vertices[:-1, :-1].ravel()
    top_right = vertices[:-1, 1:].ravel()
    bottom_left = vertices[1:, :-1].ravel()
    bottom_right = vertices[1:, 1:].ravel()
    triangles = np.concatenate((
        np.column_stack((top_left, top_right, bottom_right)),
        np.column_stack((top_left, bottom_right, bottom_left)),
    ))
    figure = plt.figure(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    axes = []
    titles = ("Tilt removed", "Tilt and defocus removed", "Residual wavefront")
    for panel, (surface, title) in enumerate(zip(surfaces, titles), start=1):
        axis = figure.add_subplot(1, 3, panel, projection="3d")
        axes.append(axis)
        invalid = np.ma.getmaskarray(surface)
        summed_invalid = np.pad(invalid.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        invalid_per_cell = (
            summed_invalid[np.ix_(rows[1:] + 1, columns[1:] + 1)]
            - summed_invalid[np.ix_(rows[:-1], columns[1:] + 1)]
            - summed_invalid[np.ix_(rows[1:] + 1, columns[:-1])]
            + summed_invalid[np.ix_(rows[:-1], columns[:-1])]
        )
        triangle_mask = np.tile(invalid_per_cell.ravel() > 0, 2)
        if not np.all(triangle_mask):
            mesh = Triangulation(grid_x.ravel(), grid_y.ravel(), triangles)
            mesh.set_mask(triangle_mask)
            sampled = surface.filled(0.0)[np.ix_(rows, columns)]
            axis.plot_trisurf(
                mesh, sampled.ravel(), cmap="viridis", norm=normalization,
                linewidth=0, edgecolor="none", antialiased=False,
                shade=False, alpha=1.0,
            )
        axis.set_title(title, pad=18)
        axis.set_xlabel("x pixel", labelpad=8)
        axis.set_ylabel("y pixel", labelpad=8)
        axis.set_zlabel("Wavefront OPD (nm)", labelpad=10)
        axis.set_xlim(first_column, last_column)
        axis.set_ylim(last_row, first_row)
        axis.set_zlim(*limits)
        axis.set_box_aspect((
            valid_width / max(valid_height, valid_width),
            valid_height / max(valid_height, valid_width),
            0.75,
        ))
        axis.view_init(elev=30, azim=-60)
        axis.zaxis.set_major_locator(plt.MaxNLocator(5))
    colour_axis = figure.add_axes([0.25, 0.16, 0.5, 0.025])
    figure.colorbar(
        plt.cm.ScalarMappable(norm=normalization, cmap="viridis"),
        cax=colour_axis, orientation="horizontal", label="Wavefront OPD (nm)",
    )
    figure.subplots_adjust(left=0.02, right=0.94, bottom=0.26, top=0.92, wspace=0.18)
    style_plot_text(figure)
    for axis in axes:
        axis.title.set_fontsize(14)
        axis.zaxis.label.set_fontsize(11)
        axis.tick_params(axis="both", labelsize=9)
        axis.tick_params(axis="z", labelsize=9)
    return figure_to_rgb(figure)


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

    zernike_figure, zernike_axis = plt.subplots(
        figsize=PLOT_FIGSIZE, dpi=PLOT_DPI
    )
    bar_positions = zernike_axis.bar(
        zernike_bar_labels,
        coefficients_before,
    )
    zernike_axis.bar_label(
        bar_positions,
        fmt="%+.3f",
        padding=6,
        rotation=0,
        fontsize=14,
        fontweight="bold",
    )
    zernike_axis.margins(y=0.18)
    zernike_axis.axhline(0, linewidth=1)
    zernike_axis.set_title(
        "Zernike Coefficients Before Real-Space Piston/Tilt/Defocus Removal"
    )
    zernike_axis.set_xlabel("Noll index and aberration")
    zernike_axis.set_ylabel("Coefficient / waves")
    zernike_axis.tick_params(axis="x", labelrotation=45)
    style_plot_text(zernike_figure)
    zernike_figure.tight_layout()
    zernike_before_rgb = figure_to_rgb(zernike_figure)

    aberration_component_labels = [
        "Astigmatism\nJ5/J6",
        "Coma\nJ7/J8",
        "Trefoil\nJ9/J10",
        "Primary spherical\nJ11",
    ]
    composition_figure, composition_axis = plt.subplots(
        figsize=PLOT_FIGSIZE, dpi=PLOT_DPI
    )
    composition_bars = composition_axis.bar(
        aberration_component_labels,
        component_waves,
    )
    composition_axis.bar_label(
        composition_bars,
        fmt="%.4f",
        padding=6,
        rotation=0,
        fontsize=14,
        fontweight="bold",
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
    style_plot_text(composition_figure)
    composition_figure.tight_layout()
    composition_rgb = figure_to_rgb(composition_figure)

    zernike_after_figure, zernike_after_axis = plt.subplots(
        figsize=PLOT_FIGSIZE, dpi=PLOT_DPI
    )
    zernike_after_bar_positions = zernike_after_axis.bar(
        zernike_bar_labels,
        coefficients_after,
    )
    zernike_after_axis.bar_label(
        zernike_after_bar_positions,
        fmt="%+.3f",
        padding=6,
        rotation=0,
        fontsize=14,
        fontweight="bold",
    )
    zernike_after_axis.margins(y=0.18)
    zernike_after_axis.axhline(0, linewidth=1)
    zernike_after_axis.set_title(
        "Zernike Coefficients of Residual After Piston, Tilt and Defocus Removal"
    )
    zernike_after_axis.set_xlabel("Noll index and aberration")
    zernike_after_axis.set_ylabel("Coefficient / waves")
    zernike_after_axis.tick_params(axis="x", labelrotation=45)
    style_plot_text(zernike_after_figure)
    zernike_after_figure.tight_layout()
    zernike_after_rgb = figure_to_rgb(zernike_after_figure)

    return AberrationPlotImages(
        zernike_bar_labels=zernike_bar_labels,
        zernike_before_rgb=zernike_before_rgb,
        zernike_after_rgb=zernike_after_rgb,
        composition_rgb=composition_rgb,
    )