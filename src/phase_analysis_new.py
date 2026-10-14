"""
Analyse a camera phase capture using the six-frame PSI algorithm recorded in
its session metadata. Older captures default to the legacy 90-degree method.
"""

from pathlib import Path
from dataclasses import dataclass
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from phase_psi import (
    ALGORITHMS as PSI_ALGORITHMS,
    DEFAULT_ALGORITHM as DEFAULT_PSI_ALGORITHM,
    get_algorithm_id_from_metadata,
    phase_components,
    reconstruct_phase,
)
from phase_unwrap import unwrap_phase_from_center
from phase_fourier import detect_fourier_carrier
from phase_aberration import ZERNIKE_MODES, analyze_aberration


BASE_DIR = Path(__file__).resolve().parent
CAPTURE_DIR = BASE_DIR / "captured_frames"
WAVELENGTH_NM = 632.8

@dataclass(frozen=True)
class PhaseAnalysisResult:
    session_dir: Path
    raw_stack_path: Path
    algorithm_id: str
    wrapped_phase: np.ndarray
    unwrapped_phase: np.ndarray
    tilt_removed_phase: np.ndarray
    wavefront_aberration_nm: np.ndarray
    residual_surface_nm: np.ndarray
    zernike_coefficients_before_waves: np.ndarray
    zernike_coefficients_after_waves: np.ndarray
    aberration_rms_nm: float
    aberration_pv_nm: float
    export_path: Path | None


def figure_to_rgb(figure):
    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba())
    rgb = np.asarray(rgba[..., :3], dtype=np.uint8).copy()
    plt.close(figure)
    return rgb


def save_intermediate_png(path, image, title, cmap, vmin=None, vmax=None):
    figure, axis = plt.subplots(figsize=(10, 7))
    plotted_image = axis.imshow(
        image,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    axis.set_title(title)
    axis.set_xlabel("x pixel")
    axis.set_ylabel("y pixel")
    figure.colorbar(plotted_image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_result_images(result_dir, result_images):
    result_dir.mkdir(parents=True, exist_ok=True)

    for stale_png in result_dir.glob("*.png"):
        stale_png.unlink()

    for name, image in result_images.items():
        if image is None:
            continue

        array = np.asarray(image)
        if array.ndim == 2:
            path = result_dir / f"{name}.png"
            masked = np.ma.masked_invalid(array)
            if masked.count() == 0:
                continue

            vmin = float(np.nanmin(masked))
            vmax = float(np.nanmax(masked))
            if not np.isfinite(vmin) or not np.isfinite(vmax):
                continue
            if np.isclose(vmin, vmax):
                vmin -= 1.0
                vmax += 1.0

            cmap_name = "viridis"
            if name in {"wrapped", "phase_0", "phase_1", "phase_2", "phase_3", "phase_4", "phase_5"}:
                cmap_name = "twilight"
            elif name in {"wavefront_aberration_nm", "residual_nm", "tilt_removed", "unwrapped"}:
                cmap_name = "viridis"

            plt.imsave(path, masked, cmap=cmap_name, vmin=vmin, vmax=vmax)
        elif array.ndim == 3 and array.shape[-1] in (3, 4):
            path = result_dir / f"{name}.png"
            plt.imsave(path, array)


def write_analysis_export(export_path, output_arrays):
    if export_path is None:
        return None
    export_path = Path(export_path).resolve()
    export_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(export_path, **output_arrays)
    return export_path


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "session_dir",
        nargs="?",
        type=Path,
        help="capture session directory; defaults to the latest phase_capture session",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="save results without opening the interactive viewer",
    )
    parser.add_argument(
        "--export",
        type=Path,
        metavar="OUTPUT_NPZ",
        help="write analysis arrays to an NPZ file",
    )
    return parser


def get_export_path(export_path):
    return None if export_path is None else Path(export_path).resolve()


# ---------------------------------------------------------------------
# Find and load a camera phase-capture session
# ---------------------------------------------------------------------
def find_capture_input(session_dir=None):
    if session_dir is not None:
        session_dir = Path(session_dir)
        if not session_dir.is_absolute():
            session_dir = (BASE_DIR / session_dir).resolve()
        if not session_dir.is_dir():
            raise FileNotFoundError(
                f"Specified capture folder does not exist:\n{session_dir}"
            )

        raw_stack_files = sorted(session_dir.glob("*_raw_stack.npy"))
        if not raw_stack_files:
            raise FileNotFoundError(
                f"No *_raw_stack.npy file found in:\n{session_dir}"
            )
        return session_dir, raw_stack_files[0]

    raw_stack_files = list(
        CAPTURE_DIR.glob("phase_capture_*/*_raw_stack.npy")
    )
    if not raw_stack_files:
        raise FileNotFoundError(
            f"No phase_capture_* raw stack found in:\n{CAPTURE_DIR}"
        )

    raw_stack_path = max(
        raw_stack_files,
        key=lambda path: path.stat().st_mtime,
    )
    return raw_stack_path.parent, raw_stack_path



def analyze_capture(
    session_dir,
    raw_stack_path,
    *,
    export_path=None,
    show_viewer=False,
):
    session_dir = Path(session_dir).resolve()
    raw_stack_path = Path(raw_stack_path).resolve()
    phase_result_dir = session_dir / "phase_analysis_result_images"
    phase_result_dir.mkdir(parents=True, exist_ok=True)

    metadata_files = sorted(session_dir.glob("*_metadata.json"))
    algorithm_id = DEFAULT_PSI_ALGORITHM
    if metadata_files:
        with metadata_files[-1].open("r", encoding="utf-8") as metadata_file:
            capture_metadata = json.load(metadata_file)
        algorithm_id = get_algorithm_id_from_metadata(capture_metadata)
    psi_config = PSI_ALGORITHMS[algorithm_id]
    expected_phase_deg = psi_config["expected_phase_deg"]

    data = np.asarray(np.load(raw_stack_path), dtype=np.float64)

    if data.ndim != 3 or data.shape[0] != 6:
        raise ValueError(
            f"Expected camera raw stack shape (6, H, W), got {data.shape}."
        )

    _, sizey, sizex = data.shape

    roi_mask_path = session_dir / "roi_mask.npy"
    if roi_mask_path.exists():
        valid_mask = np.asarray(np.load(roi_mask_path), dtype=bool)
        if valid_mask.shape != (sizey, sizex):
            raise ValueError(
                f"ROI mask shape {valid_mask.shape} does not match "
                f"frame shape {(sizey, sizex)}."
            )
        mask = ~valid_mask
    else:
        valid_mask = np.ones((sizey, sizex), dtype=bool)
        mask = ~valid_mask

    if np.all(mask):
        raise ValueError("The capture ROI mask contains no valid pixels.")

    roi_valid_y, roi_valid_x = np.nonzero(~mask)
    roi_centroid_y = float(np.mean(roi_valid_y))
    roi_centroid_x = float(np.mean(roi_valid_x))
    nearest_roi_index = np.argmin(
        (roi_valid_y - roi_centroid_y) ** 2
        + (roi_valid_x - roi_centroid_x) ** 2
    )
    unwrap_center_y = int(roi_valid_y[nearest_roi_index])
    unwrap_center_x = int(roi_valid_x[nearest_roi_index])

    print("Phase capture folder:")
    print(session_dir)
    print("Raw stack:")
    print(raw_stack_path)
    print("Data shape:", data.shape)
    print("ROI mask:", roi_mask_path if roi_mask_path.exists() else "not present")
    print(
        "ROI unwrap centre: "
        f"x={unwrap_center_x}, y={unwrap_center_y} "
        f"(centroid x={roi_centroid_x:.2f}, y={roi_centroid_y:.2f})"
    )


    # ---------------------------------------------------------------------
    # Detect the first-order Fourier carrier peak in frame 0
    # ---------------------------------------------------------------------
    fourier_carrier = detect_fourier_carrier(data[0])
    fourier_spectrum = fourier_carrier.spectrum
    fourier_cy = fourier_carrier.center_y
    fourier_cx = fourier_carrier.center_x
    fourier_peak_y = fourier_carrier.peak_y
    fourier_peak_x = fourier_carrier.peak_x
    fourier_peak_distance = fourier_carrier.peak_distance
    fourier_mask_radius = fourier_carrier.mask_radius

    print(
        "Detected first-order Fourier peak: "
        f"x={fourier_peak_x}, y={fourier_peak_y}, "
        f"distance={fourier_peak_distance:.2f} px, "
        f"mask radius={fourier_mask_radius} px"
    )


    # ---------------------------------------------------------------------
    # Six-frame wrapped-phase recovery selected by capture metadata.
    # ---------------------------------------------------------------------
    phase_numerator, phase_denominator = phase_components(data, algorithm_id)
    phase = reconstruct_phase(data, algorithm_id)
    phase = np.ma.masked_array(phase, mask=mask)

    for index, frame in enumerate(data):
        save_intermediate_png(
            session_dir / f"phase_intermediate_01_input_{index}.png",
            np.ma.masked_array(frame, mask=mask),
            f"Input frame {index}: {expected_phase_deg[index]:g} degrees",
            "gray",
        )

    save_intermediate_png(
        session_dir / "phase_intermediate_02_numerator.png",
        np.ma.masked_array(phase_numerator, mask=mask),
        "N-bucket phase numerator",
        "RdBu_r",
    )
    save_intermediate_png(
        session_dir / "phase_intermediate_03_denominator.png",
        np.ma.masked_array(phase_denominator, mask=mask),
        "N-bucket phase denominator",
        "RdBu_r",
    )

    save_intermediate_png(
        session_dir / "phase_intermediate_total.png",
        np.ma.masked_array(np.sqrt(phase_denominator ** 2 + phase_numerator ** 2), mask=mask),
        "phase magnitude",
        "grey",
    )

    save_intermediate_png(
        session_dir / "phase_intermediate_04_wrapped.png",
        phase,
        "Wrapped phase",
        "twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )


    vertical_unwrapped, phaseUW = unwrap_phase_from_center(
        phase,
        mask,
        unwrap_center_y,
        unwrap_center_x,
    )

    save_intermediate_png(
        session_dir / "phase_intermediate_05_vertical_unwrapped.png",
        vertical_unwrapped,
        "Phase after vertical unwrapping",
        "viridis",
    )
    save_intermediate_png(
        session_dir / "phase_intermediate_06_unwrapped.png",
        phaseUW,
        "Unwrapped phase",
        "viridis",
    )
    print("Saved phase-unwrapping intermediate PNG files:")
    print(session_dir)


    aberration_result = analyze_aberration(
        phaseUW,
        valid_mask,
        WAVELENGTH_NM,
    )
    zernike_fit_mask = aberration_result.fit_mask
    zernike_coefficients_before_correction = (
        aberration_result.zernike_coefficients_before_waves
    )
    zernike_coefficients_after_tilt = (
        aberration_result.zernike_coefficients_after_waves
    )
    aberration_component_waves = aberration_result.aberration_component_waves
    phaseT = aberration_result.tilt_removed_phase
    residual_surface_nm = aberration_result.residual_surface_nm
    wavefront_aberration_nm = aberration_result.wavefront_aberration_nm
    aberration_rms_nm = aberration_result.aberration_rms_nm
    aberration_pv_nm = aberration_result.aberration_pv_nm
    zernike_center_x = aberration_result.zernike_center_x
    zernike_center_y = aberration_result.zernike_center_y
    zernike_radius = aberration_result.zernike_radius
    zernike_after_bar_labels = [
        f"J{mode_index:02d}\n{mode_name}"
        for mode_index, _, _, mode_name in ZERNIKE_MODES
    ]
    print(
        "Zernike coefficients before piston/tilt removal (waves):"
    )
    for mode, coefficient in zip(
        ZERNIKE_MODES,
        zernike_coefficients_before_correction,
    ):
        mode_index, n, m, mode_name = mode
        print(
            f"J{mode_index:02d} n={n}, m={m:2d} "
            f"{mode_name:<24} {coefficient:+.6f}"
        )

    print("Zernike coefficients before piston/tilt removal (waves):")
    for mode, coefficient in zip(
        ZERNIKE_MODES,
        zernike_coefficients_before_correction,
    ):
        mode_index, n, m, mode_name = mode
        print(
            f"J{mode_index:02d} n={n}, m={m:2d} "
            f"{mode_name:<24} {coefficient:+.6f}"
        )

    aberration_component_labels = [
        "Astigmatism\nJ5/J6",
        "Coma\nJ7/J8",
        "Trefoil\nJ9/J10",
        "Spherical\nJ11",
        "Secondary astig.\nJ12/J13",
        "Quadrafoil\nJ14/J15",
    ]
    fourier_figure, fourier_axis = plt.subplots(figsize=(12, 7))
    fourier_axis.imshow(np.log1p(fourier_spectrum), cmap="gray")
    fourier_axis.add_patch(plt.Circle(
        (fourier_peak_x, fourier_peak_y),
        fourier_mask_radius,
        fill=False,
        linewidth=2,
    ))
    fourier_axis.scatter(
        fourier_peak_x,
        fourier_peak_y,
        s=45,
        label="First-order peak",
    )
    fourier_axis.scatter(fourier_cx, fourier_cy, s=45, label="DC")
    fourier_axis.set_title("Detected First-order Fourier Peak")
    fourier_axis.set_xlabel("Frequency x")
    fourier_axis.set_ylabel("Frequency y")
    fourier_axis.legend()
    fourier_figure.tight_layout()
    fourier_peak_plot_rgb = figure_to_rgb(fourier_figure)

    zernike_bar_labels = [
        f"J{mode_index:02d}\n{mode_name}"
        for mode_index, _, _, mode_name in ZERNIKE_MODES
    ]
    zernike_figure, zernike_axis = plt.subplots(figsize=(12, 7))
    bar_positions = zernike_axis.bar(
        zernike_bar_labels,
        zernike_coefficients_before_correction,
    )
    for bar, coefficient in zip(bar_positions, zernike_coefficients_before_correction):
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
    zernike_before_plot_rgb = figure_to_rgb(zernike_figure)

    composition_figure, composition_axis = plt.subplots(figsize=(12, 7))
    composition_bars = composition_axis.bar(
        aberration_component_labels,
        aberration_component_waves,
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
        max(float(np.max(aberration_component_waves)) * 1.18, 0.01),
    )
    composition_figure.tight_layout()
    aberration_composition_plot_rgb = figure_to_rgb(composition_figure)


    print(
        f"centre x = {aberration_result.pupil_center_x}, "
        f"centre y = {aberration_result.pupil_center_y}"
    )
    print(
        f"z11 = {aberration_result.tilt_coefficient_x}, "
        f"z1m1 = {aberration_result.tilt_coefficient_y}"
    )

    zernike_after_figure, zernike_after_axis = plt.subplots(figsize=(12, 7))
    zernike_after_bar_positions = zernike_after_axis.bar(
        zernike_after_bar_labels,
        zernike_coefficients_after_tilt,
    )
    for bar, coefficient in zip(
        zernike_after_bar_positions,
        zernike_coefficients_after_tilt,
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
    zernike_after_plot_rgb = figure_to_rgb(zernike_after_figure)


    print(
        f"z20 = {aberration_result.defocus_coefficient:.4f} rad, "
        f"z22 = {aberration_result.astigmatism_coefficient_x:.4f} rad, "
        f"z2m2 = {aberration_result.astigmatism_coefficient_y:.4f} rad, "
        f"z40 = {aberration_result.spherical_coefficient:.4f} rad"
    )

    print(
        f"Lens wavefront aberration in ROI: RMS = {aberration_rms_nm:.2f} nm, "
        f"PV = {aberration_pv_nm:.2f} nm"
    )

    valid_output_mask = ~mask
    if export_path is not None:
        output_arrays = {
            f"phase_{index}": np.where(valid_output_mask, frame, np.nan).astype(
                np.float32
            )
            for index, frame in enumerate(data)
        }
        output_arrays.update({
            "wrapped": phase.filled(np.nan).astype(np.float32),
            "algorithm_id": np.asarray(algorithm_id),
            "expected_phase_deg": np.asarray(expected_phase_deg, dtype=np.float32),
            "unwrapped": phaseUW.filled(np.nan).astype(np.float32),
            "tilt_removed": phaseT.filled(np.nan).astype(np.float32),
            "wavefront_aberration_nm": wavefront_aberration_nm.filled(
                np.nan
            ).astype(np.float32),
            "wavefront_aberration_rms_nm": np.float32(aberration_rms_nm),
            "wavefront_aberration_pv_nm": np.float32(aberration_pv_nm),
            "fourier_spectrum": fourier_spectrum.astype(np.float32),
            "fourier_peak_xy": np.asarray(
                [fourier_peak_x, fourier_peak_y],
                dtype=np.int32,
            ),
            "fourier_dc_xy": np.asarray(
                [fourier_cx, fourier_cy],
                dtype=np.int32,
            ),
            "fourier_mask_radius": np.int32(fourier_mask_radius),
            "zernike_coefficients_before_correction_waves": (
                zernike_coefficients_before_correction.astype(np.float32)
            ),
            "aberration_component_waves": aberration_component_waves.astype(
                np.float32
            ),
            "zernike_before_plot_rgb": zernike_before_plot_rgb,
            "zernike_after_plot_rgb": zernike_after_plot_rgb,
            "aberration_composition_plot_rgb": aberration_composition_plot_rgb,
            "residual_nm": residual_surface_nm.filled(np.nan).astype(np.float32),
        })
        export_path = write_analysis_export(export_path, output_arrays)
        print("Saved phase analysis results:")
        print(export_path)

    saved_image_arrays = {
        "wrapped": phase.filled(np.nan),
        "unwrapped": phaseUW.filled(np.nan),
        "tilt_removed": phaseT.filled(np.nan),
        "wavefront_aberration_nm": wavefront_aberration_nm.filled(np.nan),
        "residual_nm": residual_surface_nm.filled(np.nan),
        "zernike_before": zernike_before_plot_rgb,
        "zernike_after": zernike_after_plot_rgb,
        "aberration_composition": aberration_composition_plot_rgb,
    }
    save_result_images(phase_result_dir, saved_image_arrays)
    print(f"Saved final phase-analysis PNGs to: {phase_result_dir}")


    # ---------------------------------------------------------------------
    # Show one selectable result at a time
    # ---------------------------------------------------------------------
    def show_result_viewer():
        results = []

        for index, frame in enumerate(data):
            results.append({
                "button": f"Phase {index}",
                "image": np.ma.masked_array(frame, mask=mask),
                "title": f"Phase {index}: {expected_phase_deg[index]:g} degrees",
                "cmap": "gray",
                "colorbar": "Intensity",
            })

        results.extend([
            {
                "button": "Wrapped",
                "image": phase,
                "title": f"Six-frame {psi_config['phase_step_deg']:g}° wrapped phase",
                "cmap": "twilight",
                "colorbar": "Wrapped phase (rad)",
                "vmin": -np.pi,
                "vmax": np.pi,
            },
            {
                "button": "Unwrapped",
                "image": phaseUW,
                "title": "Unwrapped phase",
                "cmap": "viridis",
                "colorbar": "Unwrapped phase (rad)",
            },
            {
                "button": "Tilt removed",
                "image": phaseT,
                "title": "Recovered phase after piston and tilt removal",
                "cmap": "viridis",
                "colorbar": "Phase (rad)",
            },
            {
                "button": "Defocus removed",
                "image": wavefront_aberration_nm,
                "title": (
                    "Defocus-removed wavefront aberration\n"
                    f"RMS = {aberration_rms_nm:.2f} nm, "
                    f"PV = {aberration_pv_nm:.2f} nm"
                ),
                "cmap": "RdBu_r",
                "colorbar": "Defocus-removed OPD (nm)",
            },
            {
                "button": "Residual",
                "image": residual_surface_nm,
                "title": "Residual surface shape",
                "cmap": "viridis",
                "colorbar": "Surface error (nm)",
            },
            {
                "button": "Zernike J1-J15 before",
                "plot_type": "bar",
                "labels": zernike_bar_labels,
                "values": zernike_coefficients_before_correction,
                "title": "Zernike Coefficients Before Piston and Tilt Removal",
                "ylabel": "Coefficient / waves",
            },
            {
                "button": "Zernike J1-J15 after",
                "plot_type": "bar",
                "labels": zernike_after_bar_labels,
                "values": zernike_coefficients_after_tilt,
                "title": "Zernike Coefficients After Piston and Tilt Removal",
                "ylabel": "Coefficient / waves",
            },
        ])

        figure = plt.figure(figsize=(12, 8))
        figure.canvas.manager.set_window_title("Phase Analysis Result Viewer")
        image_axis = figure.add_axes([0.08, 0.08, 0.76, 0.72])
        colorbar_axis = figure.add_axes([0.87, 0.12, 0.025, 0.64])
        buttons = []

        def display_result(result_index):
            result = results[result_index]
            image_axis.clear()
            colorbar_axis.clear()
            plot_type = result.get("plot_type", "image")
            if plot_type == "fourier":
                image_axis.imshow(np.log1p(result["spectrum"]), cmap="gray")
                peak_circle = plt.Circle(
                    (result["peak_x"], result["peak_y"]),
                    result["mask_radius"],
                    fill=False,
                    linewidth=2,
                )
                image_axis.add_patch(peak_circle)
                image_axis.scatter(
                    result["peak_x"],
                    result["peak_y"],
                    s=45,
                    label="First-order peak",
                )
                image_axis.scatter(
                    result["dc_x"],
                    result["dc_y"],
                    s=45,
                    label="DC",
                )
                image_axis.legend()
                image_axis.set_xlabel("Frequency x")
                image_axis.set_ylabel("Frequency y")
            elif plot_type == "bar":
                image_axis.bar(result["labels"], result["values"])
                image_axis.axhline(0, linewidth=1)
                image_axis.set_xlabel("Noll index")
                image_axis.set_ylabel(result["ylabel"])
            else:
                image = image_axis.imshow(
                    result["image"],
                    cmap=result["cmap"],
                    vmin=result.get("vmin"),
                    vmax=result.get("vmax"),
                )
                image_axis.set_xlabel("x pixel")
                image_axis.set_ylabel("y pixel")
                figure.colorbar(
                    image,
                    cax=colorbar_axis,
                    label=result["colorbar"],
                )
            image_axis.set_title(result["title"])
            figure.canvas.draw_idle()

        buttons_per_row = 6
        button_width = 0.14
        button_height = 0.045
        button_x_start = 0.035
        button_x_gap = 0.155
        button_y_positions = [0.94, 0.89, 0.84]

        for index, result in enumerate(results):
            row = index // buttons_per_row
            column = index % buttons_per_row
            button_axis = figure.add_axes([
                button_x_start + column * button_x_gap,
                button_y_positions[row],
                button_width,
                button_height,
            ])
            button = Button(button_axis, result["button"])
            button.on_clicked(
                lambda event, result_index=index: display_result(result_index)
            )
            buttons.append(button)

        display_result(0)
        plt.show()


    if show_viewer:
        show_result_viewer()

    return PhaseAnalysisResult(
        session_dir=session_dir,
        raw_stack_path=raw_stack_path,
        algorithm_id=algorithm_id,
        wrapped_phase=phase,
        unwrapped_phase=phaseUW,
        tilt_removed_phase=phaseT,
        wavefront_aberration_nm=wavefront_aberration_nm,
        residual_surface_nm=residual_surface_nm,
        zernike_coefficients_before_waves=zernike_coefficients_before_correction,
        zernike_coefficients_after_waves=zernike_coefficients_after_tilt,
        aberration_rms_nm=aberration_rms_nm,
        aberration_pv_nm=aberration_pv_nm,
        export_path=export_path,
    )


def main(argv=None):
    arguments = build_argument_parser().parse_args(argv)
    session_dir, raw_stack_path = find_capture_input(arguments.session_dir)
    return analyze_capture(
        session_dir,
        raw_stack_path,
        export_path=get_export_path(arguments.export),
        show_viewer=not arguments.no_viewer,
    )


if __name__ == "__main__":
    main()
