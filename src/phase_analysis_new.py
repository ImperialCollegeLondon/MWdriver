"""
Analyse the latest camera phase capture using the six-frame 90°
phase-shifting algorithm.

Expected frame phases:
0, 90, 180, 270, 360, 450 degrees.
"""

from pathlib import Path
from math import factorial
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


BASE_DIR = Path(__file__).resolve().parent
CAPTURE_DIR = BASE_DIR / "captured_frames"
WAVELENGTH_NM = 632.8
NO_VIEWER = "--no-viewer" in sys.argv

ZERNIKE_MODES = [
    (1, 0, 0, "Piston"),
    (2, 1, -1, "Tilt Y"),
    (3, 1, 1, "Tilt X"),
    (4, 2, 0, "Defocus"),
    (5, 2, -2, "Oblique Astigmatism"),
    (6, 2, 2, "Vertical Astigmatism"),
    (7, 3, -1, "Vertical Coma"),
    (8, 3, 1, "Horizontal Coma"),
    (9, 3, -3, "Vertical Trefoil"),
    (10, 3, 3, "Oblique Trefoil"),
    (11, 4, 0, "Primary Spherical"),
    (12, 4, -2, "Secondary Astigmatism"),
    (13, 4, 2, "Secondary Astigmatism"),
    (14, 4, -4, "Quadrafoil"),
    (15, 4, 4, "Quadrafoil"),
]


def zernike_radial(n, m, rho_values):
    m = abs(m)
    if (n - m) % 2 != 0:
        return np.zeros_like(rho_values, dtype=np.float64)

    radial = np.zeros_like(rho_values, dtype=np.float64)
    for index in range((n - m) // 2 + 1):
        numerator = (-1) ** index * factorial(n - index)
        denominator = (
            factorial(index)
            * factorial((n + m) // 2 - index)
            * factorial((n - m) // 2 - index)
        )
        radial += numerator / denominator * rho_values ** (n - 2 * index)
    return radial


def zernike_polynomial(n, m, rho_values, theta_values):
    radial = zernike_radial(n, m, rho_values)
    if m == 0:
        return np.sqrt(n + 1) * radial

    normalization = np.sqrt(2 * (n + 1))
    if m > 0:
        return normalization * radial * np.cos(m * theta_values)
    return normalization * radial * np.sin(abs(m) * theta_values)


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


def get_export_path():
    if "--export" not in sys.argv:
        return None

    option_index = sys.argv.index("--export")
    if option_index + 1 >= len(sys.argv):
        raise ValueError("--export requires an output .npz path.")

    return Path(sys.argv[option_index + 1]).resolve()


# ---------------------------------------------------------------------
# Find and load a camera phase-capture session
# ---------------------------------------------------------------------
def find_capture_input():
    if len(sys.argv) >= 2:
        session_dir = Path(sys.argv[1])
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


session_dir, raw_stack_path = find_capture_input()
phase_result_dir = session_dir / "phase_analysis_result_images"
phase_result_dir.mkdir(parents=True, exist_ok=True)

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
    mask = np.zeros((sizey, sizex), dtype=bool)

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
fourier_spectrum = np.abs(np.fft.fftshift(np.fft.fft2(data[0])))
fourier_cy = sizey // 2
fourier_cx = sizex // 2
fourier_yy, fourier_xx = np.ogrid[:sizey, :sizex]

dc_radius = 35
lower_half_margin = 10
refine_radius = 45
dc_region = (
    (fourier_yy - fourier_cy) ** 2
    + (fourier_xx - fourier_cx) ** 2
) <= dc_radius ** 2

search_spectrum = fourier_spectrum.copy()
search_spectrum[dc_region] = 0
search_spectrum[:fourier_cy + lower_half_margin, :] = 0
coarse_peak_y, coarse_peak_x = np.unravel_index(
    np.argmax(search_spectrum),
    search_spectrum.shape,
)

refine_window = (
    (fourier_yy - coarse_peak_y) ** 2
    + (fourier_xx - coarse_peak_x) ** 2
) <= refine_radius ** 2
refine_spectrum = fourier_spectrum.copy()
refine_spectrum[~refine_window] = 0
refine_spectrum[dc_region] = 0
fourier_peak_y, fourier_peak_x = np.unravel_index(
    np.argmax(refine_spectrum),
    refine_spectrum.shape,
)

fourier_peak_distance = float(np.hypot(
    fourier_peak_y - fourier_cy,
    fourier_peak_x - fourier_cx,
))
fourier_mask_radius = int(np.clip(0.35 * fourier_peak_distance, 6, 30))

print(
    "Detected first-order Fourier peak: "
    f"x={fourier_peak_x}, y={fourier_peak_y}, "
    f"distance={fourier_peak_distance:.2f} px, "
    f"mask radius={fourier_mask_radius} px"
)


# ---------------------------------------------------------------------
# Six-frame 90° wrapped-phase recovery (reference: Class A, N = 6)
# tan(phi) = (4*I4 - 3*I2 - I6) / (I1 - 4*I3 + 3*I5)
# ---------------------------------------------------------------------
I1 = data[0]  #     0 degrees
I2 = data[1]  #    90 degrees
I3 = data[2]  #   180 degrees
I4 = data[3]  #   270 degrees
I5 = data[4]  #   360 degrees
I6 = data[5]  #   450 degrees

phase_numerator = 4.0 * I4 - 3.0 * I2 - I6
phase_denominator = I1 - 4.0 * I3 + 3.0 * I5

phase = np.arctan2(phase_numerator, phase_denominator)
phase = np.ma.masked_array(phase, mask=mask)

for index, frame in enumerate(data):
    save_intermediate_png(
        session_dir / f"phase_intermediate_01_input_{index}.png",
        np.ma.masked_array(frame, mask=mask),
        f"Input frame {index}: {index * 90} degrees",
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
    session_dir / "phase_intermediate_04_wrapped.png",
    phase,
    "Wrapped phase",
    "twilight",
    vmin=-np.pi,
    vmax=np.pi,
)


# ---------------------------------------------------------------------
# Unwrap from the centre outward, preserving the original notebook logic
# ---------------------------------------------------------------------
q = np.zeros_like(phase, dtype=np.float64)

q[unwrap_center_y:, :] = np.unwrap(
    phase[unwrap_center_y:, :].filled(0.0),
    axis=0,
)
q[unwrap_center_y::-1, :] = np.unwrap(
    phase[unwrap_center_y::-1, :].filled(0.0),
    axis=0,
)
vertical_unwrapped = np.ma.masked_array(q.copy(), mask=mask)
q[:, unwrap_center_x:] = np.unwrap(q[:, unwrap_center_x:], axis=1)
q[:, unwrap_center_x::-1] = np.unwrap(q[:, unwrap_center_x::-1], axis=1)

phaseUW = np.ma.masked_array(q, mask=mask)

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


# ---------------------------------------------------------------------
# Fit J1-J15 to the wavefront before piston and tilt removal
# ---------------------------------------------------------------------
zernike_valid_mask = valid_mask & np.isfinite(phaseUW.filled(np.nan))
valid_y, valid_x = np.nonzero(zernike_valid_mask)
if valid_x.size < len(ZERNIKE_MODES):
    raise ValueError("Not enough valid ROI pixels for J1-J15 Zernike fitting.")

zernike_center_x = float(np.mean(valid_x))
zernike_center_y = float(np.mean(valid_y))
zernike_radius = float(np.max(np.hypot(
    valid_x - zernike_center_x,
    valid_y - zernike_center_y,
)))
if zernike_radius <= 0:
    raise ValueError("The ROI pupil radius must be greater than zero.")

zernike_y_grid, zernike_x_grid = np.indices((sizey, sizex))
zernike_x_normalized = (
    zernike_x_grid - zernike_center_x
) / zernike_radius
zernike_y_normalized = (
    zernike_y_grid - zernike_center_y
) / zernike_radius
zernike_rho = np.hypot(zernike_x_normalized, zernike_y_normalized)
zernike_theta = np.arctan2(
    zernike_y_normalized,
    zernike_x_normalized,
)
zernike_fit_mask = zernike_valid_mask & (zernike_rho <= 1.0)
rho_values = zernike_rho[zernike_fit_mask]
theta_values = zernike_theta[zernike_fit_mask]
zernike_design_matrix = np.column_stack([
    zernike_polynomial(n, m, rho_values, theta_values)
    for _, n, m, _ in ZERNIKE_MODES
])
wavefront_before_correction_waves = (
    phaseUW.filled(np.nan)[zernike_fit_mask] / (2.0 * np.pi)
)
zernike_coefficients_before_correction, _, zernike_rank, _ = np.linalg.lstsq(
    zernike_design_matrix,
    wavefront_before_correction_waves,
    rcond=None,
)
if zernike_rank != len(ZERNIKE_MODES):
    raise ValueError(
        f"J1-J15 Zernike fit is rank deficient: {zernike_rank}/15."
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
aberration_component_waves = np.asarray([
    np.hypot(
        zernike_coefficients_before_correction[4],
        zernike_coefficients_before_correction[5],
    ),
    np.hypot(
        zernike_coefficients_before_correction[6],
        zernike_coefficients_before_correction[7],
    ),
    np.hypot(
        zernike_coefficients_before_correction[8],
        zernike_coefficients_before_correction[9],
    ),
    abs(zernike_coefficients_before_correction[10]),
    np.hypot(
        zernike_coefficients_before_correction[11],
        zernike_coefficients_before_correction[12],
    ),
    np.hypot(
        zernike_coefficients_before_correction[13],
        zernike_coefficients_before_correction[14],
    ),
], dtype=np.float64)

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


# ---------------------------------------------------------------------
# Estimate pupil centre and construct normalised polar coordinates
# ---------------------------------------------------------------------
xg, yg = np.meshgrid(np.arange(sizex), np.arange(sizey))
xg = np.ma.masked_array(xg, mask=mask)
yg = np.ma.masked_array(yg, mask=mask)

xc = np.mean(xg)
yc = np.mean(yg)

print(f"centre x = {xc}, centre y = {yc}")

xg = (xg - xc) * 0.5 / np.std(xg - xc)
yg = (yg - yc) * 0.5 / np.std(yg - yc)

rg = np.sqrt(xg ** 2 + yg ** 2)
tg = np.arctan2(yg, xg)


# ---------------------------------------------------------------------
# Remove piston and tilt
# ---------------------------------------------------------------------
phase_zero_piston = phaseUW - np.mean(phaseUW)

Z11 = 2.0 * rg * np.cos(tg)
Z1m1 = 2.0 * rg * np.sin(tg)

z11 = np.mean(phase_zero_piston * Z11)
z1m1 = np.mean(phase_zero_piston * Z1m1)

phaseT = phase_zero_piston - z11 * Z11 - z1m1 * Z1m1

print(f"z11 = {z11}, z1m1 = {z1m1}")

wavefront_after_tilt_waves = phaseT.filled(np.nan)[zernike_fit_mask] / (2.0 * np.pi)
zernike_coefficients_after_tilt, _, zernike_after_rank, _ = np.linalg.lstsq(
    zernike_design_matrix,
    wavefront_after_tilt_waves,
    rcond=None,
)
if zernike_after_rank != len(ZERNIKE_MODES):
    raise ValueError(
        f"J1-J15 Zernike fit after piston/tilt removal is rank deficient: {zernike_after_rank}/15."
    )

zernike_after_bar_labels = [
    f"J{mode_index:02d}\n{mode_name}"
    for mode_index, _, _, mode_name in ZERNIKE_MODES
]
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


# ---------------------------------------------------------------------
# Fit selected low-order Zernike terms and plot residual surface shape
# ---------------------------------------------------------------------
Z20 = np.sqrt(3.0) * (2.0 * rg ** 2 - 1.0)
Z22 = np.sqrt(6.0) * rg ** 2 * np.cos(2.0 * tg)
Z2m2 = np.sqrt(6.0) * rg ** 2 * np.sin(2.0 * tg)
Z40 = np.sqrt(5.0) * (6.0 * rg ** 4 - 6.0 * rg ** 2 + 1.0)

z20 = np.mean(phaseT * Z20)
z22 = np.mean(phaseT * Z22)
z2m2 = np.mean(phaseT * Z2m2)
z40 = np.mean(phaseT * Z40)

print(
    f"z20 = {z20:.4f} rad, "
    f"z22 = {z22:.4f} rad, "
    f"z2m2 = {z2m2:.4f} rad, "
    f"z40 = {z40:.4f} rad"
)

residual_phase = (
    phaseT
    - z20 * Z20
    - z22 * Z22
    - z2m2 * Z2m2
    - z40 * Z40
)

residual_surface_nm = residual_phase * WAVELENGTH_NM / (4.0 * np.pi)

# For a transmissive lens, phase converts to optical path difference with
# lambda/(2*pi). Remove best-fit defocus as well as piston and tilt so the
# lens's intended focusing power is not reported as aberration.
wavefront_aberration_phase = phaseT - z20 * Z20
wavefront_aberration_nm = (
    wavefront_aberration_phase * WAVELENGTH_NM / (2.0 * np.pi)
)
aberration_values_nm = wavefront_aberration_nm.compressed()
aberration_rms_nm = float(np.sqrt(np.mean(aberration_values_nm ** 2)))
aberration_pv_nm = float(
    np.max(aberration_values_nm) - np.min(aberration_values_nm)
)

print(
    f"Lens wavefront aberration in ROI: RMS = {aberration_rms_nm:.2f} nm, "
    f"PV = {aberration_pv_nm:.2f} nm"
)

export_path = get_export_path()
valid_output_mask = ~mask
if export_path is not None:
    export_path.parent.mkdir(parents=True, exist_ok=True)
    output_arrays = {
        f"phase_{index}": np.where(valid_output_mask, frame, np.nan).astype(
            np.float32
        )
        for index, frame in enumerate(data)
    }
    output_arrays.update({
        "wrapped": phase.filled(np.nan).astype(np.float32),
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
    np.savez(export_path, **output_arrays)
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
            "title": f"Phase {index}: {index * 90} degrees",
            "cmap": "gray",
            "colorbar": "Intensity",
        })

    results.extend([
        {
            "button": "Wrapped",
            "image": phase,
            "title": "Six-frame 90° wrapped phase",
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
            "button": "Lens aberration",
            "image": wavefront_aberration_nm,
            "title": (
                "Lens wavefront aberration (piston/tilt/defocus removed)\n"
                f"RMS = {aberration_rms_nm:.2f} nm, "
                f"PV = {aberration_pv_nm:.2f} nm"
            ),
            "cmap": "RdBu_r",
            "colorbar": "Wavefront aberration (nm OPD)",
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


if not NO_VIEWER:
    show_result_viewer()
