
"""
check_measured_vs_expected_phase.py

Repeatability-improved phase checker.

Key changes
-----------
1. Uses the same FFT sideband extraction and unwrapped phase logic as the
    calibration script.
2. The program is non-interactive and can be launched automatically by
    camera.py.
3. The result figure and JSON are saved into the exact phase_capture_* folder.
"""

from pathlib import Path
import json
import sys
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Paths and constants
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
CAPTURE_DIR = BASE_DIR / "captured_frames"

REFERENCE_CONFIG_PATH = (
    BASE_DIR
    /
    "phase_check_reference_config.json"
)

EXPECTED_PHASE_DEG = np.array(
    [0.0, 90.0, 180.0, 270.0, 360.0, 450.0],
    dtype=np.float64
)

OUTPUT_FIGURE_NAME = "measured_vs_expected_phase_annotated.png"
OUTPUT_JSON_NAME = "measured_phase_check.json"

PHASE_COUNT = 6
PHASE_ERROR_TOLERANCE_DEG = 5.0


# ============================================================
# Input discovery
# ============================================================

def find_capture_folder():
    if len(sys.argv) >= 2:
        folder = Path(sys.argv[1])

        if not folder.is_absolute():
            folder = (BASE_DIR / folder).resolve()

        if not folder.exists():
            raise FileNotFoundError(
                f"Specified capture folder does not exist:\n{folder}"
            )

        return folder

    session_dirs = sorted(
        CAPTURE_DIR.glob("phase_capture_*"),
        key=lambda path: path.stat().st_mtime,
    )

    if not session_dirs:
        raise FileNotFoundError(
            f"No phase_capture_* folders found in:\n{CAPTURE_DIR}"
        )

    return session_dirs[-1]


def find_raw_stack(session_dir):
    raw_stack_files = sorted(
        session_dir.glob("*_raw_stack.npy")
    )

    if not raw_stack_files:
        raise FileNotFoundError(
            f"No *_raw_stack.npy file found in:\n{session_dir}"
        )

    return raw_stack_files[0]


# ============================================================
# Shared phase extraction
# ============================================================

def _local_patch_sum(array, y, x, radius):
    y1 = max(0, int(y) - radius)
    y2 = min(array.shape[0], int(y) + radius + 1)
    x1 = max(0, int(x) - radius)
    x2 = min(array.shape[1], int(x) + radius + 1)
    return float(np.sum(array[y1:y2, x1:x2]))


def _find_adaptive_conjugate_peak(spectrum):
    """
    Find a Fourier carrier peak without assuming a fixed fringe direction.

    The real-valued interferogram produces a conjugate pair around DC.  The
    selected candidate is therefore scored using both members of that pair,
    rather than searching only the lower half-plane.
    """
    spectrum = np.asarray(spectrum, dtype=np.float64)
    height, width = spectrum.shape
    cy = height // 2
    cx = width // 2

    yy, xx = np.ogrid[:height, :width]
    radial_distance = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    min_dimension = min(height, width)
    dc_radius = max(7, int(round(min_dimension * 0.007)))
    maximum_radius = 0.46 * min_dimension
    edge_margin = max(8, int(round(min_dimension * 0.008)))

    valid = (
        (radial_distance >= dc_radius)
        & (radial_distance <= maximum_radius)
    )
    valid[:edge_margin, :] = False
    valid[-edge_margin:, :] = False
    valid[:, :edge_margin] = False
    valid[:, -edge_margin:] = False

    search = np.log1p(np.maximum(spectrum, 0.0))
    search[~valid] = -np.inf

    valid_values = search[np.isfinite(search)]
    if valid_values.size == 0:
        raise RuntimeError("No valid Fourier carrier search area remains.")

    background_level = float(np.median(valid_values))
    background_mad = float(
        np.median(np.abs(valid_values - background_level))
    )
    robust_scale = max(1e-12, 1.4826 * background_mad)

    candidate_count = min(6000, valid_values.size)
    flat_search = search.ravel()
    candidate_indices = np.argpartition(
        flat_search,
        -candidate_count,
    )[-candidate_count:]
    candidate_indices = candidate_indices[
        np.argsort(flat_search[candidate_indices])[::-1]
    ]

    patch_radius = max(2, int(round(min_dimension * 0.0025)))
    best = None
    evaluated_centres = []

    for flat_index in candidate_indices:
        y, x = np.unravel_index(int(flat_index), search.shape)
        dy = int(y - cy)
        dx = int(x - cx)

        # Evaluate one member of each conjugate pair only.
        if dy < 0 or (dy == 0 and dx <= 0):
            continue

        conjugate_y = int(cy - dy)
        conjugate_x = int(cx - dx)
        if not (
            edge_margin <= conjugate_y < height - edge_margin
            and edge_margin <= conjugate_x < width - edge_margin
        ):
            continue

        # Avoid repeatedly scoring neighbouring pixels from the same peak.
        if any(
            (y - old_y) ** 2 + (x - old_x) ** 2
            <= (2 * patch_radius + 1) ** 2
            for old_y, old_x in evaluated_centres
        ):
            continue
        evaluated_centres.append((y, x))

        first_power = _local_patch_sum(
            spectrum,
            y,
            x,
            patch_radius,
        )
        conjugate_power = _local_patch_sum(
            spectrum,
            conjugate_y,
            conjugate_x,
            patch_radius,
        )

        if first_power <= 0 or conjugate_power <= 0:
            continue

        symmetry = min(first_power, conjugate_power) / max(
            first_power,
            conjugate_power,
        )
        pair_power = np.sqrt(first_power * conjugate_power)
        radius = float(np.hypot(dx, dy))
        peak_z = (
            min(search[y, x], search[conjugate_y, conjugate_x])
            - background_level
        ) / robust_scale

        # Pair power is primary.  Conjugate symmetry and robust prominence
        # reject aperture edges, isolated dust features and one-sided noise.
        score = (
            np.log1p(pair_power)
            + 2.0 * np.log(max(symmetry, 1e-6))
            + 0.08 * min(float(peak_z), 50.0)
            + 0.04 * np.log1p(radius)
        )

        result = {
            "score": float(score),
            "coarse_y": int(y),
            "coarse_x": int(x),
            "conjugate_y": int(conjugate_y),
            "conjugate_x": int(conjugate_x),
            "symmetry": float(symmetry),
            "peak_z": float(peak_z),
            "dc_radius": int(dc_radius),
            "patch_radius": int(patch_radius),
        }

        if best is None or result["score"] > best["score"]:
            best = result

        if len(evaluated_centres) >= 350:
            break

    if best is None:
        raise RuntimeError(
            "Adaptive Fourier carrier detection could not find a conjugate peak pair."
        )

    if best["symmetry"] < 0.12:
        raise RuntimeError(
            "Detected Fourier candidate has poor conjugate symmetry "
            f"({best['symmetry']:.3f}). Check fringe contrast and ROI."
        )

    if best["peak_z"] < 3.0:
        raise RuntimeError(
            "Detected Fourier carrier is not sufficiently prominent above background "
            f"(robust score {best['peak_z']:.2f})."
        )

    coarse_y = best["coarse_y"]
    coarse_x = best["coarse_x"]
    refine_radius = max(
        5,
        min(24, int(round(0.18 * np.hypot(coarse_x - cx, coarse_y - cy)))),
    )

    y1 = max(0, coarse_y - refine_radius)
    y2 = min(height, coarse_y + refine_radius + 1)
    x1 = max(0, coarse_x - refine_radius)
    x2 = min(width, coarse_x + refine_radius + 1)

    patch = np.maximum(spectrum[y1:y2, x1:x2], 0.0)
    local_y, local_x = np.indices(patch.shape)
    patch_floor = float(np.percentile(patch, 35.0))
    weights = np.maximum(patch - patch_floor, 0.0)

    if float(np.sum(weights)) > 0:
        peak_y_float = y1 + float(np.sum(local_y * weights) / np.sum(weights))
        peak_x_float = x1 + float(np.sum(local_x * weights) / np.sum(weights))
    else:
        peak_y_float = float(coarse_y)
        peak_x_float = float(coarse_x)

    peak_y = int(round(peak_y_float))
    peak_x = int(round(peak_x_float))
    peak_y = int(np.clip(peak_y, 0, height - 1))
    peak_x = int(np.clip(peak_x, 0, width - 1))

    peak_distance = float(np.hypot(peak_x_float - cx, peak_y_float - cy))
    if peak_distance <= dc_radius:
        raise RuntimeError(
            "Detected fringe carrier is too close to DC. Increase the number of visible fringes."
        )

    carrier_angle_deg = float(
        np.degrees(np.arctan2(peak_y_float - cy, peak_x_float - cx))
    )
    fringe_period_px = float(width / abs(peak_x_float - cx)) \
        if abs(peak_x_float - cx) >= abs(peak_y_float - cy) and abs(peak_x_float - cx) > 1e-9 \
        else float(height / abs(peak_y_float - cy))

    return {
        **best,
        "peak_y": peak_y,
        "peak_x": peak_x,
        "peak_y_float": peak_y_float,
        "peak_x_float": peak_x_float,
        "peak_distance": peak_distance,
        "carrier_angle_deg": carrier_angle_deg,
        "fringe_period_px": fringe_period_px,
        "refine_radius": int(refine_radius),
    }


def extract_complex_fields(stack):
    data = np.asarray(stack, dtype=np.float64)
    if data.ndim != 3 or data.shape[0] < 2:
        raise ValueError(
            f"Expected image stack shape (N,H,W) with N >= 2, got {data.shape}."
        )

    # Remove a per-frame scalar background before FFT.  This reduces DC leakage
    # while leaving the carrier phase unchanged.
    data = data - np.median(data, axis=(1, 2), keepdims=True)

    data_ft = np.fft.fftshift(
        np.fft.fft2(data, axes=(1, 2)),
        axes=(1, 2),
    )
    spectrum = np.abs(data_ft[0])

    height, width = spectrum.shape
    cy = height // 2
    cx = width // 2
    yy, xx = np.ogrid[:height, :width]

    peak = _find_adaptive_conjugate_peak(spectrum)
    peak_y_float = peak["peak_y_float"]
    peak_x_float = peak["peak_x_float"]
    peak_distance = peak["peak_distance"]

    # Keep the sideband mask below half the DC-to-carrier distance so it cannot
    # overlap the DC lobe.  The upper cap prevents excessively broad masks on
    # high-fringe-count images.
    mask_radius = int(np.clip(0.30 * peak_distance, 5, 34))
    mask_sigma = max(2.5, mask_radius / 2.2)

    distance = np.sqrt(
        (yy - peak_y_float) ** 2
        + (xx - peak_x_float) ** 2
    )
    mask = np.exp(-0.5 * (distance / mask_sigma) ** 2)

    complex_fields = np.fft.ifft2(
        np.fft.ifftshift(
            data_ft * mask[None, :, :],
            axes=(1, 2),
        ),
        axes=(1, 2),
    )

    info = {
        "spectrum": spectrum,
        "peak_x": int(peak["peak_x"]),
        "peak_y": int(peak["peak_y"]),
        "peak_x_float": float(peak_x_float),
        "peak_y_float": float(peak_y_float),
        "dc_x": int(cx),
        "dc_y": int(cy),
        "dc_radius": int(peak["dc_radius"]),
        "mask_radius": int(mask_radius),
        "peak_distance": float(peak_distance),
        "carrier_angle_deg": float(peak["carrier_angle_deg"]),
        "fringe_period_px": float(peak["fringe_period_px"]),
        "conjugate_symmetry": float(peak["symmetry"]),
        "robust_peak_score": float(peak["peak_z"]),
        "detection_method": "adaptive_full_plane_conjugate_pair",
    }

    print("Adaptive FFT carrier detection:")
    print(
        "  peak =",
        f"({info['peak_x_float']:.2f}, {info['peak_y_float']:.2f})",
    )
    print("  carrier angle =", f"{info['carrier_angle_deg']:.3f} deg")
    print("  fringe period =", f"{info['fringe_period_px']:.3f} px")
    print("  conjugate symmetry =", f"{info['conjugate_symmetry']:.3f}")
    print("  robust peak score =", f"{info['robust_peak_score']:.3f}")

    return complex_fields, info


def estimate_phase_curve(stack):
    complex_fields, fft_info = extract_complex_fields(stack)
    _, height, width = complex_fields.shape

    cy = height // 2
    cx = width // 2
    roi_half_size = min(200, height // 4, width // 4)
    y1 = cy - roi_half_size
    y2 = cy + roi_half_size
    x1 = cx - roi_half_size
    x2 = cx + roi_half_size

    reference = complex_fields[0, y1:y2, x1:x2]
    reference_energy = float(np.sum(np.abs(reference) ** 2))
    if reference_energy <= 0:
        raise RuntimeError("Reference sideband field has zero energy.")

    relative_phase = []
    correlation_magnitudes = []

    for field in complex_fields:
        current = field[y1:y2, x1:x2]
        numerator = np.sum(current * np.conj(reference))
        current_energy = float(np.sum(np.abs(current) ** 2))
        denominator = np.sqrt(current_energy * reference_energy)
        if denominator <= 0:
            raise RuntimeError("Failed to compute phase correlation denominator.")

        normalized = numerator / denominator
        relative_phase.append(np.angle(normalized))
        correlation_magnitudes.append(float(np.abs(normalized)))

    phase_rad = np.unwrap(np.asarray(relative_phase, dtype=float))
    phase_rad -= phase_rad[0]

    # Either member of a conjugate pair contains the same phase information
    # with opposite sign.  Normalize to the increasing falling-edge convention.
    if np.median(np.diff(phase_rad)) < 0:
        phase_rad = -phase_rad

    fft_info["phase_correlation_magnitudes"] = correlation_magnitudes
    fft_info["minimum_phase_correlation"] = float(min(correlation_magnitudes))

    return np.rad2deg(phase_rad), fft_info


# ============================================================
# Plot
# ============================================================

def create_annotated_plot(
    measured_phase_deg,
    expected_phase_deg,
    output_path,
):
    frame_indices = np.arange(
        len(measured_phase_deg)
    )

    phase_error_deg = (
        measured_phase_deg
        -
        expected_phase_deg
    )

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.plot(
        frame_indices,
        measured_phase_deg,
        marker="o",
        markersize=9,
        linewidth=2,
        label="Measured",
    )

    axis.plot(
        frame_indices,
        expected_phase_deg,
        marker="o",
        markersize=8,
        linewidth=2,
        linestyle="--",
        label="Expected",
    )

    for index, measured_value in enumerate(
        measured_phase_deg
    ):
        error_value = phase_error_deg[index]

        label_text = (
            f"{measured_value:.2f}°\n"
            f"error {error_value:+.2f}°"
        )

        offset_y = 14 if index % 2 == 0 else -34

        axis.annotate(
            label_text,
            xy=(index, measured_value),
            xytext=(0, offset_y),
            textcoords="offset points",
            ha="center",
            va="bottom" if offset_y > 0 else "top",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.25",
                "alpha": 0.75,
            },
            arrowprops={
                "arrowstyle": "-",
                "linewidth": 0.8,
            },
        )

    axis.set_xlabel("Frame index")
    axis.set_ylabel("Cumulative phase / degree")
    axis.set_title(
        "Measured vs Expected Phase Positions"
    )
    axis.set_xticks(frame_indices)

    all_values = np.concatenate(
        [
            measured_phase_deg,
            expected_phase_deg,
        ]
    )

    y_min = min(
        -25.0,
        float(np.min(all_values)) - 45.0,
    )

    y_max = max(
        325.0,
        float(np.max(all_values)) + 55.0,
    )

    axis.set_ylim(
        y_min,
        y_max,
    )

    axis.grid(
        True,
        alpha=0.35,
    )

    axis.legend()
    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# Main
# ============================================================

def main():
    session_dir = find_capture_folder()
    raw_stack_path = find_raw_stack(session_dir)

    print("=" * 70)
    print("Repeatability-improved phase-point measurement check")
    print("=" * 70)
    print("Capture folder:")
    print(session_dir)
    print("\nRaw stack:")
    print(raw_stack_path)

    data = np.load(
        raw_stack_path
    )

    print("\nData shape:", data.shape)
    print("Data dtype:", data.dtype)

    measured_phase_deg, diagnostics = (
        estimate_phase_curve(data)
    )

    phase_error_deg = (
        measured_phase_deg
        -
        EXPECTED_PHASE_DEG
    )

    actual_steps_deg = np.diff(
        measured_phase_deg
    )

    print("\n" + "=" * 70)
    print("Measured phase positions")
    print("=" * 70)

    print(
        "FFT peak:",
        f"({diagnostics['peak_x']}, {diagnostics['peak_y']})",
        "mask_radius:",
        diagnostics["mask_radius"],
        "peak_distance:",
        f"{diagnostics['peak_distance']:.3f}",
    )

    for index in range(PHASE_COUNT):
        print(
            f"Frame {index}: "
            f"measured = {measured_phase_deg[index]:.3f}°, "
            f"expected = {EXPECTED_PHASE_DEG[index]:.3f}°, "
            f"error = {phase_error_deg[index]:+.3f}°"
        )

    print("\n" + "=" * 70)
    print("Measured phase steps")
    print("=" * 70)

    for index, step_value in enumerate(
        actual_steps_deg
    ):
        print(
            f"Frame {index} -> Frame {index + 1}: "
            f"{step_value:.3f}°"
        )

    output_figure_path = (
        session_dir
        /
        OUTPUT_FIGURE_NAME
    )

    output_json_path = (
        session_dir
        /
        OUTPUT_JSON_NAME
    )

    diagnostics_for_json = {
        key: value
        for key, value in diagnostics.items()
        if key != "spectrum"
    }

    result = {
        "analysis_method": (
            "adaptive_full_plane_conjugate_pair_fft_sideband_unwrapped_phase"
        ),
        "capture_folder": str(session_dir),
        "raw_stack_path": str(raw_stack_path),
        "expected_phase_deg": (
            EXPECTED_PHASE_DEG.tolist()
        ),
        "measured_phase_deg": (
            measured_phase_deg.tolist()
        ),
        "phase_error_deg": (
            phase_error_deg.tolist()
        ),
        "measured_phase_steps_deg": (
            actual_steps_deg.tolist()
        ),
        "average_phase_step_deg": float(
            np.mean(actual_steps_deg)
        ),
        "phase_step_std_deg": float(
            np.std(actual_steps_deg)
        ),
        "maximum_absolute_phase_error_deg": float(
            np.max(
                np.abs(phase_error_deg)
            )
        ),
        "within_plus_minus_5_deg": bool(
            np.all(
                np.abs(phase_error_deg)
                <=
                PHASE_ERROR_TOLERANCE_DEG
            )
        ),
        "fft_and_phase_diagnostics": diagnostics_for_json,
        "output_figure_path": str(
            output_figure_path
        ),
    }

    with open(
        output_json_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=4,
            ensure_ascii=False,
        )

    create_annotated_plot(
        measured_phase_deg=measured_phase_deg,
        expected_phase_deg=EXPECTED_PHASE_DEG,
        output_path=output_figure_path,
    )

    print("\nSaved annotated figure:")
    print(output_figure_path)

    print("\nSaved phase-check JSON:")
    print(output_json_path)

    print(
        "\nAll six points within ±5°:",
        result["within_plus_minus_5_deg"],
    )


if __name__ == "__main__":
    main()
