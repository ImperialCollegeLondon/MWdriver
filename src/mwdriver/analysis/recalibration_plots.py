"""Combined phase verification plots for a recalibration run."""

from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def save_recalibration_iterations_plot(history, session_dir):
    output_path = _save_iterations_plot(history, session_dir, show_error=False)
    _save_iterations_plot(history, session_dir, show_error=True)
    _save_iterations_plot(history, session_dir, show_time=True)
    return output_path


def _save_iterations_plot(history, session_dir, *, show_error=False, show_time=False):
    figure = Figure(figsize=(10, 6), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    expected = np.asarray(history[0]["expected_phase_deg"], dtype=float)
    if not show_time:
        axis.plot(
            expected,
            np.zeros_like(expected) if show_error else expected,
            "k--",
            linewidth=2,
            label="Zero error" if show_error else "Expected",
        )
    for index, record in enumerate(history):
        if show_time and not record.get("trigger_positions_ms"):
            continue
        measured = np.asarray(record["measured_phase_deg"], dtype=float)
        measured = measured - measured[0]
        axis.plot(
            measured if show_time else record["expected_phase_deg"],
            record["trigger_positions_ms"] if show_time else (
                measured - np.asarray(record["expected_phase_deg"]) if show_error else measured
            ),
            "o-",
            label=record["label"],
            color=colormaps["turbo"](index / max(1, len(history) - 1)),
            linewidth=2 if index == len(history) - 1 else 1,
        )
    axis.set_xlabel("Expected phase (deg)")
    axis.set_ylabel("Measured - expected phase (deg)" if show_error else "Measured phase (deg)")
    axis.set_title("Recalibration: phase error" if show_error else "Recalibration: measured vs expected phase")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    filename = "recalibration_iteration_errors.png" if show_error else "recalibration_iterations.png"
    if show_time:
        axis.set_xlabel("Measured phase (deg)")
        axis.set_ylabel("Requested trigger time from falling-edge start (ms)")
        axis.set_title("Recalibration: requested trigger time vs measured phase")
        filename = "recalibration_iterations_time_vs_phase.png"
    output_path = Path(session_dir) / filename
    figure.savefig(output_path, dpi=150)
    return output_path