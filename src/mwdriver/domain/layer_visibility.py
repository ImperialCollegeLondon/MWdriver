"""Pure visibility plans for camera and analysis layer modes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerVisibilityPlan:
    camera_visible: bool
    roi_visible: bool
    analysis_layers_visible: tuple[bool, ...]
    time_series_layers_visible: tuple[bool, ...]


def build_layer_visibility_plan(
    mode,
    analysis_layers,
    time_series_layers,
    selected_layer=None,
):
    if mode not in {"preview", "roi", "analysis", "time_series"}:
        raise ValueError(f"Unknown layer visibility mode: {mode}")

    show_analysis = mode == "analysis"
    show_time_series = mode == "time_series"
    return LayerVisibilityPlan(
        camera_visible=mode in {"preview", "roi"},
        roi_visible=mode != "analysis",
        analysis_layers_visible=tuple(
            show_analysis and layer is selected_layer
            for layer in analysis_layers
        ),
        time_series_layers_visible=tuple(
            show_time_series and layer is selected_layer
            for layer in time_series_layers
        ),
    )
