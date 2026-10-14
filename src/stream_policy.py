"""Pure camera stream-mode selection policy."""


def get_camera_stream_mode(
    streaming_allowed,
    preview_layer_visible,
    camera_mode,
    phase_capture_active,
    recalibration_active,
):
    if phase_capture_active and camera_mode == "external":
        return "capture"
    if (
        streaming_allowed
        and preview_layer_visible
        and camera_mode == "live"
        and not phase_capture_active
        and not recalibration_active
    ):
        return "preview"
    return "stopped"
