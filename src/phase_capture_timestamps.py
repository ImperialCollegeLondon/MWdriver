import math
from typing import Any


def _extract_from_mapping_or_object(source: Any, candidate_keys: tuple[str, ...]) -> float | None:
    if source is None:
        return None

    if isinstance(source, dict):
        for key in candidate_keys:
            if key in source:
                return normalize_frame_timestamp(source[key], candidate_keys=candidate_keys)
        return None

    for attr in candidate_keys:
        if hasattr(source, attr):
            value = getattr(source, attr)
            if value is not None:
                normalized = normalize_frame_timestamp(value, candidate_keys=candidate_keys)
                if normalized is not None:
                    return normalized

    if hasattr(source, "__dict__"):
        for attr in candidate_keys:
            if attr in getattr(source, "__dict__", {}):
                value = getattr(source, attr)
                if value is not None:
                    normalized = normalize_frame_timestamp(value, candidate_keys=candidate_keys)
                    if normalized is not None:
                        return normalized

    return None


def normalize_frame_timestamp(source: Any, candidate_keys: tuple[str, ...] = ("camera_timestamp", "timestamp", "frame_timestamp", "time", "framestamp")) -> float | None:
    """Best-effort extraction of a numeric frame timestamp.

    Supports a few common forms:
    - plain numbers
    - dicts with timestamp-like keys
    - objects exposing a timestamp attribute or a comparable field
    - strings that can be converted to float
    """

    if source is None:
        return None

    if isinstance(source, (int, float)) and not isinstance(source, bool):
        return float(source)

    if isinstance(source, str):
        try:
            return float(source)
        except ValueError:
            return None

    if isinstance(source, dict):
        for key in candidate_keys:
            if key in source:
                return normalize_frame_timestamp(source[key], candidate_keys=candidate_keys)
        return None

    extracted = _extract_from_mapping_or_object(source, candidate_keys)
    if extracted is not None:
        return extracted

    return None


def normalize_frame_framestamp(source: Any) -> float | None:
    return normalize_frame_timestamp(source, candidate_keys=("framestamp",))


def build_phase_timestamp_payload(frame, host_time_s: float, phase_index: int) -> dict[str, Any]:
    """Collect camera and host timestamps for one phase frame."""
    frame_info = None
    try:
        if isinstance(frame, dict):
            frame_info = frame.get("info")
        else:
            frame_info = getattr(frame, "info", None)
    except Exception:
        frame_info = None

    if frame_info is None:
        frame_info = frame

    frame_timestamp = None
    try:
        frame_timestamp = normalize_frame_timestamp(
            frame_info,
            candidate_keys=("camera_timestamp", "timestamp", "frame_timestamp", "time"),
        )
    except Exception:
        frame_timestamp = None

    if frame_timestamp is None:
        try:
            frame_timestamp = normalize_frame_timestamp(
                getattr(frame, "timestamp", None),
                candidate_keys=("camera_timestamp", "timestamp", "frame_timestamp", "time"),
            )
        except Exception:
            frame_timestamp = None

    framestamp = None
    try:
        framestamp = normalize_frame_timestamp(
            frame_info,
            candidate_keys=("framestamp",),
        )
    except Exception:
        framestamp = None

    if framestamp is None:
        try:
            framestamp = normalize_frame_timestamp(
                getattr(frame, "framestamp", None),
                candidate_keys=("framestamp",),
            )
        except Exception:
            framestamp = None

    # If the camera does not supply a framestamp, fall back to the phase index.
    if framestamp is None:
        framestamp = float(phase_index + 1)

    return {
        "phase_index": phase_index,
        "camera_timestamp": frame_timestamp,
        "framestamp": framestamp,
        "host_time_s": host_time_s,
        "host_time_iso": __import__("datetime").datetime.fromtimestamp(host_time_s).isoformat(),
    }
