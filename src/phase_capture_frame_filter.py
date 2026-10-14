from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class NativePhaseFrameDecision:
    status: str
    phase_index: int | None = None


def decide_native_phase_frame(frame_meta, expected_acquisition_id, phase_count=6):
    if not isinstance(frame_meta, dict):
        return NativePhaseFrameDecision("fallback")

    acquisition_index = frame_meta.get("acquisition_index")
    if acquisition_index is None:
        return NativePhaseFrameDecision("fallback")
    if frame_meta.get("trigger_mode") != "ext":
        return NativePhaseFrameDecision("wrong_trigger_mode")
    if frame_meta.get("acquisition_id") != expected_acquisition_id:
        return NativePhaseFrameDecision("stale_acquisition")
    if (
        not isinstance(acquisition_index, Integral)
        or not 0 <= acquisition_index < phase_count
    ):
        return NativePhaseFrameDecision("invalid_index")
    return NativePhaseFrameDecision("accepted", acquisition_index)


def should_accept_phase_frame(
    phase_capture_count,
    frame_arrival_time_s,
    phase_capture_mode_enter_time_s,
    grace_period_s=0.25,
    first_frame_min_delay_s=1.6,
):
    """Return True when a newly arrived frame should be accepted as a phase capture.

    During the initial period after switching the camera into external-trigger mode,
    a stale frame from the previous live-view state can still be delivered. Those
    frames must be ignored so they do not consume one of the six real phase slots.
    """

    if phase_capture_count is None:
        return False

    if phase_capture_count >= 6:
        return False

    if phase_capture_mode_enter_time_s is None:
        return True

    elapsed_s = frame_arrival_time_s - phase_capture_mode_enter_time_s

    # The first valid hardware-triggered frame should arrive well after
    # camera mode switching and Pico pre-trigger delays. Rejecting very early
    # frames avoids stale-buffer frames being mis-assigned as phase 0.
    if phase_capture_count == 0 and elapsed_s < first_frame_min_delay_s:
        return False

    return elapsed_s >= grace_period_s
