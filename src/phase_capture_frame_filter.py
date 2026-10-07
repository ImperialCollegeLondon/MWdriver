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
