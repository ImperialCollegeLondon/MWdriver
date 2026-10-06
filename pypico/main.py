from machine import Pin, PWM
from utime import (
    sleep,
    sleep_ms,
    sleep_us,
    ticks_ms,
    ticks_us,
    ticks_add,
    ticks_diff,
)
import sys
import select
import gc


# ============================================================
# Pin configuration
# ============================================================

OUTPUT_PIN = 15
BUZZER_PIN = 27
TRIGGER_PIN = 12


# ============================================================
# Waveform configuration
# ============================================================

PWM_FREQ_HZ = 10_000

# Triangle full period:
# rising edge  = 650 ms
# falling edge = 650 ms
HALF_PERIOD_MS = 650

# Absolute waveform update interval.
WAVEFORM_STEP_US = 1000

MAX_DUTY = 65535
MIN_DUTY = 0


# ============================================================
# Phase capture configuration
# ============================================================

PHASE_CAPTURE_COUNT = 6

FALLING_EDGE_PHASE_CYCLES = 2.25
TARGET_CAPTURE_CYCLES = 1.0
CENTER_CAPTURE_WINDOW = True


# ============================================================
# Fixed trigger position configuration
#
# Important:
# Once calculated, the six positions are stored explicitly.
#
# They are measured from the beginning of the falling edge.
#
# Current original calculation:
#
# capture_window_ms = int(650 / 2.25) = 288 ms
# capture_window_start = (650 - 288) / 2 = 181 ms
# phase step = 288 / 6 = 48 ms
#
# Therefore:
# 181, 229, 277, 325, 373, 421 ms
# ============================================================

USE_FIXED_TRIGGER_POSITIONS = True

FIXED_TRIGGER_POSITIONS_MS = [
    181,
    229,
    277,
    325,
    373,
    421,
]

# Active trigger positions can be updated from PC at runtime.
ACTIVE_TRIGGER_POSITIONS_MS = list(
    FIXED_TRIGGER_POSITIONS_MS
)


# ============================================================
# Camera trigger configuration
# ============================================================

TRIGGER_PULSE_MS = 5
TRIGGER_PULSE_US = TRIGGER_PULSE_MS * 1000

CAMERA_READY_DELAY_MS = 1200

PC_START_COMMAND = "PC_START_PHASE"
PC_SET_TRIGGER_POSITIONS_MS_PREFIX = "SET_TRIGGER_POSITIONS_MS:"


# ============================================================
# Absolute timing configuration
# ============================================================

# When a deadline is still far away, sleep for most of the
# remaining time. The final short interval is handled by a
# busy wait for improved timing repeatability.
COARSE_SLEEP_THRESHOLD_US = 1500
COARSE_SLEEP_MARGIN_US = 500

# Collect timing diagnostics for each trigger.
ENABLE_TIMING_DIAGNOSTICS = True


# ============================================================
# Hardware setup
# ============================================================

output_pin = Pin(
    OUTPUT_PIN,
    Pin.OUT
)
output_pin.value(0)

trigger = Pin(
    TRIGGER_PIN,
    Pin.OUT
)
trigger.value(0)

buzzer = PWM(
    Pin(BUZZER_PIN)
)
buzzer.freq(1000)
buzzer.duty_u16(0)


# ============================================================
# Busy / debounce configuration
# ============================================================

last_press_time = 0
debounce_time = 300
is_busy = False


# ============================================================
# Non-blocking USB serial command setup
# ============================================================

poll = select.poll()
poll.register(
    sys.stdin,
    select.POLLIN
)


def pc_command_available():
    try:
        return bool(
            poll.poll(0)
        )
    except Exception:
        return False


def read_pc_command():
    try:
        return sys.stdin.readline().strip()
    except Exception:
        return ""


# ============================================================
# Helper functions
# ============================================================

def beep(duration=0.08):
    buzzer.duty_u16(1000)
    sleep(duration)
    buzzer.duty_u16(0)


def wait_until_us(deadline_us):
    """
    Wait until an absolute ticks_us deadline.

    This avoids cumulative timing error.

    The old method:

        sleep_ms(1)
        sleep_ms(1)
        sleep_ms(1)

    accumulates the execution time of every loop.

    This method always compares against one absolute timeline.
    """

    while True:
        remaining_us = ticks_diff(
            deadline_us,
            ticks_us()
        )

        if remaining_us <= 0:
            return

        if remaining_us > COARSE_SLEEP_THRESHOLD_US:
            sleep_time_us = (
                remaining_us
                -
                COARSE_SLEEP_MARGIN_US
            )

            if sleep_time_us > 0:
                sleep_us(
                    sleep_time_us
                )

        elif remaining_us > 100:
            sleep_us(50)

        else:
            # Final busy wait.
            pass


def build_rising_duty_table():
    """
    Precalculate the complete rising-edge duty sequence.

    Precalculation prevents repeated arithmetic during the
    timing-critical waveform output.
    """

    duties = []

    for step in range(
        HALF_PERIOD_MS
    ):
        duty = int(
            round(
                MAX_DUTY
                *
                (step + 1)
                /
                HALF_PERIOD_MS
            )
        )

        duty = max(
            MIN_DUTY,
            min(
                duty,
                MAX_DUTY
            )
        )

        duties.append(
            duty
        )

    return duties


def build_falling_duty_table():
    """
    Precalculate the complete falling-edge duty sequence.
    """

    duties = []

    for step in range(
        HALF_PERIOD_MS
    ):
        duty = int(
            round(
                MAX_DUTY
                *
                (
                    1.0
                    -
                    (step + 1)
                    /
                    HALF_PERIOD_MS
                )
            )
        )

        duty = max(
            MIN_DUTY,
            min(
                duty,
                MAX_DUTY
            )
        )

        duties.append(
            duty
        )

    return duties


RISING_DUTY_TABLE = build_rising_duty_table()
FALLING_DUTY_TABLE = build_falling_duty_table()


def calculate_trigger_positions():
    """
    Calculate the six trigger positions.

    When USE_FIXED_TRIGGER_POSITIONS is True, the stored positions
    are used directly. This ensures that the same code execution
    always uses exactly the same nominal trigger times.
    """

    capture_window_ms = int(
        HALF_PERIOD_MS
        *
        TARGET_CAPTURE_CYCLES
        /
        FALLING_EDGE_PHASE_CYCLES
    )

    if capture_window_ms >= HALF_PERIOD_MS:
        capture_window_ms = HALF_PERIOD_MS

    if CENTER_CAPTURE_WINDOW:
        capture_window_start = int(
            (
                HALF_PERIOD_MS
                -
                capture_window_ms
            )
            /
            2
        )
    else:
        capture_window_start = 0

    if USE_FIXED_TRIGGER_POSITIONS:
        trigger_positions_ms = list(
            ACTIVE_TRIGGER_POSITIONS_MS
        )

    else:
        phase_step_ms = (
            capture_window_ms
            /
            PHASE_CAPTURE_COUNT
        )

        trigger_positions_ms = [
            int(
                round(
                    capture_window_start
                    +
                    phase_index
                    *
                    phase_step_ms
                )
            )
            for phase_index in range(
                PHASE_CAPTURE_COUNT
            )
        ]

    trigger_positions_ms = [
        max(
            0,
            min(
                int(position),
                HALF_PERIOD_MS - 1
            )
        )
        for position in trigger_positions_ms
    ]

    if len(trigger_positions_ms) != PHASE_CAPTURE_COUNT:
        raise ValueError(
            "Exactly six trigger positions are required."
        )

    for index in range(
        1,
        len(trigger_positions_ms)
    ):
        if (
            trigger_positions_ms[index]
            <=
            trigger_positions_ms[index - 1]
        ):
            raise ValueError(
                "Trigger positions must be strictly increasing."
            )

    minimum_spacing_ms = min(
        trigger_positions_ms[index + 1]
        -
        trigger_positions_ms[index]
        for index in range(
            PHASE_CAPTURE_COUNT - 1
        )
    )

    if minimum_spacing_ms <= TRIGGER_PULSE_MS:
        raise ValueError(
            "Trigger spacing must be greater than trigger pulse width."
        )

    return (
        capture_window_ms,
        capture_window_start,
        trigger_positions_ms
    )


def validate_trigger_positions_ms(trigger_positions_ms):
    if len(trigger_positions_ms) != PHASE_CAPTURE_COUNT:
        raise ValueError(
            "Exactly six trigger positions are required."
        )

    validated = [
        max(
            0,
            min(
                int(value),
                HALF_PERIOD_MS - 1
            )
        )
        for value in trigger_positions_ms
    ]

    for index in range(
        1,
        len(validated)
    ):
        if validated[index] <= validated[index - 1]:
            raise ValueError(
                "Trigger positions must be strictly increasing."
            )

    minimum_spacing_ms = min(
        validated[index + 1]
        -
        validated[index]
        for index in range(
            PHASE_CAPTURE_COUNT - 1
        )
    )

    if minimum_spacing_ms <= TRIGGER_PULSE_MS:
        raise ValueError(
            "Trigger spacing must be greater than trigger pulse width."
        )

    return validated


def apply_trigger_positions_from_pc_command(cmd):
    global ACTIVE_TRIGGER_POSITIONS_MS

    if not cmd.startswith(
        PC_SET_TRIGGER_POSITIONS_MS_PREFIX
    ):
        return False

    raw_text = cmd.split(
        ":",
        1
    )[1].strip()

    try:
        parsed = [
            int(part.strip())
            for part in raw_text.split(",")
            if part.strip()
        ]

        validated = validate_trigger_positions_ms(parsed)

        ACTIVE_TRIGGER_POSITIONS_MS = validated

        print(
            "TRIGGER_POSITIONS_UPDATED_MS:",
            ACTIVE_TRIGGER_POSITIONS_MS
        )

        return True

    except Exception as error:
        print(
            "TRIGGER_POSITIONS_UPDATE_ERROR:",
            error
        )
        return True


def output_timing_configuration(
    capture_window_ms,
    capture_window_start,
    trigger_positions_ms
):
    """
    Print configuration before the timing-critical waveform starts.
    No serial printing occurs while the triangle wave is running.
    """

    capture_window_end = (
        capture_window_start
        +
        capture_window_ms
    )

    print("PHASE_CAPTURE_CONFIG_START")
    print(
        "HALF_PERIOD_MS:",
        HALF_PERIOD_MS
    )
    print(
        "WAVEFORM_STEP_US:",
        WAVEFORM_STEP_US
    )
    print(
        "FALLING_EDGE_PHASE_CYCLES:",
        FALLING_EDGE_PHASE_CYCLES
    )
    print(
        "TARGET_CAPTURE_CYCLES:",
        TARGET_CAPTURE_CYCLES
    )
    print(
        "USE_FIXED_TRIGGER_POSITIONS:",
        USE_FIXED_TRIGGER_POSITIONS
    )
    print(
        "capture_window_ms:",
        capture_window_ms
    )
    print(
        "capture_window_start:",
        capture_window_start
    )
    print(
        "capture_window_end:",
        capture_window_end
    )
    print(
        "trigger_positions_ms:",
        trigger_positions_ms
    )
    print(
        "TRIGGER_PULSE_MS:",
        TRIGGER_PULSE_MS
    )
    print(
        "timing_method:",
        "absolute ticks_us deadline scheduling"
    )
    print(
        "serial_during_waveform:",
        "disabled"
    )
    print("PHASE_CAPTURE_CONFIG_END")


def output_one_triangle_cycle_with_hardware_triggers():
    """
    Output one complete triangle cycle using absolute timing.

    Important improvements:

    1. No cumulative sleep_ms(1) timing.
    2. No print statements during waveform generation.
    3. Trigger timing is referenced directly to falling_edge_start_us.
    4. Garbage collection is disabled during the critical waveform.
    5. Actual trigger timing errors are recorded and printed afterwards.
    """

    global output_pin

    # ========================================================
    # 1. Tell PC to prepare external trigger mode
    # ========================================================

    print("TRIANGLE_START")

    trigger.value(0)

    sleep_ms(
        CAMERA_READY_DELAY_MS
    )

    # ========================================================
    # 2. Calculate trigger positions
    # ========================================================

    (
        capture_window_ms,
        capture_window_start,
        trigger_positions_ms
    ) = calculate_trigger_positions()

    output_timing_configuration(
        capture_window_ms,
        capture_window_start,
        trigger_positions_ms
    )

    trigger_positions_us = [
        position_ms * 1000
        for position_ms in trigger_positions_ms
    ]

    # Store actual trigger timing errors.
    trigger_timing_errors_us = [
        None
    ] * PHASE_CAPTURE_COUNT

    # ========================================================
    # 3. Prepare PWM
    # ========================================================

    pwm = PWM(
        Pin(OUTPUT_PIN)
    )

    pwm.freq(
        PWM_FREQ_HZ
    )

    pwm.duty_u16(
        MIN_DUTY
    )

    trigger.value(0)

    # Run garbage collection before entering timing-critical code.
    gc.collect()

    gc_was_enabled = gc.isenabled()

    if gc_was_enabled:
        gc.disable()

    try:
        # ====================================================
        # 4. Rising edge
        # ====================================================

        rising_start_us = ticks_us()

        for step in range(
            HALF_PERIOD_MS
        ):
            step_deadline_us = ticks_add(
                rising_start_us,
                step * WAVEFORM_STEP_US
            )

            wait_until_us(
                step_deadline_us
            )

            pwm.duty_u16(
                RISING_DUTY_TABLE[step]
            )

        # Ensure the final rising-edge value is exactly MAX.
        rising_end_deadline_us = ticks_add(
            rising_start_us,
            HALF_PERIOD_MS * WAVEFORM_STEP_US
        )

        wait_until_us(
            rising_end_deadline_us
        )

        pwm.duty_u16(
            MAX_DUTY
        )

        # ====================================================
        # 5. Falling edge
        # ====================================================

        falling_start_us = ticks_us()

        next_trigger_index = 0
        trigger_active = False
        trigger_off_deadline_us = None

        for step in range(
            HALF_PERIOD_MS
        ):
            step_deadline_us = ticks_add(
                falling_start_us,
                step * WAVEFORM_STEP_US
            )

            wait_until_us(
                step_deadline_us
            )

            # Update triangle duty.
            pwm.duty_u16(
                FALLING_DUTY_TABLE[step]
            )

            current_time_us = ticks_us()

            # End trigger pulse when its absolute off deadline is reached.
            if (
                trigger_active
                and
                ticks_diff(
                    current_time_us,
                    trigger_off_deadline_us
                )
                >=
                0
            ):
                trigger.value(0)
                trigger_active = False
                trigger_off_deadline_us = None

            # Start next trigger at its absolute falling-edge time.
            if next_trigger_index < PHASE_CAPTURE_COUNT:
                target_trigger_deadline_us = ticks_add(
                    falling_start_us,
                    trigger_positions_us[next_trigger_index]
                )

                if (
                    ticks_diff(
                        current_time_us,
                        target_trigger_deadline_us
                    )
                    >=
                    0
                ):
                    actual_trigger_time_us = ticks_us()

                    trigger.value(1)
                    trigger_active = True

                    trigger_off_deadline_us = ticks_add(
                        actual_trigger_time_us,
                        TRIGGER_PULSE_US
                    )

                    trigger_timing_errors_us[
                        next_trigger_index
                    ] = ticks_diff(
                        actual_trigger_time_us,
                        target_trigger_deadline_us
                    )

                    next_trigger_index += 1

        # Wait for the exact end of the falling edge.
        falling_end_deadline_us = ticks_add(
            falling_start_us,
            HALF_PERIOD_MS * WAVEFORM_STEP_US
        )

        wait_until_us(
            falling_end_deadline_us
        )

        pwm.duty_u16(
            MIN_DUTY
        )

        # Finish any trigger pulse that is still high.
        if trigger_active:
            wait_until_us(
                trigger_off_deadline_us
            )

            trigger.value(0)

    finally:
        if gc_was_enabled:
            gc.enable()

        trigger.value(0)

        pwm.duty_u16(
            MIN_DUTY
        )

        sleep_ms(5)

        pwm.deinit()

        output_pin = Pin(
            OUTPUT_PIN,
            Pin.OUT
        )

        output_pin.value(0)

    # ========================================================
    # 6. Send phase markers after waveform completion
    #
    # camera.py currently assigns frames by arrival order.
    # Therefore CAPTURE_PHASE_x does not need to be transmitted
    # before each hardware trigger.
    #
    # Moving these print statements outside the waveform avoids
    # serial USB timing interference.
    # ========================================================

    for phase_index in range(
        PHASE_CAPTURE_COUNT
    ):
        print(
            "CAPTURE_PHASE_{}".format(
                phase_index
            )
        )

    # ========================================================
    # 7. Print timing diagnostics
    # ========================================================

    if ENABLE_TIMING_DIAGNOSTICS:
        print("TRIGGER_TIMING_DIAGNOSTICS_START")

        for phase_index in range(
            PHASE_CAPTURE_COUNT
        ):
            print(
                "phase_{}_target_ms:{} actual_lateness_us:{}".format(
                    phase_index,
                    trigger_positions_ms[phase_index],
                    trigger_timing_errors_us[phase_index]
                )
            )

        valid_errors = [
            error
            for error in trigger_timing_errors_us
            if error is not None
        ]

        if valid_errors:
            maximum_error_us = max(
                valid_errors
            )

            minimum_error_us = min(
                valid_errors
            )

            mean_error_us = (
                sum(valid_errors)
                /
                len(valid_errors)
            )

            print(
                "trigger_lateness_min_us:",
                minimum_error_us
            )

            print(
                "trigger_lateness_max_us:",
                maximum_error_us
            )

            print(
                "trigger_lateness_mean_us:",
                mean_error_us
            )

            print(
                "trigger_lateness_peak_to_peak_us:",
                maximum_error_us
                -
                minimum_error_us
            )

        print("TRIGGER_TIMING_DIAGNOSTICS_END")

    print("TRIANGLE_DONE")


# ============================================================
# Startup
# ============================================================

print("PICO_READY")
print(
    "Click Start Phase Capture in camera.py "
    "to start one triangle cycle."
)
print(
    "GP15: triangle wave output for phase shifter."
)
print(
    "GP12: camera hardware trigger output."
)
print(
    "HALF_PERIOD_MS:",
    HALF_PERIOD_MS
)
print(
    "WAVEFORM_STEP_US:",
    WAVEFORM_STEP_US
)
print(
    "TRIGGER_PULSE_MS:",
    TRIGGER_PULSE_MS
)
print(
    "CAMERA_READY_DELAY_MS:",
    CAMERA_READY_DELAY_MS
)
print(
    "PC_START_COMMAND:",
    PC_START_COMMAND
)
print(
    "PHASE_CAPTURE_COUNT:",
    PHASE_CAPTURE_COUNT
)
print(
    "USE_FIXED_TRIGGER_POSITIONS:",
    USE_FIXED_TRIGGER_POSITIONS
)
print(
    "FIXED_TRIGGER_POSITIONS_MS:",
    FIXED_TRIGGER_POSITIONS_MS
)
print(
    "ACTIVE_TRIGGER_POSITIONS_MS:",
    ACTIVE_TRIGGER_POSITIONS_MS
)
print(
    "FALLING_EDGE_PHASE_CYCLES:",
    FALLING_EDGE_PHASE_CYCLES
)
print(
    "TARGET_CAPTURE_CYCLES:",
    TARGET_CAPTURE_CYCLES
)
print(
    "CENTER_CAPTURE_WINDOW:",
    CENTER_CAPTURE_WINDOW
)
print(
    "Timing method: absolute ticks_us scheduling"
)

beep(0.08)
sleep(0.08)
beep(0.08)


# ============================================================
# Main loop
# ============================================================

while True:
    if pc_command_available():
        cmd = read_pc_command()

        if cmd:
            print(
                "PC_COMMAND_RECEIVED:",
                cmd
            )

        if apply_trigger_positions_from_pc_command(cmd):
            sleep_ms(10)
            continue

        if cmd == PC_START_COMMAND:
            now = ticks_ms()

            if (
                ticks_diff(
                    now,
                    last_press_time
                )
                >
                debounce_time
                and
                not is_busy
            ):
                last_press_time = now
                is_busy = True

                try:
                    print("START_FROM_PC_BUTTON")
                    beep(0.05)

                    output_one_triangle_cycle_with_hardware_triggers()

                except Exception as error:
                    trigger.value(0)

                    output_pin = Pin(
                        OUTPUT_PIN,
                        Pin.OUT
                    )
                    output_pin.value(0)

                    print(
                        "PHASE_CAPTURE_ERROR:",
                        error
                    )

                finally:
                    is_busy = False

            else:
                print(
                    "PC_START_PHASE ignored: "
                    "busy or debounce active"
                )

    sleep_ms(10)