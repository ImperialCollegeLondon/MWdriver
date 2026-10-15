import sys
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mwdriver.capture.phase_capture_frame_filter import (
    decide_framestamp_phase_slot,
    decide_native_phase_frame,
    decide_phase_frame_assignment,
    decide_triangle_done_action,
    select_queued_phase_slots,
    should_accept_phase_frame,
)
from mwdriver.capture.phase_capture_timestamps import (
    build_phase_timestamp_payload,
    normalize_frame_framestamp,
    normalize_frame_timestamp,
)


class PhaseCaptureFrameFilterTests(unittest.TestCase):
    def test_triangle_done_checks_finalized_state_before_reset_count(self):
        self.assertEqual(
            decide_triangle_done_action(False, 0),
            "finalized",
        )

    def test_triangle_done_waits_for_late_frames_or_finishes_complete_capture(self):
        self.assertEqual(
            decide_triangle_done_action(True, 5),
            "await_late_frames",
        )
        self.assertEqual(
            decide_triangle_done_action(True, 6),
            "finish_now",
        )

    def test_phase_assignment_decision_rejects_inactive_invalid_and_occupied(self):
        phase_frames = [None, object(), None]

        inactive = decide_phase_frame_assignment(
            False,
            0,
            phase_frames,
            phase_count=3,
        )
        invalid = decide_phase_frame_assignment(
            True,
            3,
            phase_frames,
            phase_count=3,
        )
        duplicate = decide_phase_frame_assignment(
            True,
            1,
            phase_frames,
            phase_count=3,
        )
        accepted = decide_phase_frame_assignment(
            True,
            2,
            phase_frames,
            phase_count=3,
        )

        self.assertEqual(inactive.status, "inactive")
        self.assertEqual(invalid.status, "invalid_index")
        self.assertEqual(duplicate.status, "duplicate")
        self.assertEqual(accepted.status, "accepted")

    def test_queued_frame_slots_are_empty_ordered_and_bounded(self):
        phase_frames = [None, object(), None, None, None, None]

        self.assertEqual(
            select_queued_phase_slots(
                phase_frames,
                queued_frame_count=3,
                before_slot=4,
            ),
            [0, 2, 3],
        )
        self.assertEqual(
            select_queued_phase_slots(
                phase_frames,
                queued_frame_count=2,
            ),
            [0, 2],
        )
        self.assertEqual(
            select_queued_phase_slots(
                phase_frames,
                queued_frame_count=0,
            ),
            [],
        )

    def test_framestamp_decision_rounds_and_maps_one_based_stamp_to_slot(self):
        decision = decide_framestamp_phase_slot(4.7)

        self.assertEqual(decision.status, "valid")
        self.assertEqual(decision.rounded_framestamp, 5)
        self.assertEqual(decision.phase_index, 4)

    def test_framestamp_decision_reports_missing_and_out_of_range(self):
        missing = decide_framestamp_phase_slot(None)
        too_early = decide_framestamp_phase_slot(0.4)
        too_late = decide_framestamp_phase_slot(6.6)

        self.assertEqual(missing.status, "missing")
        self.assertIsNone(missing.phase_index)
        self.assertEqual(too_early.status, "out_of_range")
        self.assertEqual(too_early.rounded_framestamp, 0)
        self.assertEqual(too_late.status, "out_of_range")
        self.assertEqual(too_late.rounded_framestamp, 7)

    def test_native_frame_decision_accepts_matching_external_acquisition(self):
        decision = decide_native_phase_frame(
            {
                "acquisition_index": 4,
                "trigger_mode": "ext",
                "acquisition_id": 12,
            },
            expected_acquisition_id=12,
        )

        self.assertEqual(decision.status, "accepted")
        self.assertEqual(decision.phase_index, 4)

    def test_native_frame_decision_falls_back_without_native_index(self):
        decision = decide_native_phase_frame(
            {"trigger_mode": "ext", "acquisition_id": 12},
            expected_acquisition_id=12,
        )

        self.assertEqual(decision.status, "fallback")
        self.assertIsNone(decision.phase_index)

    def test_native_frame_decision_rejects_wrong_mode_stale_or_invalid_index(self):
        base = {
            "acquisition_index": 2,
            "trigger_mode": "ext",
            "acquisition_id": 12,
        }
        wrong_mode = decide_native_phase_frame(
            {**base, "trigger_mode": "live"}, 12
        )
        stale = decide_native_phase_frame(
            {**base, "acquisition_id": 11}, 12
        )
        invalid = decide_native_phase_frame(
            {**base, "acquisition_index": 6}, 12
        )

        self.assertEqual(wrong_mode.status, "wrong_trigger_mode")
        self.assertEqual(stale.status, "stale_acquisition")
        self.assertEqual(invalid.status, "invalid_index")

    def test_rejects_missing_or_complete_capture(self):
        self.assertFalse(should_accept_phase_frame(None, 5.0, 1.0))
        self.assertFalse(should_accept_phase_frame(6, 5.0, 1.0))

    def test_first_frame_waits_for_stale_frame_delay(self):
        self.assertFalse(should_accept_phase_frame(0, 2.599, 1.0))
        self.assertTrue(should_accept_phase_frame(0, 2.6, 1.0))

    def test_later_frames_wait_for_grace_period(self):
        self.assertFalse(should_accept_phase_frame(1, 1.249, 1.0))
        self.assertTrue(should_accept_phase_frame(1, 1.25, 1.0))
        self.assertFalse(should_accept_phase_frame(1, 0.9, 1.0))

    def test_missing_mode_entry_time_accepts_incomplete_capture(self):
        self.assertTrue(should_accept_phase_frame(3, 0.0, None))


class PhaseCaptureTimestampTests(unittest.TestCase):
    def test_normalizes_supported_timestamp_shapes(self):
        self.assertEqual(normalize_frame_timestamp(12), 12.0)
        self.assertEqual(normalize_frame_timestamp("12.5"), 12.5)
        self.assertEqual(normalize_frame_timestamp({"camera_timestamp": "13"}), 13.0)
        self.assertEqual(
            normalize_frame_timestamp(SimpleNamespace(timestamp=14)),
            14.0,
        )
        self.assertIsNone(normalize_frame_timestamp("not-a-time"))
        self.assertIsNone(normalize_frame_timestamp(True))

    def test_framestamp_uses_only_framestamp_field(self):
        self.assertEqual(normalize_frame_framestamp({"framestamp": 7}), 7.0)
        self.assertIsNone(normalize_frame_framestamp({"timestamp": 7}))

    def test_payload_preserves_camera_metadata_and_host_time(self):
        payload = build_phase_timestamp_payload(
            {"info": {"camera_timestamp": 1234, "framestamp": 8}},
            host_time_s=1.0,
            phase_index=2,
        )

        self.assertEqual(payload["phase_index"], 2)
        self.assertEqual(payload["camera_timestamp"], 1234.0)
        self.assertEqual(payload["framestamp"], 8.0)
        self.assertEqual(payload["host_time_s"], 1.0)
        self.assertEqual(payload["host_time_iso"], datetime.fromtimestamp(1.0).isoformat())

    def test_payload_falls_back_to_phase_slot_for_missing_framestamp(self):
        payload = build_phase_timestamp_payload(
            {"timestamp": "2.5"},
            host_time_s=3.0,
            phase_index=4,
        )

        self.assertEqual(payload["camera_timestamp"], 2.5)
        self.assertEqual(payload["framestamp"], 5.0)


if __name__ == "__main__":
    unittest.main()