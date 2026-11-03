from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from time_study.timer import Observation, OperationTimer, TimerState, TimingConfig


class OperationTimerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = TimingConfig(
            entry_confirm_seconds=0.25,
            exit_confirm_seconds=0.25,
            lost_grace_seconds=0.50,
        )
        self.timer = OperationTimer(self.config)
        self.wall = datetime(2026, 10, 14, 9, 0, tzinfo=timezone.utc)

    def step(self, observation: Observation, seconds: float):
        return self.timer.step(
            observation,
            now_mono=seconds,
            wall_time=self.wall + timedelta(seconds=seconds),
        )

    def test_idle_when_no_hand_is_inside(self) -> None:
        snapshot = self.step(Observation.OUTSIDE, 0.0)

        self.assertEqual(snapshot.state, TimerState.IDLE)
        self.assertEqual(snapshot.total_accumulated_seconds, 0.0)

    def test_entry_requires_confirmation_before_active(self) -> None:
        first = self.step(Observation.INSIDE, 0.0)
        second = self.step(Observation.INSIDE, 0.20)
        third = self.step(Observation.INSIDE, 0.26)

        self.assertEqual(first.state, TimerState.ENTER_PENDING)
        self.assertEqual(second.state, TimerState.ENTER_PENDING)
        self.assertEqual(third.state, TimerState.ACTIVE)
        self.assertEqual(third.active_duration_seconds, 0.0)

    def test_confirmed_exit_excludes_confirmation_delay(self) -> None:
        self.step(Observation.INSIDE, 0.0)
        self.step(Observation.INSIDE, 0.30)
        pending = self.step(Observation.OUTSIDE, 2.00)
        completed = self.step(Observation.OUTSIDE, 2.26).completed_interval

        self.assertEqual(pending.state, TimerState.EXIT_PENDING)
        self.assertIsNotNone(completed)
        self.assertAlmostEqual(completed.active_duration_seconds, 1.70, places=3)
        self.assertAlmostEqual(completed.total_accumulated_seconds, 1.70, places=3)

    def test_brief_detection_loss_resumes_same_interval(self) -> None:
        self.step(Observation.INSIDE, 0.0)
        self.step(Observation.INSIDE, 0.30)
        pending = self.step(Observation.UNKNOWN, 1.00)
        resumed = self.step(Observation.INSIDE, 1.30)

        self.assertEqual(pending.state, TimerState.EXIT_PENDING)
        self.assertEqual(resumed.state, TimerState.ACTIVE)
        self.assertIsNone(resumed.completed_interval)
        self.assertAlmostEqual(resumed.active_duration_seconds, 1.0, places=3)

    def test_detection_loss_confirmed_excludes_grace_period(self) -> None:
        self.step(Observation.INSIDE, 0.0)
        self.step(Observation.INSIDE, 0.30)
        self.step(Observation.UNKNOWN, 1.00)
        completed = self.step(Observation.UNKNOWN, 1.51).completed_interval

        self.assertIsNotNone(completed)
        self.assertEqual(self.timer.state, TimerState.IDLE)
        self.assertAlmostEqual(completed.active_duration_seconds, 0.70, places=3)

    def test_force_stop_during_exit_pending_keeps_original_stop_time(self) -> None:
        self.step(Observation.INSIDE, 0.0)
        self.step(Observation.INSIDE, 0.30)
        self.step(Observation.OUTSIDE, 1.00)
        completed = self.timer.force_stop(
            "video_end",
            now_mono=1.20,
            wall_time=self.wall + timedelta(seconds=1.20),
        )

        self.assertIsNotNone(completed)
        self.assertAlmostEqual(completed.active_duration_seconds, 0.70, places=3)

    def test_second_hand_inside_keeps_single_interval_active(self) -> None:
        self.step(Observation.INSIDE, 0.0)
        self.step(Observation.INSIDE, 0.30)
        self.step(Observation.INSIDE, 1.00)
        snapshot = self.step(Observation.INSIDE, 2.00)

        self.assertEqual(snapshot.state, TimerState.ACTIVE)
        self.assertEqual(snapshot.interval_count, 0)
        self.assertAlmostEqual(snapshot.active_duration_seconds, 1.70, places=3)


if __name__ == "__main__":
    unittest.main()
