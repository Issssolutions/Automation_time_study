from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class TimerState(str, Enum):
    IDLE = "IDLE"
    ENTER_PENDING = "ENTER_PENDING"
    ACTIVE = "ACTIVE"
    EXIT_PENDING = "EXIT_PENDING"


class Observation(str, Enum):
    INSIDE = "inside"
    OUTSIDE = "outside"
    UNKNOWN = "unknown"

@dataclass(frozen=True)
class TimingConfig:
    entry_confirm_seconds: float = 0.25
    exit_confirm_seconds: float = 0.25
    lost_grace_seconds: float = 0.50

    def __post_init__(self) -> None:
        for name, value in (
            ("entry_confirm_seconds", self.entry_confirm_seconds),
            ("exit_confirm_seconds", self.exit_confirm_seconds),
            ("lost_grace_seconds", self.lost_grace_seconds),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass
class CompletedInterval:
    date: str
    start_timestamp: str
    end_timestamp: str
    active_duration_seconds: float
    total_accumulated_seconds: float
    interval_number: int
    detection_status: str


@dataclass
class TimerSnapshot:
    state: TimerState
    observation: Observation
    active_duration_seconds: float
    total_accumulated_seconds: float
    interval_count: int
    message: str
    completed_interval: Optional[CompletedInterval] = None


class OperationTimer:
    """Finite-state timer based on monotonic time, independent of video FPS."""

    def __init__(self, config: TimingConfig | None = None) -> None:
        self.config = config or TimingConfig()
        self.state = TimerState.IDLE
        self.total_active_seconds = 0.0
        self.interval_count = 0

        self._enter_since_mono: float | None = None
        self._interval_start_mono: float | None = None
        self._interval_start_wall: datetime | None = None
        self._exit_pending_since_mono: float | None = None
        self._exit_candidate_stop_mono: float | None = None
        self._exit_candidate_stop_wall: datetime | None = None
        self._exit_reason: Observation | None = None

    def step(
        self,
        observation: Observation,
        now_mono: float | None = None,
        wall_time: datetime | None = None,
    ) -> TimerSnapshot:
        now_mono, wall_time = self._resolve_time(now_mono, wall_time)
        completed: CompletedInterval | None = None
        message = "waiting"

        if self.state == TimerState.IDLE:
            self._clear_entry()
            self._clear_exit()
            if observation == Observation.INSIDE:
                self.state = TimerState.ENTER_PENDING
                self._enter_since_mono = now_mono
                message = "entry confirmation pending"
                if self.config.entry_confirm_seconds == 0:
                    self._start_interval(now_mono, wall_time)
                    message = "active interval started"
            else:
                message = "idle"

        elif self.state == TimerState.ENTER_PENDING:
            if observation == Observation.INSIDE:
                assert self._enter_since_mono is not None
                entry_elapsed = now_mono - self._enter_since_mono
                if entry_elapsed >= self.config.entry_confirm_seconds:
                    self._start_interval(now_mono, wall_time)
                    message = "active interval started"
                else:
                    message = "entry confirmation pending"
            else:
                self._reset_to_idle()
                message = "entry rejected"

        elif self.state == TimerState.ACTIVE:
            if observation == Observation.INSIDE:
                message = "active"
            else:
                self._begin_exit_pending(observation, now_mono, wall_time)
                message = self._pending_message()

        elif self.state == TimerState.EXIT_PENDING:
            if observation == Observation.INSIDE:
                self._clear_exit()
                self.state = TimerState.ACTIVE
                message = "active interval resumed"
            else:
                if self._exit_reason == Observation.UNKNOWN and observation == Observation.OUTSIDE:
                    self._exit_reason = Observation.OUTSIDE
                if self._exit_is_confirmed(now_mono):
                    status = (
                        "confirmed_exit"
                        if self._exit_reason == Observation.OUTSIDE
                        else "confirmed_detection_loss"
                    )
                    completed = self._finalize_interval(status)
                    self._reset_to_idle()
                    message = status
                else:
                    message = self._pending_message()

        return self.snapshot(observation, now_mono, message, completed)

    def force_stop(
        self,
        status: str = "manual_stop",
        now_mono: float | None = None,
        wall_time: datetime | None = None,
    ) -> CompletedInterval | None:
        now_mono, wall_time = self._resolve_time(now_mono, wall_time)
        if self._interval_start_mono is None:
            self._reset_to_idle()
            return None

        if self.state != TimerState.EXIT_PENDING or self._exit_candidate_stop_mono is None:
            self._exit_candidate_stop_mono = now_mono
            self._exit_candidate_stop_wall = wall_time
        completed = self._finalize_interval(status)
        self._reset_to_idle()
        return completed

    def snapshot(
        self,
        observation: Observation,
        now_mono: float | None = None,
        message: str = "",
        completed: CompletedInterval | None = None,
    ) -> TimerSnapshot:
        now_mono = time.monotonic() if now_mono is None else now_mono
        active_duration = self.current_interval_seconds(now_mono)
        return TimerSnapshot(
            state=self.state,
            observation=observation,
            active_duration_seconds=active_duration,
            total_accumulated_seconds=self.total_active_seconds + active_duration,
            interval_count=self.interval_count,
            message=message,
            completed_interval=completed,
        )

    def current_interval_seconds(self, now_mono: float | None = None) -> float:
        if self._interval_start_mono is None:
            return 0.0

        now_mono = time.monotonic() if now_mono is None else now_mono
        if self.state == TimerState.EXIT_PENDING and self._exit_candidate_stop_mono is not None:
            end_mono = self._exit_candidate_stop_mono
        else:
            end_mono = now_mono

        return max(0.0, end_mono - self._interval_start_mono)

    def _start_interval(self, now_mono: float, wall_time: datetime) -> None:
        self.state = TimerState.ACTIVE
        self._interval_start_mono = now_mono
        self._interval_start_wall = wall_time
        self._clear_entry()
        self._clear_exit()

    def _begin_exit_pending(
        self, reason: Observation, now_mono: float, wall_time: datetime
    ) -> None:
        self.state = TimerState.EXIT_PENDING
        self._exit_reason = reason
        self._exit_pending_since_mono = now_mono
        self._exit_candidate_stop_mono = now_mono
        self._exit_candidate_stop_wall = wall_time

    def _exit_is_confirmed(self, now_mono: float) -> bool:
        if self._exit_pending_since_mono is None:
            return False

        required = (
            self.config.lost_grace_seconds
            if self._exit_reason == Observation.UNKNOWN
            else self.config.exit_confirm_seconds
        )
        return (now_mono - self._exit_pending_since_mono) >= required

    def _finalize_interval(self, status: str) -> CompletedInterval:
        if self._interval_start_mono is None or self._interval_start_wall is None:
            raise RuntimeError("No active interval to finalize")

        stop_mono = self._exit_candidate_stop_mono
        stop_wall = self._exit_candidate_stop_wall
        if stop_mono is None or stop_wall is None:
            stop_mono = time.monotonic()
            stop_wall = datetime.now().astimezone()

        duration = max(0.0, stop_mono - self._interval_start_mono)
        self.total_active_seconds += duration
        self.interval_count += 1

        return CompletedInterval(
            date=stop_wall.date().isoformat(),
            start_timestamp=self._interval_start_wall.isoformat(timespec="milliseconds"),
            end_timestamp=stop_wall.isoformat(timespec="milliseconds"),
            active_duration_seconds=round(duration, 3),
            total_accumulated_seconds=round(self.total_active_seconds, 3),
            interval_number=self.interval_count,
            detection_status=status,
        )

    def _pending_message(self) -> str:
        if self._exit_reason == Observation.UNKNOWN:
            return "exit pending: detection uncertain"
        return "exit pending: outside area"

    def _clear_entry(self) -> None:
        self._enter_since_mono = None

    def _clear_exit(self) -> None:
        self._exit_pending_since_mono = None
        self._exit_candidate_stop_mono = None
        self._exit_candidate_stop_wall = None
        self._exit_reason = None

    def _reset_to_idle(self) -> None:
        self.state = TimerState.IDLE
        self._clear_entry()
        self._clear_exit()
        self._interval_start_mono = None
        self._interval_start_wall = None

    @staticmethod
    def _resolve_time(
        now_mono: float | None, wall_time: datetime | None
    ) -> tuple[float, datetime]:
        return (
            time.monotonic() if now_mono is None else now_mono,
            datetime.now().astimezone() if wall_time is None else wall_time,
        )
