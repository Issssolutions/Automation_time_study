from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .hand_tracking import HandObservation
from .timer import TimerSnapshot


def draw_status(
    frame: np.ndarray,
    snapshot: TimerSnapshot,
    observations: list[HandObservation],
    fps: float,
    camera_ok: bool,
    polygon_ready: bool,
    output_csv: str,
) -> None:
    inside_count = sum(hand.inside for hand in observations)
    status_color = {
        "IDLE": (210, 210, 210),
        "ENTER_PENDING": (0, 210, 255),
        "ACTIVE": (40, 230, 120),
        "EXIT_PENDING": (0, 170, 255),
    }[snapshot.state.value]

    lines = [
        f"State: {snapshot.state.value}",
        f"Current active: {snapshot.active_duration_seconds:.2f}s",
        f"Total active: {snapshot.total_accumulated_seconds:.2f}s",
        f"Intervals: {snapshot.interval_count}",
        f"Observation: {snapshot.observation.value}",
        f"Hands: {len(observations)}  Inside: {inside_count}",
        f"FPS: {fps:.1f}  Camera: {'OK' if camera_ok else 'FILE'}",
        f"Machine area: {'READY' if polygon_ready else 'NOT SET'}",
        f"Log: {Path(output_csv).name}",
    ]
    if snapshot.message:
        lines.append(snapshot.message)

    x, y = 14, 18
    line_height = 22
    panel_width = 390
    panel_height = line_height * len(lines) + 16
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_width, 8 + panel_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, dst=frame)

    for index, line in enumerate(lines):
        color = status_color if index == 0 else (235, 235, 235)
        cv2.putText(
            frame,
            line,
            (x, y + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
