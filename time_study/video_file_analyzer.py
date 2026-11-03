from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import cv2
import pandas as pd

from .config import AppConfig
from .csv_logger import COLUMNS
from .geometry import Point, draw_polygon_overlay
from .hand_tracking import HandObservation, HandTracker
from .timer import CompletedInterval, Observation, OperationTimer, TimerSnapshot
from .visualization import draw_status


ProgressCallback = Callable[[int, int, TimerSnapshot], None]


@dataclass
class VideoAnalysisResult:
    intervals: list[CompletedInterval]
    total_active_seconds: float
    interval_count: int
    processed_frames: int
    source_fps: float
    duration_seconds: float
    output_csv: Path
    annotated_video: Path | None


def analyze_video_file(
    video_path: str | Path,
    polygon: list[Point],
    config: AppConfig,
    output_dir: str | Path,
    annotate: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> VideoAnalysisResult:
    if len(polygon) < 3:
        raise ValueError("Machine-area polygon must contain at least 3 points.")

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    timer = OperationTimer(config.timing_config())
    tracker = HandTracker(
        max_hands=config.max_hands,
        hand_model_path=config.hand_model_path,
        min_detection_confidence=config.min_detection_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
        smoothing_alpha=config.smoothing_alpha,
        boundary_margin_px=config.boundary_margin_px,
        min_inside_rule_points=config.min_inside_rule_points,
    )
    intervals: list[CompletedInterval] = []

    output_csv = output_dir / "analysis_intervals.csv"
    annotated_video: Path | None = None
    writer: cv2.VideoWriter | None = None
    if annotate and width > 0 and height > 0:
        annotated_video = output_dir / "annotated_analysis.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(annotated_video), fourcc, fps, (width, height))
        if not writer.isOpened():
            writer.release()
            writer = None
            annotated_video = None

    base_wall_time = datetime.now().astimezone()
    processed_frames = 0
    last_video_time = 0.0
    last_snapshot = timer.snapshot(Observation.UNKNOWN, now_mono=0.0, message="idle")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            video_time = _frame_time_seconds(cap, processed_frames, fps)
            last_video_time = video_time
            wall_time = base_wall_time + timedelta(seconds=video_time)

            observations: list[HandObservation]
            observation: Observation
            observations, observation = tracker.process(frame, polygon)

            last_snapshot = timer.step(observation, now_mono=video_time, wall_time=wall_time)
            if last_snapshot.completed_interval:
                intervals.append(last_snapshot.completed_interval)

            if writer is not None:
                display = frame.copy()
                draw_polygon_overlay(display, polygon)
                tracker.draw(display, observations)
                draw_status(
                    display,
                    snapshot=last_snapshot,
                    observations=observations,
                    fps=fps,
                    camera_ok=False,
                    polygon_ready=True,
                    output_csv=str(output_csv),
                )
                writer.write(display)

            processed_frames += 1
            if progress_callback is not None:
                progress_callback(processed_frames, frame_count, last_snapshot)

        completed = timer.force_stop(
            "video_end",
            now_mono=last_video_time,
            wall_time=base_wall_time + timedelta(seconds=last_video_time),
        )
        if completed:
            intervals.append(completed)
    finally:
        cap.release()
        tracker.close()
        if writer is not None:
            writer.release()

    _write_intervals(output_csv, intervals)

    duration_seconds = last_video_time if processed_frames else 0.0
    return VideoAnalysisResult(
        intervals=intervals,
        total_active_seconds=timer.total_active_seconds,
        interval_count=timer.interval_count,
        processed_frames=processed_frames,
        source_fps=fps,
        duration_seconds=duration_seconds,
        output_csv=output_csv,
        annotated_video=annotated_video,
    )


def extract_first_frame(video_path: str | Path, output_path: str | Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not read first frame from: {video_path}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)
    height, width = frame.shape[:2]
    return width, height


def _frame_time_seconds(cap: cv2.VideoCapture, frame_index: int, fps: float) -> float:
    position_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
    if position_ms > 0:
        return position_ms / 1000.0
    return frame_index / fps


def _write_intervals(path: Path, intervals: list[CompletedInterval]) -> None:
    rows = [{key: asdict(interval)[key] for key in COLUMNS} for interval in intervals]
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)
