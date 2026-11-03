from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np

from .config import AppConfig
from .csv_logger import IntervalCsvLogger
from .geometry import Point, draw_polygon_overlay, load_polygon, save_polygon
from .hand_tracking import HandObservation, HandTracker
from .timer import Observation, OperationTimer, TimerSnapshot
from .visualization import draw_status

WINDOW_NAME = "AI Time Study - Hand Motion Timing"


@dataclass
class PolygonEditor:
    editing: bool = False
    points: list[Point] = field(default_factory=list)

    def start(self, existing: list[Point]) -> None:
        self.editing = True
        self.points = list(existing)

    def cancel(self) -> None:
        self.editing = False
        self.points = []

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if not self.editing:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and self.points:
            self.points.pop()

    def draw(self, frame: np.ndarray) -> None:
        if not self.editing:
            return
        for point in self.points:
            cv2.circle(frame, point, 5, (0, 230, 255), thickness=-1)
        if len(self.points) >= 2:
            cv2.polylines(
                frame,
                [np.asarray(self.points, dtype=np.int32)],
                isClosed=False,
                color=(0, 230, 255),
                thickness=2,
            )
        if len(self.points) >= 3:
            cv2.polylines(
                frame,
                [np.asarray(self.points, dtype=np.int32)],
                isClosed=True,
                color=(0, 180, 255),
                thickness=1,
            )


def run_video_app(config: AppConfig) -> int:
    polygon = load_polygon(config.polygon_path)
    timer = OperationTimer(config.timing_config())
    logger = IntervalCsvLogger(config.output_csv)
    tracker = HandTracker(
        max_hands=config.max_hands,
        hand_model_path=config.hand_model_path,
        min_detection_confidence=config.min_detection_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
        smoothing_alpha=config.smoothing_alpha,
        boundary_margin_px=config.boundary_margin_px,
        min_inside_rule_points=config.min_inside_rule_points,
    )
    editor = PolygonEditor()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, editor.on_mouse)

    cap = _open_capture(config)
    fps_meter = FpsMeter()
    last_snapshot = timer.snapshot(Observation.UNKNOWN, message="idle")

    try:
        while True:
            if cap is None or not cap.isOpened():
                last_snapshot = _step_unknown(timer, logger)
                cap = _reconnect(config)
                continue

            ok, frame = cap.read()
            if not ok:
                last_snapshot = _step_unknown(timer, logger)
                cap.release()
                cap = _reconnect(config)
                continue

            now_mono = time.monotonic()
            wall_time = datetime.now().astimezone()
            fps = fps_meter.update(now_mono)

            observations: list[HandObservation] = []
            observation = Observation.UNKNOWN
            if len(polygon) >= 3 and not editor.editing:
                observations, observation = tracker.process(frame, polygon)

            last_snapshot = timer.step(observation, now_mono, wall_time)
            if last_snapshot.completed_interval:
                logger.append(last_snapshot.completed_interval)

            display = frame.copy()
            if len(polygon) >= 3:
                draw_polygon_overlay(display, polygon)
            tracker.draw(display, observations)
            editor.draw(display)
            draw_status(
                display,
                snapshot=last_snapshot,
                observations=observations,
                fps=fps,
                camera_ok=True,
                polygon_ready=len(polygon) >= 3,
                output_csv=config.output_csv,
            )
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if editor.editing:
                saved = _handle_editor_key(key, editor)
                if saved:
                    polygon = list(editor.points)
                    height, width = frame.shape[:2]
                    save_polygon(
                        config.polygon_path,
                        polygon,
                        frame_size=(width, height),
                        source=config.source,
                    )
                    editor.cancel()
                continue

            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("p"), ord("P")):
                editor.start(polygon)

    finally:
        completed = timer.force_stop("manual_shutdown")
        if completed:
            logger.append(completed)
        if cap is not None:
            cap.release()
        tracker.close()
        cv2.destroyAllWindows()

    return 0


@dataclass
class FpsMeter:
    value: float = 0.0
    previous_time: float | None = None

    def update(self, now_mono: float) -> float:
        if self.previous_time is None:
            self.previous_time = now_mono
            return self.value
        delta = max(1e-6, now_mono - self.previous_time)
        instant = 1.0 / delta
        self.value = instant if self.value == 0.0 else 0.9 * self.value + 0.1 * instant
        self.previous_time = now_mono
        return self.value


def _open_capture(config: AppConfig) -> cv2.VideoCapture:
    source = _parse_source(config.source)
    cap = cv2.VideoCapture(source)
    if config.camera_width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
    if config.camera_height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)
    return cap


def _parse_source(source: str) -> int | str:
    return int(source) if source.isdigit() else source


def _reconnect(config: AppConfig) -> cv2.VideoCapture:
    time.sleep(max(0.05, config.reconnect_delay_seconds))
    return _open_capture(config)


def _step_unknown(timer: OperationTimer, logger: IntervalCsvLogger) -> TimerSnapshot:
    snapshot = timer.step(Observation.UNKNOWN)
    if snapshot.completed_interval:
        logger.append(snapshot.completed_interval)
    return snapshot


def _handle_editor_key(key: int, editor: PolygonEditor) -> bool:
    if key in (13, 10) and len(editor.points) >= 3:
        return True
    if key in (27,):
        editor.cancel()
    elif key in (8, 127) and editor.points:
        editor.points.pop()
    elif key in (ord("c"), ord("C")):
        editor.points.clear()
    return False

