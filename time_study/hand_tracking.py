from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from .geometry import Point, point_in_polygon
from .timer import Observation


PALM_LANDMARKS = (0, 5, 9, 13, 17)
RULE_LANDMARKS = (0, 5, 9, 13, 17)


@dataclass
class HandObservation:
    label: str
    confidence: float
    palm_center: Point
    wrist: Point
    inside: bool
    inside_points: int
    rule_points: int
    landmarks_px: list[Point]
    landmark_list: Any


class PointSmoother:
    def __init__(self, alpha: float) -> None:
        self.alpha = min(1.0, max(0.0, alpha))
        self._points: dict[str, np.ndarray] = {}

    def update(self, key: str, point: Point) -> Point:
        current = np.asarray(point, dtype=np.float32)
        previous = self._points.get(key)
        if previous is None or self.alpha >= 1.0:
            smoothed = current
        else:
            smoothed = self.alpha * current + (1.0 - self.alpha) * previous
        self._points[key] = smoothed
        return int(round(smoothed[0])), int(round(smoothed[1]))

    def prune(self, active_keys: set[str]) -> None:
        for key in list(self._points):
            if key not in active_keys:
                del self._points[key]


class HandTracker:
    def __init__(
        self,
        max_hands: int = 2,
        hand_model_path: str = "models/hand_landmarker.task",
        min_detection_confidence: float = 0.60,
        min_tracking_confidence: float = 0.50,
        smoothing_alpha: float = 0.35,
        boundary_margin_px: float = 8.0,
        min_inside_rule_points: int = 2,
    ) -> None:
        self.boundary_margin_px = boundary_margin_px
        self.min_inside_rule_points = min_inside_rule_points
        self.min_handedness_confidence = min_detection_confidence
        self._smoother = PointSmoother(smoothing_alpha)
        self._backend = _build_backend(
            max_hands=max_hands,
            hand_model_path=hand_model_path,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(
        self, frame_bgr: np.ndarray, polygon: list[Point]
    ) -> tuple[list[HandObservation], Observation]:
        detected_hands = self._backend.detect(frame_bgr)
        if not detected_hands:
            self._smoother.prune(set())
            return [], Observation.UNKNOWN

        height, width = frame_bgr.shape[:2]
        observations: list[HandObservation] = []
        active_keys: set[str] = set()

        for index, detected in enumerate(detected_hands):
            label = detected.label
            confidence = detected.confidence
            if confidence < self.min_handedness_confidence:
                continue

            smoother_key = f"{label}-{index}"
            active_keys.add(smoother_key)

            landmarks_px = [
                (
                    int(round(landmark.x * width)),
                    int(round(landmark.y * height)),
                )
                for landmark in detected.landmarks
            ]
            palm_center_raw = _mean_point([landmarks_px[i] for i in PALM_LANDMARKS])
            palm_center = self._smoother.update(smoother_key, palm_center_raw)
            wrist = landmarks_px[0]

            rule_points = [landmarks_px[i] for i in RULE_LANDMARKS] + [palm_center]
            inside_points = sum(
                point_in_polygon(point, polygon, self.boundary_margin_px)
                for point in rule_points
            )
            palm_inside = point_in_polygon(palm_center, polygon, self.boundary_margin_px)
            inside = palm_inside or inside_points >= self.min_inside_rule_points

            observations.append(
                HandObservation(
                    label=label,
                    confidence=confidence,
                    palm_center=palm_center,
                    wrist=wrist,
                    inside=inside,
                    inside_points=inside_points,
                    rule_points=len(rule_points),
                    landmarks_px=landmarks_px,
                    landmark_list=detected.landmark_list,
                )
            )

        self._smoother.prune(active_keys)
        if not observations:
            return [], Observation.UNKNOWN

        return observations, Observation.INSIDE if any(hand.inside for hand in observations) else Observation.OUTSIDE

    def draw(self, frame: np.ndarray, observations: list[HandObservation]) -> None:
        self._backend.draw(frame, observations)

    def close(self) -> None:
        self._backend.close()


@dataclass
class DetectedHand:
    label: str
    confidence: float
    landmarks: list[Any]
    landmark_list: Any


class _SolutionsHandsBackend:
    def __init__(
        self,
        max_hands: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._drawing = mp.solutions.drawing_utils
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame_bgr: np.ndarray) -> list[DetectedHand]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self._hands.process(frame_rgb)
        if not results.multi_hand_landmarks:
            return []

        detected_hands: list[DetectedHand] = []
        handedness = results.multi_handedness or []
        for index, landmark_list in enumerate(results.multi_hand_landmarks):
            classification = handedness[index].classification[0] if index < len(handedness) else None
            detected_hands.append(
                DetectedHand(
                    label=classification.label if classification else f"Hand{index + 1}",
                    confidence=float(classification.score) if classification else 1.0,
                    landmarks=list(landmark_list.landmark),
                    landmark_list=landmark_list,
                )
            )
        return detected_hands

    def draw(self, frame: np.ndarray, observations: list[HandObservation]) -> None:
        for hand in observations:
            color = (40, 220, 90) if hand.inside else (50, 80, 240)
            self._drawing.draw_landmarks(
                frame,
                hand.landmark_list,
                self._mp_hands.HAND_CONNECTIONS,
                self._drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                self._drawing.DrawingSpec(color=color, thickness=2),
            )
            cv2.circle(frame, hand.palm_center, 7, color, thickness=-1)
            cv2.putText(
                frame,
                f"{hand.label} {hand.confidence:.2f} {hand.inside_points}/{hand.rule_points}",
                (hand.palm_center[0] + 8, hand.palm_center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )

    def close(self) -> None:
        self._hands.close()


class _TasksHandLandmarkerBackend:
    def __init__(
        self,
        max_hands: int,
        hand_model_path: str,
        min_detection_confidence: float,
        min_tracking_confidence: float,
    ) -> None:
        model_path = Path(hand_model_path)
        if not model_path.exists():
            raise RuntimeError(
                "This MediaPipe install uses the newer Tasks API and needs a "
                f"hand landmarker model file at {model_path}. Run "
                "`python download_models.py` or pass --hand-model-path with a "
                "valid hand_landmarker.task file."
            )

        self._vision = mp.tasks.vision
        base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options = self._vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=self._vision.RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = self._vision.HandLandmarker.create_from_options(options)
        self._connections = [
            (connection.start, connection.end)
            for connection in self._vision.HandLandmarksConnections.HAND_CONNECTIONS
        ]

    def detect(self, frame_bgr: np.ndarray) -> list[DetectedHand]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect(image)
        detected_hands: list[DetectedHand] = []
        for index, landmarks in enumerate(result.hand_landmarks):
            categories = result.handedness[index] if index < len(result.handedness) else []
            category = categories[0] if categories else None
            detected_hands.append(
                DetectedHand(
                    label=category.category_name if category else f"Hand{index + 1}",
                    confidence=float(category.score) if category else 1.0,
                    landmarks=list(landmarks),
                    landmark_list=list(landmarks),
                )
            )
        return detected_hands

    def draw(self, frame: np.ndarray, observations: list[HandObservation]) -> None:
        for hand in observations:
            color = (40, 220, 90) if hand.inside else (50, 80, 240)
            for start, end in self._connections:
                cv2.line(frame, hand.landmarks_px[start], hand.landmarks_px[end], color, 2)
            for point in hand.landmarks_px:
                cv2.circle(frame, point, 2, color, thickness=-1)
            cv2.circle(frame, hand.palm_center, 7, color, thickness=-1)
            cv2.putText(
                frame,
                f"{hand.label} {hand.confidence:.2f} {hand.inside_points}/{hand.rule_points}",
                (hand.palm_center[0] + 8, hand.palm_center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )

    def close(self) -> None:
        self._landmarker.close()


def _build_backend(
    max_hands: int,
    hand_model_path: str,
    min_detection_confidence: float,
    min_tracking_confidence: float,
) -> _SolutionsHandsBackend | _TasksHandLandmarkerBackend:
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        return _SolutionsHandsBackend(
            max_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    if hasattr(mp, "tasks") and hasattr(mp.tasks, "vision"):
        return _TasksHandLandmarkerBackend(
            max_hands=max_hands,
            hand_model_path=hand_model_path,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    raise RuntimeError(
        "Unsupported MediaPipe installation. Install MediaPipe 0.10.x on "
        "Python 3.11/3.12, or use MediaPipe Tasks with hand_landmarker.task."
    )


def _mean_point(points: list[Point]) -> Point:
    array = np.asarray(points, dtype=np.float32)
    mean = array.mean(axis=0)
    return int(round(mean[0])), int(round(mean[1]))
