from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

Point = tuple[int, int]


def load_polygon(path: str | Path) -> list[Point]:
    polygon_path = Path(path)
    if not polygon_path.exists():
        return []

    data = json.loads(polygon_path.read_text(encoding="utf-8"))
    points = data.get("polygon", data)
    return [(int(x), int(y)) for x, y in points]


def save_polygon(
    path: str | Path,
    polygon: Iterable[Point],
    frame_size: tuple[int, int],
    source: str,
) -> None:
    polygon_path = Path(path)
    polygon_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = frame_size
    data = {
        "source": source,
        "frame_width": width,
        "frame_height": height,
        "polygon": [[int(x), int(y)] for x, y in polygon],
    }
    polygon_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def point_in_polygon(point: Point, polygon: list[Point], margin_px: float = 0.0) -> bool:
    if len(polygon) < 3:
        return False

    contour = np.asarray(polygon, dtype=np.float32)
    distance = cv2.pointPolygonTest(
        contour, (float(point[0]), float(point[1])), measureDist=True
    )
    return distance >= -float(margin_px)


def draw_polygon_overlay(
    frame: np.ndarray,
    polygon: list[Point],
    fill_color: tuple[int, int, int] = (30, 180, 90),
    outline_color: tuple[int, int, int] = (40, 230, 120),
    alpha: float = 0.22,
) -> None:
    if len(polygon) < 3:
        return

    contour = np.asarray(polygon, dtype=np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [contour], fill_color)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, dst=frame)
    cv2.polylines(frame, [contour], isClosed=True, color=outline_color, thickness=2)
