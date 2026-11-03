from __future__ import annotations

import argparse
import json
from pathlib import Path

from time_study.config import AppConfig
from time_study.geometry import Point
from time_study.video_file_analyzer import analyze_video_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a recorded video file.")
    parser.add_argument("video", help="Path to the recorded video.")
    parser.add_argument(
        "--polygon",
        required=True,
        help="JSON file containing [[x, y], ...] or {'polygon': [[x, y], ...]}.",
    )
    parser.add_argument("--output-dir", default="analysis_output")
    parser.add_argument(
        "--hand-model-path",
        default="models/hand_landmarker.task",
        help="MediaPipe Tasks hand_landmarker.task path.",
    )
    parser.add_argument("--no-annotated-video", action="store_true")
    args = parser.parse_args()

    polygon = _load_polygon(args.polygon)
    result = analyze_video_file(
        args.video,
        polygon,
        config=AppConfig(hand_model_path=args.hand_model_path),
        output_dir=args.output_dir,
        annotate=not args.no_annotated_video,
    )
    print(f"Intervals: {result.interval_count}")
    print(f"Total active time: {result.total_active_seconds:.3f}s")
    print(f"CSV: {result.output_csv}")
    if result.annotated_video:
        print(f"Annotated video: {result.annotated_video}")
    return 0


def _load_polygon(path: str | Path) -> list[Point]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_points = data.get("polygon", data) if isinstance(data, dict) else data
    return [(int(x), int(y)) for x, y in raw_points]


if __name__ == "__main__":
    raise SystemExit(main())
