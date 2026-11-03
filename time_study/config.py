from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from .timer import TimingConfig


@dataclass
class AppConfig:
    source: str = "0"
    polygon_path: str = "machine_area.json"
    output_csv: str = "time_study_log.csv"
    max_hands: int = 2
    hand_model_path: str = "models/hand_landmarker.task"
    min_detection_confidence: float = 0.60
    min_tracking_confidence: float = 0.50
    smoothing_alpha: float = 0.35
    boundary_margin_px: float = 8.0
    min_inside_rule_points: int = 2
    entry_confirm_seconds: float = 0.25
    exit_confirm_seconds: float = 0.25
    lost_grace_seconds: float = 0.50
    reconnect_delay_seconds: float = 1.0
    camera_width: int = 0
    camera_height: int = 0

    def timing_config(self) -> TimingConfig:
        return TimingConfig(
            entry_confirm_seconds=self.entry_confirm_seconds,
            exit_confirm_seconds=self.exit_confirm_seconds,
            lost_grace_seconds=self.lost_grace_seconds,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hand-motion machine-area time study prototype."
    )
    parser.add_argument("--config", help="Optional JSON config file.")
    parser.add_argument("--source", help="Camera index, video file, or RTSP URL.")
    parser.add_argument("--polygon-path", help="Path to saved machine polygon JSON.")
    parser.add_argument("--output-csv", help="Path to interval CSV output.")
    parser.add_argument("--max-hands", type=int, help="Maximum hands to detect.")
    parser.add_argument(
        "--hand-model-path",
        help="MediaPipe Tasks hand landmarker .task model path.",
    )
    parser.add_argument(
        "--min-detection-confidence", type=float, help="MediaPipe detection threshold."
    )
    parser.add_argument(
        "--min-tracking-confidence", type=float, help="MediaPipe tracking threshold."
    )
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        help="EMA smoothing alpha for palm position, 0.0 to 1.0.",
    )
    parser.add_argument(
        "--boundary-margin-px",
        type=float,
        help="Hysteresis margin around polygon boundary in pixels.",
    )
    parser.add_argument(
        "--min-inside-rule-points",
        type=int,
        help="Minimum wrist/palm/MCP rule points inside polygon.",
    )
    parser.add_argument(
        "--entry-confirm-seconds", type=float, help="Inside duration before start."
    )
    parser.add_argument(
        "--exit-confirm-seconds", type=float, help="Outside duration before stop."
    )
    parser.add_argument(
        "--lost-grace-seconds", type=float, help="No-detection grace before stop."
    )
    parser.add_argument(
        "--reconnect-delay-seconds",
        type=float,
        help="Delay between camera reconnect attempts.",
    )
    parser.add_argument("--camera-width", type=int, help="Requested capture width.")
    parser.add_argument("--camera-height", type=int, help="Requested capture height.")
    return parser


def load_config(args: argparse.Namespace) -> AppConfig:
    config = AppConfig()
    if args.config:
        config = _load_json_config(Path(args.config), config)

    for field in fields(AppConfig):
        cli_name = field.name.replace("_", "-")
        value = getattr(args, field.name, None)
        if value is None:
            value = getattr(args, cli_name, None)
        if value is not None:
            setattr(config, field.name, value)

    return config


def _load_json_config(path: Path, default: AppConfig) -> AppConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(AppConfig)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        names = ", ".join(unknown)
        raise ValueError(f"Unknown config keys in {path}: {names}")

    values: dict[str, Any] = {field.name: getattr(default, field.name) for field in fields(default)}
    values.update(data)
    return AppConfig(**values)
