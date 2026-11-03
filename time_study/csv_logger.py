from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .timer import CompletedInterval


COLUMNS = [
    "date",
    "start_timestamp",
    "end_timestamp",
    "active_duration_seconds",
    "total_accumulated_seconds",
    "interval_number",
    "detection_status",
]


class IntervalCsvLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists() or self.path.stat().st_size == 0:
            pd.DataFrame(columns=COLUMNS).to_csv(self.path, index=False)

    def append(self, interval: CompletedInterval) -> None:
        row = {key: asdict(interval)[key] for key in COLUMNS}
        pd.DataFrame([row], columns=COLUMNS).to_csv(
            self.path, mode="a", header=False, index=False
        )
