from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve


HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
OUTPUT_PATH = Path("models") / "hand_landmarker.task"


def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {HAND_LANDMARKER_URL}")
    urlretrieve(HAND_LANDMARKER_URL, OUTPUT_PATH)
    print(f"Saved {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
