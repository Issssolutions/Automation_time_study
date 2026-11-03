# AI Hand-Motion Time Study Prototype

Python prototype for phase 1 of a garment-industry time study workflow: detect hands, monitor a configured machine working area, and log active operation time. This version intentionally does not classify 3-element operations or calculate standard time.

## What It Does

- Uses prerecored video for detection.
- Uses MediaPipe Hands for left/right hand landmarks.
- Lets you draw and reuse a polygon around the machine working area.
- Starts timing only after a hand is confirmed inside the polygon.
- Counts both hands as one operation interval.
- Pauses/stops timing when no qualifying hand remains inside.
- Handles brief detection loss with an explicit `EXIT_PENDING` state.
- Logs each completed active interval to CSV.
- Keeps the timing engine independent from video code for testing.
- In future we can use django framework for better structure.

## Install

Recommended Python versions:

- Python 3.11 or 3.12 with MediaPipe 0.10.x can use the classic MediaPipe `solutions` API.
- Python 3.14 installs MediaPipe 1.x in this environment, which uses the newer Tasks API and requires a model file.

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you are using Python 3.14 or you see `module 'mediapipe' has no attribute 'solutions'`, download the Tasks model:

```powershell
python download_models.py
```

This saves:

```text
models/hand_landmarker.task
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run: Uploaded Video Analysis

This is the recommended mode when you want to upload a recorded production video instead of using a live camera.

```powershell
python upload_app.py
```

Open:

```text
http://127.0.0.1:5000/
```

Workflow:

1. Upload a video file.
2. Draw the machine working-area polygon on the first frame.
3. Adjust timing thresholds if needed.
4. Analyze the video.
5. Wait on the progress page while the background analyzer processes frames.
6. Download the interval CSV and annotated video.

You can also analyze a local video from the command line if you already have polygon coordinates:

```powershell
python analyze_video.py "path\to\video.mp4" --polygon machine_area.json --output-dir analysis_output
```

## Run: Live Camera

Default webcam:

```powershell
python main.py --source 0
```

RTSP/IP camera:

```powershell
python main.py --source "rtsp://user:password@camera-ip/stream1"
```

With the sample config:

```powershell
python main.py --config time_study_config.example.json
```

## Live Controls

- `P`: start or edit the machine-area polygon.
- Left click: add a polygon point.
- Right click or Backspace: remove the last point.
- Enter: save the polygon when it has at least 3 points.
- `C`: clear points while editing.
- Esc: cancel polygon editing.
- `Q`: quit.

The polygon is saved to `machine_area.json` by default and reused on the next run.

## Timing Logic

The timing engine uses monotonic timestamps, not frame counts:

- `IDLE`: no confirmed active operation.
- `ENTER_PENDING`: a hand is inside but must remain inside for the entry confirmation duration.
- `ACTIVE`: at least one qualifying hand is confirmed inside the machine polygon.
- `EXIT_PENDING`: no hand is confidently inside; the app waits through the configured exit/loss duration before closing the interval.

If an exit or detection loss is confirmed, the interval ends at the moment the hand first became outside/uncertain, so the grace period is not counted as productive time.

## Test

The timer can be tested without camera dependencies:

```powershell
python -m unittest discover -s tests
```

## Output

Completed intervals are appended to `time_study_log.csv` by default. See `sample_output.csv` for the column format.

For uploaded-video analysis, each job is saved under `analysis_jobs/<job-id>/` with:

- `source.<ext>`: uploaded video.
- `first_frame.jpg`: frame used for polygon setup.
- `analysis_intervals.csv`: completed interval log.
- `annotated_analysis.mp4`: optional annotated output video.

## Main Files

- `main.py`: CLI entry point.
- `upload_app.py`: browser upload, polygon setup, analysis, CSV/video downloads.
- `analyze_video.py`: command-line recorded-video analyzer.
- `download_models.py`: downloads the MediaPipe Tasks hand model needed by MediaPipe 1.x.
- `time_study/video_file_analyzer.py`: offline video processing pipeline.
- `time_study/video_app.py`: OpenCV loop, polygon editing, visualization, camera reconnect behavior.
- `time_study/hand_tracking.py`: MediaPipe hand detection, smoothing, inside/outside rule.
- `time_study/timer.py`: finite-state timing engine.
- `time_study/csv_logger.py`: interval CSV writer.
- `tests/test_timer.py`: timing behavior tests.
