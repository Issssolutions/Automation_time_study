from __future__ import annotations

import csv
import json
import threading
import traceback
import uuid
from pathlib import Path

from flask import Flask, abort, jsonify, redirect, render_template_string, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from time_study.config import AppConfig
from time_study.geometry import Point
from time_study.video_file_analyzer import analyze_video_file, extract_first_frame


BASE_DIR = Path(__file__).resolve().parent
JOB_DIR = BASE_DIR / "analysis_jobs"
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024


@app.get("/")
def index():
    return render_template_string(INDEX_TEMPLATE)

@app.post("/upload")
def upload_video():
    uploaded = request.files.get("video")
    if not uploaded or uploaded.filename == "":
        abort(400, "No video file uploaded.")

    filename = secure_filename(uploaded.filename)
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        abort(400, "Unsupported video format.")

    job_id = uuid.uuid4().hex
    job_path = JOB_DIR / job_id
    job_path.mkdir(parents=True, exist_ok=True)

    video_path = job_path / f"source{extension}"
    uploaded.save(video_path)
    width, height = extract_first_frame(video_path, job_path / "first_frame.jpg")

    metadata = {
        "job_id": job_id,
        "original_filename": filename,
        "video_path": str(video_path),
        "frame_width": width,
        "frame_height": height,
    }
    (job_path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return redirect(url_for("configure", job_id=job_id))


@app.get("/configure/<job_id>")
def configure(job_id: str):
    job_path = _job_path(job_id)
    metadata = _metadata(job_path)
    return render_template_string(CONFIGURE_TEMPLATE, metadata=metadata)


@app.post("/analyze/<job_id>")
def analyze(job_id: str):
    job_path = _job_path(job_id)
    metadata = _metadata(job_path)

    polygon_json = request.form.get("polygon", "[]")
    polygon = _parse_polygon(polygon_json)

    config = AppConfig(
        hand_model_path=request.form.get("hand_model_path", "models/hand_landmarker.task"),
        entry_confirm_seconds=float(request.form.get("entry_confirm_seconds", 0.25)),
        exit_confirm_seconds=float(request.form.get("exit_confirm_seconds", 0.25)),
        lost_grace_seconds=float(request.form.get("lost_grace_seconds", 0.50)),
        min_detection_confidence=float(request.form.get("min_detection_confidence", 0.60)),
        min_tracking_confidence=float(request.form.get("min_tracking_confidence", 0.50)),
        smoothing_alpha=float(request.form.get("smoothing_alpha", 0.35)),
        boundary_margin_px=float(request.form.get("boundary_margin_px", 8.0)),
        output_csv=str(job_path / "analysis_intervals.csv"),
    )

    request_data = {
        "polygon": polygon,
        "config": config.__dict__,
        "annotate": request.form.get("annotate") == "on",
    }
    _write_json(job_path / "analysis_request.json", request_data)
    _write_status(
        job_path,
        {
            "state": "queued",
            "message": "Analysis queued",
            "processed_frames": 0,
            "total_frames": 0,
            "percent": 0,
            "total_active_seconds": 0.0,
            "interval_count": 0,
        },
    )

    worker = threading.Thread(
        target=_run_analysis_job,
        args=(job_path, metadata, request_data),
        daemon=True,
    )
    worker.start()
    return redirect(url_for("progress", job_id=job_id))


@app.get("/progress/<job_id>")
def progress(job_id: str):
    job_path = _job_path(job_id)
    metadata = _metadata(job_path)
    return render_template_string(PROGRESS_TEMPLATE, metadata=metadata)


@app.get("/status/<job_id>")
def status(job_id: str):
    job_path = _job_path(job_id)
    return jsonify(_read_status(job_path))


@app.get("/results/<job_id>")
def results(job_id: str):
    job_path = _job_path(job_id)
    metadata = _metadata(job_path)
    status_data = _read_status(job_path)
    if status_data.get("state") != "complete":
        return redirect(url_for("progress", job_id=job_id))

    rows = _read_interval_rows(job_path / "analysis_intervals.csv")
    annotated_name = status_data.get("annotated_video")
    return render_template_string(
        RESULT_TEMPLATE,
        metadata=metadata,
        result=status_data,
        rows=rows,
        annotated_name=annotated_name,
    )


@app.get("/jobs/<job_id>/<path:filename>")
def job_file(job_id: str, filename: str):
    job_path = _job_path(job_id)
    return send_from_directory(job_path, filename, as_attachment=False)


def _job_path(job_id: str) -> Path:
    if not job_id or any(char not in "0123456789abcdef" for char in job_id):
        abort(404)
    job_path = JOB_DIR / job_id
    if not job_path.exists():
        abort(404)
    return job_path


def _metadata(job_path: Path) -> dict:
    return json.loads((job_path / "metadata.json").read_text(encoding="utf-8"))


def _parse_polygon(polygon_json: str) -> list[Point]:
    raw_points = json.loads(polygon_json)
    polygon = [(int(round(point["x"])), int(round(point["y"]))) for point in raw_points]
    if len(polygon) < 3:
        abort(400, "Draw at least 3 machine-area points.")
    return polygon


def _run_analysis_job(job_path: Path, metadata: dict, request_data: dict) -> None:
    try:
        config = AppConfig(**request_data["config"])
        polygon = [tuple(point) for point in request_data["polygon"]]
        last_status_frame = 0

        def update_progress(processed_frames: int, total_frames: int, snapshot) -> None:
            nonlocal last_status_frame
            if processed_frames - last_status_frame < 15 and processed_frames != total_frames:
                return
            last_status_frame = processed_frames
            percent = round((processed_frames / total_frames) * 100, 1) if total_frames else 0
            _write_status(
                job_path,
                {
                    "state": "running",
                    "message": "Analyzing video",
                    "processed_frames": processed_frames,
                    "total_frames": total_frames,
                    "percent": percent,
                    "total_active_seconds": round(snapshot.total_accumulated_seconds, 3),
                    "interval_count": snapshot.interval_count,
                    "current_state": snapshot.state.value,
                },
            )

        _write_status(
            job_path,
            {
                "state": "running",
                "message": "Starting analysis",
                "processed_frames": 0,
                "total_frames": 0,
                "percent": 0,
                "total_active_seconds": 0.0,
                "interval_count": 0,
            },
        )
        result = analyze_video_file(
            metadata["video_path"],
            polygon,
            config=config,
            output_dir=job_path,
            annotate=bool(request_data["annotate"]),
            progress_callback=update_progress,
        )
        _write_status(
            job_path,
            {
                "state": "complete",
                "message": "Analysis complete",
                "processed_frames": result.processed_frames,
                "total_frames": result.processed_frames,
                "percent": 100,
                "total_active_seconds": round(result.total_active_seconds, 3),
                "interval_count": result.interval_count,
                "source_fps": round(result.source_fps, 3),
                "duration_seconds": round(result.duration_seconds, 3),
                "analysis_csv": result.output_csv.name,
                "annotated_video": result.annotated_video.name if result.annotated_video else None,
            },
        )
    except Exception as exc:
        _write_status(
            job_path,
            {
                "state": "error",
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )


def _status_path(job_path: Path) -> Path:
    return job_path / "status.json"


def _read_status(job_path: Path) -> dict:
    path = _status_path(job_path)
    if not path.exists():
        return {"state": "not_started", "message": "Analysis has not started."}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_status(job_path: Path, data: dict) -> None:
    current = _read_status(job_path) if _status_path(job_path).exists() else {}
    current.update(data)
    _write_json(_status_path(job_path), current)


def _write_json(path: Path, data: dict) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _read_interval_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []

    rows: list[dict] = []
    with path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            row["interval_number"] = int(row["interval_number"])
            row["active_duration_seconds"] = float(row["active_duration_seconds"])
            row["total_accumulated_seconds"] = float(row["total_accumulated_seconds"])
            rows.append(row)
    return rows


INDEX_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Video Time Study Analysis</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #1e293b;
      --muted: #64748b;
      --line: #d9e2ec;
      --accent: #0f766e;
      --accent-strong: #115e59;
      --bg: #f7fafc;
      --panel: #ffffff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Segoe UI, Roboto, Arial, sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    header {
      padding: 20px 28px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    h1 { margin: 0; font-size: 22px; font-weight: 650; }
    main { max-width: 980px; margin: 0 auto; padding: 28px; }
    form {
      display: grid;
      gap: 16px;
      max-width: 620px;
      padding: 20px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    label { font-weight: 600; }
    input[type=file] {
      width: 100%;
      padding: 12px;
      border: 1px dashed #94a3b8;
      border-radius: 8px;
      background: #f8fafc;
    }
    button, .button {
      border: 0;
      border-radius: 7px;
      padding: 10px 14px;
      color: #fff;
      background: var(--accent);
      font-weight: 650;
      cursor: pointer;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 40px;
    }
    button:hover, .button:hover { background: var(--accent-strong); }
  </style>
</head>
<body>
  <header><h1>Video Time Study Analysis</h1></header>
  <main>
    <form action="/upload" method="post" enctype="multipart/form-data">
      <label for="video">Upload production video</label>
      <input id="video" name="video" type="file" accept="video/*" required>
      <button type="submit">Upload Video</button>
    </form>
  </main>
</body>
</html>
"""


CONFIGURE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Configure Machine Area</title>
  <style>
    :root {
      --ink: #1e293b;
      --muted: #64748b;
      --line: #d9e2ec;
      --accent: #0f766e;
      --accent-strong: #115e59;
      --warn: #b45309;
      --bg: #f7fafc;
      --panel: #ffffff;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: Segoe UI, Roboto, Arial, sans-serif; background: var(--bg); color: var(--ink); }
    header { padding: 18px 24px; border-bottom: 1px solid var(--line); background: var(--panel); }
    h1 { margin: 0; font-size: 21px; font-weight: 650; }
    main { display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 20px; padding: 20px; }
    .stage { min-width: 0; }
    .frame-wrap { position: relative; display: inline-block; max-width: 100%; background: #111827; border-radius: 8px; overflow: hidden; }
    #frame { display: block; max-width: 100%; height: auto; }
    #canvas { position: absolute; left: 0; top: 0; cursor: crosshair; }
    aside { display: grid; gap: 14px; align-content: start; }
    section { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }
    h2 { margin: 0 0 10px; font-size: 15px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    label { display: grid; gap: 5px; color: var(--muted); font-size: 13px; }
    input[type=number] { width: 100%; padding: 8px; border: 1px solid var(--line); border-radius: 7px; }
    button, .button {
      border: 0; border-radius: 7px; padding: 10px 12px; color: #fff; background: var(--accent);
      font-weight: 650; cursor: pointer; min-height: 38px;
    }
    button.secondary { background: #475569; }
    button.warning { background: var(--warn); }
    .actions { display: flex; flex-wrap: wrap; gap: 8px; }
    .points { color: var(--muted); font-size: 13px; }
    @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header><h1>Configure Machine Area</h1></header>
  <main>
    <div class="stage">
      <div class="frame-wrap" id="frameWrap">
        <img id="frame" src="{{ url_for('job_file', job_id=metadata.job_id, filename='first_frame.jpg') }}" alt="First video frame">
        <canvas id="canvas"></canvas>
      </div>
    </div>
    <aside>
      <section>
        <h2>Machine Polygon</h2>
        <div class="actions">
          <button type="button" class="secondary" id="undo">Undo</button>
          <button type="button" class="warning" id="clear">Clear</button>
        </div>
        <p class="points"><span id="pointCount">0</span> points selected</p>
      </section>
      <section>
        <h2>Analysis Settings</h2>
        <form id="analyzeForm" action="{{ url_for('analyze', job_id=metadata.job_id) }}" method="post">
          <input type="hidden" name="polygon" id="polygonInput">
          <div class="grid">
            <label>Entry seconds<input name="entry_confirm_seconds" type="number" min="0" step="0.05" value="0.25"></label>
            <label>Exit seconds<input name="exit_confirm_seconds" type="number" min="0" step="0.05" value="0.25"></label>
            <label>Lost grace<input name="lost_grace_seconds" type="number" min="0" step="0.05" value="0.50"></label>
            <label>Boundary px<input name="boundary_margin_px" type="number" min="0" step="1" value="8"></label>
            <label>Detection<input name="min_detection_confidence" type="number" min="0" max="1" step="0.05" value="0.60"></label>
            <label>Tracking<input name="min_tracking_confidence" type="number" min="0" max="1" step="0.05" value="0.50"></label>
          </div>
          <label style="margin-top:10px;">Model path<input name="hand_model_path" type="text" value="models/hand_landmarker.task"></label>
          <label style="margin-top:10px;"><input name="annotate" type="checkbox" checked> Annotated video</label>
          <button type="submit" style="width:100%; margin-top:12px;">Analyze Video</button>
        </form>
      </section>
    </aside>
  </main>
  <script>
    const frame = document.getElementById("frame");
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const points = [];
    const nativeWidth = {{ metadata.frame_width }};
    const nativeHeight = {{ metadata.frame_height }};

    function resizeCanvas() {
      const rect = frame.getBoundingClientRect();
      canvas.width = Math.round(rect.width);
      canvas.height = Math.round(rect.height);
      draw();
    }

    function draw() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#14b8a6";
      ctx.fillStyle = "rgba(20, 184, 166, 0.22)";
      const display = points.map(p => ({
        x: p.x * canvas.width / nativeWidth,
        y: p.y * canvas.height / nativeHeight
      }));
      if (display.length > 0) {
        ctx.beginPath();
        ctx.moveTo(display[0].x, display[0].y);
        for (const p of display.slice(1)) ctx.lineTo(p.x, p.y);
        if (display.length >= 3) {
          ctx.closePath();
          ctx.fill();
        }
        ctx.stroke();
      }
      for (const p of display) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
        ctx.fillStyle = "#f59e0b";
        ctx.fill();
        ctx.fillStyle = "rgba(20, 184, 166, 0.22)";
      }
      document.getElementById("pointCount").textContent = points.length;
      document.getElementById("polygonInput").value = JSON.stringify(points);
    }

    canvas.addEventListener("click", event => {
      const rect = canvas.getBoundingClientRect();
      points.push({
        x: (event.clientX - rect.left) * nativeWidth / rect.width,
        y: (event.clientY - rect.top) * nativeHeight / rect.height
      });
      draw();
    });
    document.getElementById("undo").addEventListener("click", () => { points.pop(); draw(); });
    document.getElementById("clear").addEventListener("click", () => { points.length = 0; draw(); });
    document.getElementById("analyzeForm").addEventListener("submit", event => {
      if (points.length < 3) {
        event.preventDefault();
        alert("Select at least 3 polygon points.");
      }
    });
    frame.addEventListener("load", resizeCanvas);
    window.addEventListener("resize", resizeCanvas);
  </script>
</body>
</html>
"""


PROGRESS_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Analyzing Video</title>
  <style>
    :root {
      --ink: #1e293b; --muted: #64748b; --line: #d9e2ec; --accent: #0f766e;
      --danger: #b91c1c; --bg: #f7fafc; --panel: #ffffff;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: Segoe UI, Roboto, Arial, sans-serif; background: var(--bg); color: var(--ink); }
    header { padding: 18px 24px; border-bottom: 1px solid var(--line); background: var(--panel); }
    h1 { margin: 0; font-size: 21px; font-weight: 650; }
    main { max-width: 760px; margin: 0 auto; padding: 28px; display: grid; gap: 16px; }
    section { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 18px; }
    .bar { height: 18px; background: #e2e8f0; border-radius: 999px; overflow: hidden; }
    .fill { height: 100%; width: 0%; background: var(--accent); transition: width 180ms ease; }
    .meta { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; margin-top: 14px; color: var(--muted); }
    .meta strong { color: var(--ink); }
    .error { color: var(--danger); white-space: pre-wrap; }
    .button {
      border: 0; border-radius: 7px; padding: 10px 12px; color: #fff; background: var(--accent);
      font-weight: 650; text-decoration: none; display: inline-flex; min-height: 38px; align-items: center;
      width: max-content;
    }
    @media (max-width: 700px) { .meta { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header><h1>Analyzing Video</h1></header>
  <main>
    <section>
      <p id="message">Starting analysis...</p>
      <div class="bar"><div class="fill" id="fill"></div></div>
      <div class="meta">
        <div>Progress: <strong id="percent">0%</strong></div>
        <div>Frames: <strong id="frames">0 / 0</strong></div>
        <div>Total active: <strong id="active">0.000s</strong></div>
        <div>Intervals: <strong id="intervals">0</strong></div>
      </div>
    </section>
    <section id="done" style="display:none;">
      <a class="button" href="{{ url_for('results', job_id=metadata.job_id) }}">Open Results</a>
    </section>
    <section id="errorPanel" style="display:none;">
      <p class="error" id="errorText"></p>
    </section>
  </main>
  <script>
    async function poll() {
      const response = await fetch("{{ url_for('status', job_id=metadata.job_id) }}", { cache: "no-store" });
      const status = await response.json();
      const percent = Number(status.percent || 0);
      document.getElementById("fill").style.width = `${Math.max(0, Math.min(100, percent))}%`;
      document.getElementById("percent").textContent = `${percent.toFixed(1)}%`;
      document.getElementById("frames").textContent = `${status.processed_frames || 0} / ${status.total_frames || 0}`;
      document.getElementById("active").textContent = `${Number(status.total_active_seconds || 0).toFixed(3)}s`;
      document.getElementById("intervals").textContent = status.interval_count || 0;
      document.getElementById("message").textContent = status.message || status.state || "Working";
      if (status.state === "complete") {
        window.location.href = "{{ url_for('results', job_id=metadata.job_id) }}";
        return;
      }
      if (status.state === "error") {
        document.getElementById("errorPanel").style.display = "block";
        document.getElementById("errorText").textContent = status.traceback || status.message || "Analysis failed.";
        return;
      }
      setTimeout(poll, 1000);
    }
    poll();
  </script>
</body>
</html>
"""


RESULT_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Analysis Results</title>
  <style>
    :root {
      --ink: #1e293b; --muted: #64748b; --line: #d9e2ec; --accent: #0f766e;
      --accent-strong: #115e59; --bg: #f7fafc; --panel: #ffffff;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: Segoe UI, Roboto, Arial, sans-serif; background: var(--bg); color: var(--ink); }
    header { padding: 18px 24px; border-bottom: 1px solid var(--line); background: var(--panel); }
    h1 { margin: 0; font-size: 21px; font-weight: 650; }
    main { padding: 20px; display: grid; gap: 18px; }
    .summary { display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap: 12px; }
    .metric, section { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }
    .metric span { display: block; color: var(--muted); font-size: 13px; }
    .metric strong { display: block; margin-top: 6px; font-size: 24px; }
    table { width: 100%; border-collapse: collapse; background: var(--panel); }
    th, td { padding: 9px 10px; border-bottom: 1px solid var(--line); text-align: left; font-size: 14px; }
    th { color: var(--muted); font-weight: 650; }
    .actions { display: flex; gap: 10px; flex-wrap: wrap; }
    .button {
      border: 0; border-radius: 7px; padding: 10px 12px; color: #fff; background: var(--accent);
      font-weight: 650; text-decoration: none; display: inline-flex; min-height: 38px; align-items: center;
    }
    .button:hover { background: var(--accent-strong); }
    video { max-width: 100%; border-radius: 8px; border: 1px solid var(--line); background: #111827; }
    @media (max-width: 900px) { .summary { grid-template-columns: 1fr 1fr; } }
  </style>
</head>
<body>
  <header><h1>Analysis Results</h1></header>
  <main>
    <div class="summary">
      <div class="metric"><span>Total active time</span><strong>{{ "%.2f"|format(result.total_active_seconds) }}s</strong></div>
      <div class="metric"><span>Intervals</span><strong>{{ result.interval_count }}</strong></div>
      <div class="metric"><span>Processed frames</span><strong>{{ result.processed_frames }}</strong></div>
      <div class="metric"><span>Video FPS</span><strong>{{ "%.1f"|format(result.source_fps) }}</strong></div>
    </div>
    <div class="actions">
      <a class="button" href="{{ url_for('job_file', job_id=metadata.job_id, filename='analysis_intervals.csv') }}" download>Download CSV</a>
      {% if annotated_name %}
      <a class="button" href="{{ url_for('job_file', job_id=metadata.job_id, filename=annotated_name) }}" download>Download Annotated Video</a>
      {% endif %}
      <a class="button" href="{{ url_for('index') }}">New Video</a>
    </div>
    {% if annotated_name %}
    <section>
      <video controls src="{{ url_for('job_file', job_id=metadata.job_id, filename=annotated_name) }}"></video>
    </section>
    {% endif %}
    <section>
      <table>
        <thead>
          <tr>
            <th>#</th><th>Date</th><th>Start</th><th>End</th><th>Duration</th><th>Total</th><th>Status</th>
          </tr>
        </thead>
        <tbody>
          {% for row in rows %}
          <tr>
            <td>{{ row.interval_number }}</td>
            <td>{{ row.date }}</td>
            <td>{{ row.start_timestamp }}</td>
            <td>{{ row.end_timestamp }}</td>
            <td>{{ "%.3f"|format(row.active_duration_seconds) }}</td>
            <td>{{ "%.3f"|format(row.total_accumulated_seconds) }}</td>
            <td>{{ row.detection_status }}</td>
          </tr>
          {% endfor %}
          {% if not rows %}
          <tr><td colspan="7">No active intervals detected.</td></tr>
          {% endif %}
        </tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""


if __name__ == "__main__":
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True, use_reloader=False)
