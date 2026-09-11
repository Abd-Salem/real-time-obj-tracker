# Object Tracker

A real-time object tracking system built with **YOLOv8 (Ultralytics)**, **ByteTrack** (via Supervision), and **OpenCV**.

The system runs continuous tracking on a camera feed, and can accept a **Region of Interest (ROI)** at any time to detect and start tracking new objects inside that region without interrupting existing tracks.

---

## How It Works

```
WAITING FOR ROI
       │
       │ ROI arrives
       ▼
DETECT ROI
       │
       ├── existing object → ignore
       │
       └── new object → create track
                       │
                       ▼
                CONTINUOUS TRACKING
                       │
                       ├── YOLO every ~200 ms
                       │
                       ├── tracker every camera frame
                       │
                       ├── existing object disappears
                       │        ↓
                       │     track lost/deleted
                       │
                       └── new ROI can arrive ANY TIME
                                │
                                ▼
                         repeat ROI detection
```

### Key behaviors

- **Detection runs at reduced FPS** (`DETECTION_FPS = 5`) to save CPU, while the **tracker updates every camera frame** (`CAMERA_FPS = 30`) so tracks stay smooth and tracks age/predict correctly.
- **ROI requests never stop tracking** they are queued and processed in the next frame.
- Objects inside an ROI that are **already tracked are ignored** (IoU check against existing tracks); only **new objects are added** to the tracker immediately, without waiting for the next full-frame YOLO pass.
- Tracked classes (COCO): `person, car, motorcycle, bus, truck` (`CLASSES = [0, 2, 3, 5, 7]`).

---

## Project Structure

```
.
├── src/
│   ├── main.py                  # Entry point: camera loop + drawing
│   ├── obj_tracking_system.py   # ObjectTrackingSystem (ROI + detection + tracking)
│   ├── detector.py              # Model creation / inference wrapper
│   ├── tracker.py               # Tracker (ByteTrack) creation
│   └── __init__.py
├── helper/
│   └── roi.py                   # ROI helper
├── configs.py                   # Config loader
├── configs.yml                  # Config values
├── pyproject.toml
├── uv.lock
└── .python-version
```

---

## Requirements

- Python **3.14** (managed automatically by `uv` via `.python-version`)
- A webcam (or change the `cv2.VideoCapture` source in `src/main.py`)
- Internet connection **on first run** Ultralytics downloads `yolov8n.pt` automatically

> PyTorch is pinned to the **CPU-only** build, so no NVIDIA GPU / CUDA is required.

---

## 1. Install `uv`

If you don't have `uv` installed yet:

**Linux / macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify:
```bash
uv --version
```

---

## 2. Install dependencies

From the project root (the folder containing `pyproject.toml`):

```bash
uv sync
```

This will:

- download Python 3.14 if needed,
- create the `.venv` virtual environment,
- install all locked dependencies from `uv.lock` — including the CPU build of PyTorch from the PyTorch index.

> Use `uv sync --locked` in CI to guarantee the exact locked versions are installed.

---

## 3. Run

```bash
uv run src/main.py
```

Or, if you prefer activating the environment manually:

```bash
# Linux / macOS
source .venv/bin/activate
python src/main.py

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
python src/main.py
```

A window titled **"Object Tracking"** opens showing the camera feed with tracked objects drawn as green boxes labeled `ID: <n>`.

**Press `q`** (with the window focused) to quit.

---

## Configuration

Settings live at the top of `src/obj_tracking_system.py` (and/or `configs.yml`, loaded via `configs.py`):

| Setting | Default | Description |
|---|---|---|
| `CAMERA_FPS` | `30` | Camera frame rate |
| `DETECTION_FPS` | `5` | Full-frame YOLO inference rate (~200 ms interval) |
| `MODEL_PATH` | `yolov8n.pt` | YOLO model weights (auto-downloaded on first run) |
| `TRACKER_CONFIG` | `bytetrack.yaml` | ByteTrack tracker configuration |
| `CLASSES` | `[0, 2, 3, 5, 7]` | COCO class IDs to detect |

---

## Troubleshooting

- **Camera not found / black window** make sure no other app is using the webcam, or change `cv2.VideoCapture(0)` to a video file path in `src/main.py`.
- **First run is slow** the YOLO weights (`yolov8n.pt`) are being downloaded; subsequent runs start instantly.
- **Slow on old hardware** lower `DETECTION_FPS` (e.g. `2`) in the config.
- **No detections** confirm the scene contains one of the tracked classes and that lighting is adequate.
