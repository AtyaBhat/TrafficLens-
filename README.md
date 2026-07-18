# 🚗 TrafficLens

**Real-time vehicle detection, tracking, and directional counting from video — built with YOLOv8, ByteTrack, and Streamlit.**

TrafficLens takes any traffic video and turns it into a live analytics tool: it detects vehicles frame-by-frame, tracks each one with a persistent ID as it moves through the scene, and counts how many cross a user-defined virtual line — broken down by direction. No hardcoded video paths, no fixed coordinates — everything is configured through an interactive UI.

---

## ✨ Features

- **Upload any video** — `.mp4`, `.mov`, `.avi`, `.mkv`
- **Choose your model** — YOLOv8 Nano through XLarge, with guidance on the CPU/GPU speed tradeoff
- **Select vehicle classes** — bicycle, car, motorcycle, bus, train, truck (COCO classes), toggle any combination
- **Adjustable confidence threshold** — filter out low-confidence detections
- **Visual line placement** — drag a slider over a live preview frame to position the counting line exactly where you want it, for *any* video resolution
- **Persistent vehicle tracking** — powered by ByteTrack, so each vehicle keeps the same ID across frames even through brief occlusion
- **Directional counting** — separate IN / OUT tallies as vehicles cross the line
- **Speed controls** — downscale processing resolution to trade accuracy for speed on CPU-only environments
- **Browser-ready output** — automatic H.264 re-encoding (via `ffmpeg`) so the result plays inline, not just downloads
- **Live progress bar** — see processing progress frame-by-frame on longer videos

---

## 🧠 How it works

```
Video upload
     │
     ▼
Frame extraction (OpenCV)
     │
     ▼
YOLOv8 object detection  ──►  filtered to selected vehicle classes
     │
     ▼
ByteTrack  ──►  assigns/maintains persistent tracker IDs per vehicle
     │
     ▼
Virtual line-zone trigger  ──►  counts crossings, direction-aware
     │
     ▼
Annotated frame (boxes + trace trails + live count overlay)
     │
     ▼
Output video (H.264) + IN/OUT metrics + download button
```

Detection and tracking are handled by [`ultralytics`](https://github.com/ultralytics/ultralytics) (YOLOv8) and [`supervision`](https://github.com/roboflow/supervision) (ByteTrack, annotators, line-zone counting) — TrafficLens wraps this pipeline in a Streamlit interface so it's usable without touching code.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Object detection | YOLOv8 (Ultralytics) |
| Object tracking | ByteTrack (via `supervision`) |
| Counting logic | `supervision.LineZone` |
| Video I/O | OpenCV |
| Re-encoding | ffmpeg (H.264, for browser playback) |
| UI / App | Streamlit |

---

## 🚀 Getting Started

### 1. Clone and install

```bash
git clone <your-repo-url>
cd trafficlens
pip install -r requirements.txt
```

> **Note:** `opencv-python-headless` requires the system library `libgl1` on Debian/Ubuntu. If running locally on Linux and you hit an import error, install it with:
> ```bash
> sudo apt-get install libgl1
> ```

### 2. Run locally

```bash
streamlit run vehicle_app.py
```

The app opens at `http://localhost:8501`.

### 3. Deploy on Streamlit Cloud

Push this repo to GitHub, then create a new app on [share.streamlit.io](https://share.streamlit.io) pointing at `vehicle_app.py`. Make sure both of these files are present in the repo root:

- `requirements.txt` — Python dependencies
- `packages.txt` — system dependencies (`ffmpeg`, `libgl1`) — Streamlit Cloud reads this automatically to run `apt-get install`

No API keys or secrets are required — the entire pipeline runs locally within the app; YOLO weights download automatically on first run.

---

## 📖 Usage

1. **Upload a video** using the file uploader.
2. **Configure settings** in the sidebar:
   - Pick a model size (Nano is recommended for CPU-only deployments — larger models are significantly slower without a GPU)
   - Select which vehicle classes to track
   - Set a confidence threshold
   - Set a max processing width (lower = faster)
3. **Place the counting line** using the slider under the live preview — the line updates in real time so you can position it exactly where vehicles should be counted.
4. Click **Process Video**. A progress bar tracks frame-by-frame processing.
5. Once complete, view the **IN / OUT counts**, preview the annotated video inline, and **download** the result.

---

## ⚠️ Known Limitations

- **CPU-only performance**: Streamlit Cloud's free tier has no GPU. Larger models (Medium/Large/XLarge) or high-resolution/long videos can be slow or hit resource limits — Nano/Small models with a downscaled processing width are recommended for cloud deployment.
- **Straight-line counting only**: the counting zone is a single straight line; it doesn't currently support polygon/zone-based counting for complex intersections.
- **`supervision` version pinning**: this project is built against `supervision==0.18.0`. Newer versions restructured annotator APIs (e.g. splitting label rendering out of `BoxAnnotator` into a separate `LabelAnnotator`) — upgrading without adjusting the code will break it.

---

## 🗺️ Possible Extensions

- Multi-line / zone-based counting for intersections
- Per-class count breakdown (not just total IN/OUT)
- Speed estimation using frame-to-frame displacement and known distances
- Live camera/RTSP stream support instead of file upload only
- Historical analytics dashboard for multiple processed videos

---

## 📄 License

Add your preferred license here (e.g. MIT).
