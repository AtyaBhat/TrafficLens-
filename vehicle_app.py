import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
import streamlit as st
import supervision as sv
from ultralytics import YOLO

st.set_page_config(page_title="Vehicle Tracker & Counter", page_icon="🚗", layout="wide")

# COCO class ids relevant to traffic. 6 (train) is included as an option
# but not selected by default, matching the notebook's original [1,2,3,5,7].
VEHICLE_CLASSES = {
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "train": 6,
    "truck": 7,
}

MODEL_OPTIONS = {
    "YOLOv8 Nano (fastest, recommended for CPU/cloud)": "yolov8n.pt",
    "YOLOv8 Small": "yolov8s.pt",
    "YOLOv8 Medium": "yolov8m.pt",
    "YOLOv8 Large": "yolov8l.pt",
    "YOLOv8 XLarge (notebook default - very slow without a GPU)": "yolov8x.pt",
}


@st.cache_resource(show_spinner="Loading YOLO model...")
def load_model(weights: str):
    model = YOLO(weights)
    model.fuse()
    return model


def draw_preview(frame: np.ndarray, model: YOLO, selected_class_ids: list[int],
                  line_y: int, confidence: float) -> np.ndarray:
    """Runs detection on a single frame and draws the counting line, so the
    user can see and adjust line placement before committing to processing
    the whole video."""
    results = model(frame, conf=confidence, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[np.isin(detections.class_id, selected_class_ids)]

    class_names = model.model.names
    labels = [
        f"{class_names[class_id]} {confidence_score:0.2f}"
        for _, _, confidence_score, class_id, _, _ in detections
    ]

    box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
    annotated = box_annotator.annotate(scene=frame.copy(), detections=detections, labels=labels)

    width = frame.shape[1]
    annotated = sv.draw_line(
        scene=annotated,
        start=sv.Point(0, line_y),
        end=sv.Point(width, line_y),
        color=sv.Color.GREEN,
        thickness=3,
    )
    return annotated


def reencode_for_browser(input_path: str, output_path: str) -> str:
    """cv2.VideoWriter's mp4v codec doesn't always play inline in browsers.
    Re-encode with ffmpeg (H.264) if it's available on the system; otherwise
    fall back to the original file - downloads still work either way."""
    if shutil.which("ffmpeg") is None:
        return input_path

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-vcodec", "libx264",
             "-pix_fmt", "yuv420p", output_path],
            check=True, capture_output=True,
        )
        return output_path
    except subprocess.CalledProcessError:
        return input_path


def process_video(source_path: str, model: YOLO, selected_class_ids: list[int],
                   line_y: int, confidence: float, max_width: int,
                   progress_callback) -> tuple[str, int, int]:
    """Runs detection + ByteTrack + line-crossing counting over every frame,
    writes an annotated output video, and returns (output_path, in_count, out_count)."""

    cap = cv2.VideoCapture(source_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Downscale large videos for speed on CPU-only deployments.
    scale = min(1.0, max_width / orig_width) if orig_width > max_width else 1.0
    width = int(orig_width * scale)
    height = int(orig_height * scale)
    scaled_line_y = int(line_y * scale)

    byte_tracker = sv.ByteTrack(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=int(fps))
    line_zone = sv.LineZone(start=sv.Point(0, scaled_line_y), end=sv.Point(width, scaled_line_y))
    box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
    trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=50)
    line_zone_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)

    class_names = model.model.names

    raw_output_path = tempfile.mktemp(suffix="_raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if scale != 1.0:
            frame = cv2.resize(frame, (width, height))

        results = model(frame, conf=confidence, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[np.isin(detections.class_id, selected_class_ids)]
        detections = byte_tracker.update_with_detections(detections)

        labels = [
            f"#{tracker_id} {class_names[class_id]} {conf_score:0.2f}"
            for _, _, conf_score, class_id, tracker_id, _ in detections
        ]

        annotated = trace_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated = box_annotator.annotate(scene=annotated, detections=detections, labels=labels)
        line_zone.trigger(detections)
        annotated = line_zone_annotator.annotate(annotated, line_counter=line_zone)

        writer.write(annotated)

        frame_index += 1
        if total_frames > 0:
            progress_callback(frame_index / total_frames)

    cap.release()
    writer.release()

    final_output_path = tempfile.mktemp(suffix="_final.mp4")
    final_output_path = reencode_for_browser(raw_output_path, final_output_path)

    return final_output_path, line_zone.in_count, line_zone.out_count


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🚗 Vehicle Tracking & Counting")
st.caption("YOLO detection + ByteTrack tracking + line-crossing counting, built on `supervision`.")

with st.sidebar:
    st.header("⚙️ Settings")

    model_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()), index=0)
    weights = MODEL_OPTIONS[model_label]

    selected_names = st.multiselect(
        "Classes to track",
        options=list(VEHICLE_CLASSES.keys()),
        default=["bicycle", "car", "motorcycle", "bus", "truck"],
    )
    selected_class_ids = [VEHICLE_CLASSES[n] for n in selected_names]

    confidence = st.slider("Confidence threshold", 0.1, 0.9, 0.3, 0.05)

    max_width = st.select_slider(
        "Max processing width (px)",
        options=[480, 640, 960, 1280, 1920],
        value=960,
        help="Frames are downscaled to this width before processing - smaller is much faster on CPU.",
    )

uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_video:
    tmp_dir = tempfile.mkdtemp()
    source_path = os.path.join(tmp_dir, uploaded_video.name)
    with open(source_path, "wb") as f:
        f.write(uploaded_video.read())

    cap = cv2.VideoCapture(source_path)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, first_frame = cap.read()
    cap.release()

    if not ok:
        st.error("Couldn't read that video file. Try a different format.")
        st.stop()

    st.subheader("1. Place the counting line")
    line_position = st.slider(
        "Line height (fraction of frame, top → bottom)",
        0.05, 0.95, 0.5, 0.01,
    )
    line_y = int(line_position * frame_height)

    if not selected_class_ids:
        st.warning("Select at least one class to track in the sidebar.")
    else:
        model = load_model(weights)
        preview = draw_preview(first_frame, model, selected_class_ids, line_y, confidence)
        st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), caption="Preview - adjust the slider above until the line is where you want it")

        st.subheader("2. Process the full video")
        if st.button("Process Video", type="primary"):
            progress_bar = st.progress(0.0, text="Processing frames...")

            def update_progress(fraction):
                progress_bar.progress(min(fraction, 1.0), text=f"Processing frames... {fraction:.0%}")

            output_path, in_count, out_count = process_video(
                source_path, model, selected_class_ids, line_y, confidence, max_width, update_progress
            )

            progress_bar.progress(1.0, text="Done!")

            col1, col2 = st.columns(2)
            col1.metric("Vehicles counted IN", in_count)
            col2.metric("Vehicles counted OUT", out_count)

            st.video(output_path)

            with open(output_path, "rb") as f:
                st.download_button(
                    "Download annotated video",
                    data=f,
                    file_name="tracked_output.mp4",
                    mime="video/mp4",
                )
else:
    st.info("Upload a video to get started.")
