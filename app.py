#!/usr/bin/env python3
"""Streamlit demo UI for the activity-detection pipeline.

    streamlit run app.py

Upload any video, run the same pipeline `inference.py` uses, and see the
detected segments, a timeline, and (optionally) geometry-branch debug frames —
all from the browser, no CLI needed. This is a thin UI layer over
`src/pipeline/run.py`; it adds no detection logic of its own.
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import streamlit as st

from src.classifiers.clip_zeroshot import CLIPZeroShotClassifier
from src.detection.detector import TrackedDetector, get_video_info
from src.eval.visualize import plot_timeline, render_bbox_overlay_frames
from src.pipeline.run import build_segments_from_series, compute_class_series
from src.rules.geometry import load_zone_config
from src.utils import format_timestamp, load_classes_config, set_seed

st.set_page_config(page_title="Activity Detection", layout="wide")

UPLOAD_DIR = Path("outputs/streamlit_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ZONES_DIR = Path("configs/zones")


@st.cache_resource(show_spinner="Loading YOLO11n + ByteTrack (one-time, cached)...")
def get_detector(weights: str, device: str) -> TrackedDetector:
    return TrackedDetector(weights=weights, device=device)


@st.cache_resource(show_spinner="Loading CLIP ViT-B/32 (one-time, cached)...")
def get_clip_classifier(device: str) -> CLIPZeroShotClassifier:
    classes_config, _ = load_classes_config()
    return CLIPZeroShotClassifier(classes_config, device=device)


def available_zone_names() -> list[str]:
    if not ZONES_DIR.exists():
        return []
    return sorted(p.stem for p in ZONES_DIR.glob("*.yaml"))


def save_upload(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    dest = UPLOAD_DIR / f"{Path(uploaded_file.name).stem}_{int(time.time())}{suffix}"
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest


def main() -> None:
    st.title("Activity Detection & Classification in Long Videos")
    st.caption(
        "Hybrid baseline — YOLO11n + ByteTrack + rule engine for position/dwell-defined classes, "
        "CLIP zero-shot for appearance-defined classes. See `report.md` for the full design rationale."
    )

    with st.sidebar:
        st.header("Settings")
        device = st.selectbox("Device", ["auto", "cpu", "mps", "cuda"], index=0)

        zone_options = ["(none)"] + available_zone_names()
        zones_choice = st.selectbox(
            "Zone config",
            zone_options,
            help="Required for Intrusion and Running to be evaluated — those two classes need a "
            "restricted/no-running zone defined for this camera. Without one, they're correctly "
            "reported as not applicable rather than guessed.",
        )
        zones_name = None if zones_choice == "(none)" else zones_choice

        classes_config, default_post_cfg = load_classes_config()
        threshold = st.slider(
            "Confidence threshold", 0.0, 1.0, float(default_post_cfg["confidence_threshold"]), 0.05
        )

        with st.expander("Advanced (temporal windowing)"):
            sample_fps = st.number_input("Geometry-branch sample fps", 1.0, 15.0, 8.0, 1.0)
            window_size_s = st.number_input("Appearance window size (s)", 1.0, 10.0, 3.0, 0.5)
            stride_s = st.number_input("Appearance window stride (s)", 0.5, 10.0, 1.5, 0.5)
            show_overlay = st.checkbox(
                "Generate geometry-branch debug frames (bbox/track/zone overlay)", value=False,
                help="Re-runs detection at a lower sample rate to render a few annotated frames — adds time.",
            )

        st.divider()
        st.caption("10 classes: Fire/Smoke, Fighting, Falling, Intrusion, Loitering, Vandalism, "
                   "Theft, Crowd Formation, Running, Abandoned Object.")

    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv", "webm"])

    sample_col1, sample_col2 = st.columns([3, 1])
    with sample_col2:
        st.caption("Or try a bundled sample:")
        sample_videos = sorted(Path("data/samples").glob("*"))
        sample_videos = [p for p in sample_videos if p.suffix in (".mp4", ".avi", ".webm")]
        sample_choice = st.selectbox(
            "Sample video", ["(none)"] + [p.name for p in sample_videos], label_visibility="collapsed"
        )

    video_path: Path | None = None
    if uploaded is not None:
        video_path = save_upload(uploaded)
    elif sample_choice != "(none)":
        video_path = Path("data/samples") / sample_choice

    if video_path is None:
        st.info("Upload a video, or pick a bundled sample, to run detection.")
        return

    info = get_video_info(video_path)
    st.video(str(video_path))
    st.caption(f"{video_path.name} — {info.duration_s:.1f}s, {info.width}x{info.height}, {info.fps:.1f} fps")

    if zones_name is None and any(c in classes_config for c in ("intrusion", "running")):
        st.warning(
            "No zone config selected — Intrusion and Running will be reported as not applicable "
            "for this run, rather than guessed. Pick a zone config in the sidebar if this camera "
            "needs those classes evaluated."
        )

    if st.button("Run detection", type="primary"):
        set_seed()
        detector = get_detector("yolo11n.pt", device)
        clip_classifier = get_clip_classifier(device)

        t0 = time.time()
        with st.spinner("Running geometry branch (YOLO + ByteTrack + rules) and appearance branch (CLIP)..."):
            raw = compute_class_series(
                video_path,
                zones_name=zones_name,
                sample_fps=sample_fps,
                window_size_s=window_size_s,
                stride_s=stride_s,
                device=device,
                detector=detector,
                clip_classifier=clip_classifier,
            )
            segments = build_segments_from_series(
                raw["class_series"], raw["class_intervals"], raw["classes_config"],
                raw["post_processing_cfg"], threshold=threshold,
            )
            for seg in segments:
                if seg["end"] > info.duration_s:
                    seg["end"] = round(info.duration_s, 2)
        elapsed = time.time() - t0

        st.session_state["result"] = {"video": str(video_path), "segments": segments}
        st.session_state["elapsed"] = elapsed
        st.session_state["video_path"] = str(video_path)
        st.session_state["zones_name"] = zones_name
        st.session_state["show_overlay"] = show_overlay
        st.session_state["sample_fps"] = sample_fps

    result = st.session_state.get("result")
    if result is None or st.session_state.get("video_path") != str(video_path):
        return

    segments = result["segments"]
    elapsed = st.session_state["elapsed"]

    st.subheader("Results")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Segments detected", len(segments))
    m2.metric("Processing time", f"{elapsed:.1f}s")
    m3.metric("Real-time factor", f"{elapsed / max(info.duration_s, 1e-6):.2f}x")
    m4.metric("Video duration", f"{info.duration_s:.1f}s")

    if not segments:
        st.write("No activities detected above threshold.")
    else:
        table_rows = [
            {
                "Start": format_timestamp(s["start"]),
                "End": format_timestamp(s["end"]),
                "Label": s["label"],
                "Confidence": round(s["confidence"], 3),
            }
            for s in sorted(segments, key=lambda s: s["start"])
        ]
        st.dataframe(table_rows, width="stretch", hide_index=True)

        fig_path = UPLOAD_DIR / f"{Path(video_path).stem}_timeline.png"
        plot_timeline(segments, [], info.duration_s, f"{Path(video_path).name} — predictions", fig_path)
        st.image(str(fig_path), caption="Predicted segments timeline (no ground truth shown for an uploaded video)")

    st.download_button(
        "Download predictions JSON",
        data=json.dumps(result, indent=2),
        file_name=f"{Path(video_path).stem}_predictions.json",
        mime="application/json",
    )

    if st.session_state.get("show_overlay"):
        with st.spinner("Rendering geometry-branch debug frames..."):
            zone_config = load_zone_config(ZONES_DIR / f"{zones_name}.yaml") if zones_name else None
            frame_paths = render_bbox_overlay_frames(
                video_path, weights="yolo11n.pt", zone_config=zone_config, device=device,
                sample_fps=st.session_state["sample_fps"], num_frames=6, output_dir=str(UPLOAD_DIR),
            )
        st.subheader("Geometry-branch debug frames")
        st.caption("Green boxes = tracked detections. Colored polygons = configured zones (red=restricted, "
                   "blue=loitering, orange=no-running).")
        cols = st.columns(3)
        for i, fp in enumerate(frame_paths):
            cols[i % 3].image(str(fp))


if __name__ == "__main__":
    main()
