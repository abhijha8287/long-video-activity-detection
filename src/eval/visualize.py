#!/usr/bin/env python3
"""Qualitative visualization: GT-vs-prediction timeline plots, and a bbox/track/
zone overlay on a handful of representative frames (for debugging the tracker and
rule engine, and for showing the geometry branch's reasoning in an interview)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from src.detection.detector import TrackedDetector, get_video_info
from src.rules.geometry import ZoneConfig, load_zone_config, scale_polygon

ZONE_COLORS = {"restricted": (0, 0, 255), "loitering": (255, 0, 0), "no_running": (0, 165, 255)}


def plot_timeline(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    video_duration: float,
    title: str,
    output_path: str | Path,
) -> None:
    labels = sorted({s["label"] for s in predictions} | {s["label"] for s in ground_truth})
    if not labels:
        labels = ["(no segments)"]
    fig, ax = plt.subplots(figsize=(13, max(1.4 * len(labels), 2) + 1.5))

    for i, label in enumerate(labels):
        y_gt, y_pred = i * 2, i * 2 + 0.8
        for seg in ground_truth:
            if seg["label"] == label:
                ax.broken_barh([(seg["start"], seg["end"] - seg["start"])], (y_gt, 0.6), facecolors="tab:blue")
        for seg in predictions:
            if seg["label"] == label:
                ax.broken_barh(
                    [(seg["start"], seg["end"] - seg["start"])], (y_pred, 0.6),
                    facecolors="tab:orange", alpha=0.85,
                )

    ax.set_yticks([i * 2 + 0.7 for i in range(len(labels))])
    ax.set_yticklabels(labels)
    ax.set_xlabel("Time (s)")
    ax.set_xlim(0, max(video_duration, 1.0))
    ax.set_title(title)
    ax.legend(
        handles=[Patch(color="tab:blue", label="Ground truth"), Patch(color="tab:orange", label="Prediction")],
        loc="upper right",
    )
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_bbox_overlay_frames(
    video_path: str | Path,
    weights: str = "yolo11n.pt",
    zone_config: ZoneConfig | None = None,
    device: str = "auto",
    sample_fps: float = 5.0,
    num_frames: int = 6,
    output_dir: str | Path = "outputs/visualizations",
) -> list[Path]:
    detector = TrackedDetector(weights=weights, device=device)
    frames_data = list(detector.iter_tracks(video_path, sample_fps=sample_fps))
    if not frames_data:
        return []

    step = max(1, len(frames_data) // num_frames)
    chosen = frames_data[::step][:num_frames]
    out_dir = Path(output_dir) / f"{Path(video_path).stem}_frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for ts, _frame_idx, frame, tracks in chosen:
        img = frame.copy()
        for t in tracks:
            x1, y1, x2, y2 = map(int, t.bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{t.cls_name}#{t.track_id}", (x1, max(y1 - 6, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if zone_config:
            for zone in zone_config.zones:
                pts = scale_polygon(zone.polygon, zone_config.frame_size, (img.shape[1], img.shape[0]))
                pts_arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
                color = ZONE_COLORS.get(zone.zone_type, (255, 255, 255))
                cv2.polylines(img, [pts_arr], isClosed=True, color=color, thickness=2)
                cv2.putText(img, zone.name, (int(pts[0][0]), int(pts[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.putText(img, f"t={ts:.1f}s", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        out_path = out_dir / f"frame_{ts:06.1f}s.jpg"
        cv2.imwrite(str(out_path), img)
        saved.append(out_path)
    return saved


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate GT-vs-prediction timeline + bbox overlay visualizations.")
    p.add_argument("--pred", "--predictions", dest="predictions", required=True)
    p.add_argument("--gt", "--ground-truth", dest="ground_truth", required=True)
    p.add_argument("--video", required=True, help="Source video (for duration + bbox overlay frames).")
    p.add_argument("--zones", default=None, help="Zone config name, for drawing zone polygons on overlay frames.")
    p.add_argument("--title", default=None)
    p.add_argument("--out", default=None, help="Timeline PNG output path.")
    p.add_argument("--skip-overlay", action="store_true", help="Skip the (slower) bbox overlay frames.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    with open(args.predictions) as f:
        pred_data = json.load(f)
    with open(args.ground_truth) as f:
        gt_data = json.load(f)

    info = get_video_info(args.video)
    title = args.title or f"{Path(args.video).name} — predictions vs. ground truth"
    out_path = args.out or Path("outputs/visualizations") / f"{Path(args.video).stem}_timeline.png"

    plot_timeline(pred_data["segments"], gt_data["segments"], info.duration_s, title, out_path)
    print(f"Timeline saved to: {out_path}")

    if not args.skip_overlay:
        zone_config = load_zone_config(f"configs/zones/{args.zones}.yaml") if args.zones else None
        saved = render_bbox_overlay_frames(args.video, zone_config=zone_config)
        print(f"Bbox overlay frames saved: {[str(p) for p in saved]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
