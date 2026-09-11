#!/usr/bin/env python3
"""Run activity detection on any video, including ones never seen during
development. See README.md for the full usage guide.

    python inference.py --video path/to/new_video.mp4

Prints a readable summary of detected activities and writes a predictions JSON
(default: outputs/predictions/<video_stem>.json).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.pipeline.run import run_pipeline
from src.utils import format_timestamp, setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect and classify activities in a long video.")
    p.add_argument("--video", required=True, help="Path to the input video.")
    p.add_argument("--output", default=None, help="Output predictions JSON path (default: outputs/predictions/<video_stem>.json).")
    p.add_argument("--zones", default=None, help="Zone config name (configs/zones/<name>.yaml) for this camera/site. Omit if the video has none of the zone-dependent classes (Intrusion, Running).")
    p.add_argument("--threshold", type=float, default=None, help="Override the confidence threshold from configs/classes.yaml.")
    p.add_argument("--window-size", type=float, default=3.0, help="Appearance-branch sliding window size in seconds.")
    p.add_argument("--stride", type=float, default=1.5, help="Appearance-branch sliding window stride in seconds.")
    p.add_argument("--sample-fps", type=float, default=8.0, help="Geometry-branch detection sampling rate (measured throughput on the reference hardware supports this; see report.md).")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"], help="Compute device.")
    p.add_argument("--config", default="configs/classes.yaml", help="Class definitions / thresholds config.")
    p.add_argument("--weights", default="yolo11n.pt", help="YOLO detector weights.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: video not found: {video_path}", file=sys.stderr)
        return 1

    output_path = Path(args.output) if args.output else Path("outputs/predictions") / f"{video_path.stem}.json"

    result = run_pipeline(
        video_path=video_path,
        output_path=output_path,
        zones_name=args.zones,
        classes_config_path=args.config,
        sample_fps=args.sample_fps,
        window_size_s=args.window_size,
        stride_s=args.stride,
        threshold=args.threshold,
        device=args.device,
        weights=args.weights,
    )

    segments = result["segments"]
    print()
    print("Detected activities:" if segments else "No activities detected above threshold.")
    print()
    for seg in segments:
        print(f"{format_timestamp(seg['start'])} - {format_timestamp(seg['end'])}")
        print(f"{seg['label']}")
        print(f"Confidence: {seg['confidence']:.2f}")
        print()
    print(f"Full predictions written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
