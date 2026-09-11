#!/usr/bin/env python3
"""Small, targeted experiment sweep — window size, stride, and confidence
threshold — evaluated against real measured results only.

Honest caveat, stated here and in report.md: only one real annotated video
(data/samples/vtest.avi) exists for this project (see the data-sourcing
limitation in docs/designs/activity-detection-baseline.md). With a single
video there is no clean validation/test split to tune on one and report the
other — the numbers below are a sensitivity analysis on that one video, not a
validation-selected config verified on a held-out test set. This is disclosed
explicitly rather than presented as more rigorous than it is.

    python -m scripts.run_experiments
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.metrics import evaluate
from src.pipeline.run import build_segments_from_series, compute_class_series
from src.utils import setup_logging

VIDEO = "data/samples/vtest.avi"
ZONES = "vtest"
GT_PATH = "data/ground_truth/vtest.json"


def load_gt() -> list[dict]:
    with open(GT_PATH) as f:
        return json.load(f)["segments"]


def main() -> int:
    setup_logging()
    ground_truth = load_gt()

    configs = [
        {"name": "A: baseline (w=3.0 s=1.5)", "window_size_s": 3.0, "stride_s": 1.5},
        {"name": "B: no window overlap (w=3.0 s=3.0)", "window_size_s": 3.0, "stride_s": 3.0},
        {"name": "D: wider window (w=5.0 s=2.5)", "window_size_s": 5.0, "stride_s": 2.5},
    ]
    thresholds = [0.4, 0.5, 0.6]

    rows = []
    for cfg in configs:
        print(f"\n=== {cfg['name']} ===")
        raw = compute_class_series(VIDEO, zones_name=ZONES, window_size_s=cfg["window_size_s"], stride_s=cfg["stride_s"])
        for threshold in thresholds:
            segments = build_segments_from_series(
                raw["class_series"], raw["class_intervals"], raw["classes_config"],
                raw["post_processing_cfg"], threshold=threshold,
            )
            result = evaluate(segments, ground_truth, iou_thresholds=(0.5,))
            m = result["overall"]["0.5"]
            row = {
                "experiment": cfg["name"], "window_s": cfg["window_size_s"], "stride_s": cfg["stride_s"],
                "threshold": threshold, "num_predictions": len(segments),
                "precision": round(m["precision"], 3), "recall": round(m["recall"], 3), "f1": round(m["f1"], 3),
            }
            rows.append(row)
            print(f"  threshold={threshold}: precision={m['precision']:.3f} recall={m['recall']:.3f} f1={m['f1']:.3f} (n_pred={len(segments)})")

    print("\n| Experiment | Window | Stride | Threshold | Precision | Recall | F1 |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['experiment']} | {r['window_s']} | {r['stride_s']} | {r['threshold']} | {r['precision']} | {r['recall']} | {r['f1']} |")

    best = max(rows, key=lambda r: r["f1"])
    print(f"\nBest by F1 (tIoU>=0.5) on this single video: {best}")

    out_path = Path("outputs/metrics/experiments.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nFull results written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
