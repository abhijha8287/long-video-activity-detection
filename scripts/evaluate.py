#!/usr/bin/env python3
"""Evaluate predictions against ground truth.

    python -m scripts.evaluate --predictions outputs/predictions/video.json \
        --ground-truth data/ground_truth/video.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.metrics import BACKGROUND_LABEL, MISSED_LABEL, evaluate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate predicted segments against ground truth.")
    p.add_argument("--pred", "--predictions", dest="predictions", required=True)
    p.add_argument("--gt", "--ground-truth", dest="ground_truth", required=True)
    p.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    p.add_argument("--output", default=None, help="Optional path to write the metrics JSON (default: outputs/metrics/<video>_metrics.json).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    with open(args.predictions) as f:
        pred_data = json.load(f)
    with open(args.ground_truth) as f:
        gt_data = json.load(f)

    predictions = pred_data["segments"]
    ground_truth = gt_data["segments"]

    result = evaluate(predictions, ground_truth, iou_thresholds=tuple(args.iou_thresholds))

    print(f"\nVideo: {pred_data.get('video', args.predictions)}")
    print(f"Predictions: {len(predictions)}   Ground truth segments: {len(ground_truth)}\n")

    for thr_str, m in result["overall"].items():
        print(
            f"tIoU >= {thr_str}: precision={m['precision']:.3f} recall={m['recall']:.3f} "
            f"f1={m['f1']:.3f} (tp={m['tp']} fp={m['fp']} fn={m['fn']}, avg_tiou={m['avg_tiou']:.3f})"
        )

    print("\nPer-class (tIoU >= 0.5):")
    for label, m in sorted(result["per_class"]["0.5"].items()):
        print(f"  {label:35s} P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f} support={m['support']}")

    print("\nConfusion matrix (rows=ground truth incl. background, cols=predicted incl. missed):")
    cm = result["confusion_matrix"]
    labels = list(cm.keys())
    col_labels = list(next(iter(cm.values())).keys()) if cm else []
    print("  " + " ".join(f"{c[:10]:>11s}" for c in col_labels))
    for row_label in labels:
        print(f"  {row_label[:20]:20s}" + " ".join(f"{cm[row_label][c]:11d}" for c in col_labels))

    output_path = Path(args.output) if args.output else Path("outputs/metrics") / f"{Path(args.predictions).stem}_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nFull metrics written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
