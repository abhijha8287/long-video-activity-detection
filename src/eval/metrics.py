"""Temporal detection evaluation: tIoU, greedy one-to-one matching,
precision/recall/F1 (overall and per-class), and a classification confusion matrix.

Matching strategy (documented in the design doc): greedy highest-tIoU-first,
one-to-one — a ground-truth segment can only be matched to one prediction, and
vice versa. A prediction counts as a true positive only if both the label matches
AND tIoU >= threshold; the confusion matrix uses a separate, label-agnostic
localization match (tIoU >= 0.3) so class-confusion patterns are visible even when
the label is wrong — the detection metrics above are what actually determine TP/FP/FN.

Accuracy is deliberately not computed here as a headline number: with long
background periods dominating most videos, a system that predicts "nothing" almost
everywhere would still score a high frame-level accuracy while catching zero real
activities — precision/recall/F1 on segments is what actually reflects behavior
that matters.
"""
from __future__ import annotations

from typing import Any

Segment = dict[str, Any]

BACKGROUND_LABEL = "<background>"
MISSED_LABEL = "<missed>"


def temporal_iou(a: Segment, b: Segment) -> float:
    inter_start = max(a["start"], b["start"])
    inter_end = min(a["end"], b["end"])
    intersection = max(0.0, inter_end - inter_start)
    union = (a["end"] - a["start"]) + (b["end"] - b["start"]) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _greedy_match(
    predictions: list[Segment],
    ground_truth: list[Segment],
    iou_threshold: float,
    require_label_match: bool,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    candidates = []
    for pi, p in enumerate(predictions):
        for gi, g in enumerate(ground_truth):
            if require_label_match and p["label"] != g["label"]:
                continue
            iou = temporal_iou(p, g)
            if iou >= iou_threshold:
                candidates.append((iou, pi, gi))
    candidates.sort(key=lambda c: c[0], reverse=True)

    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, pi, gi in candidates:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)
        matches.append((pi, gi, iou))

    unmatched_preds = [i for i in range(len(predictions)) if i not in matched_pred]
    unmatched_gts = [i for i in range(len(ground_truth)) if i not in matched_gt]
    return matches, unmatched_preds, unmatched_gts


def precision_recall_f1(
    predictions: list[Segment], ground_truth: list[Segment], iou_threshold: float
) -> dict[str, float]:
    matches, unmatched_preds, unmatched_gts = _greedy_match(
        predictions, ground_truth, iou_threshold, require_label_match=True
    )
    tp, fp, fn = len(matches), len(unmatched_preds), len(unmatched_gts)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_tiou = sum(m[2] for m in matches) / len(matches) if matches else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "avg_tiou": avg_tiou}


def per_class_metrics(
    predictions: list[Segment], ground_truth: list[Segment], iou_threshold: float
) -> dict[str, dict[str, float]]:
    labels = sorted({g["label"] for g in ground_truth} | {p["label"] for p in predictions})
    out = {}
    for label in labels:
        preds_l = [p for p in predictions if p["label"] == label]
        gts_l = [g for g in ground_truth if g["label"] == label]
        metrics = precision_recall_f1(preds_l, gts_l, iou_threshold)
        metrics["support"] = len(gts_l)
        out[label] = metrics
    return out


def confusion_matrix(
    predictions: list[Segment], ground_truth: list[Segment], localization_iou_threshold: float = 0.3
) -> dict[str, dict[str, int]]:
    """Rows = ground-truth label (+ '<background>' for unmatched predictions),
    columns = predicted label (+ '<missed>' for unmatched ground truth)."""
    matches, unmatched_preds, unmatched_gts = _greedy_match(
        predictions, ground_truth, localization_iou_threshold, require_label_match=False
    )
    labels = sorted({g["label"] for g in ground_truth} | {p["label"] for p in predictions})
    columns = labels + [MISSED_LABEL]
    matrix = {label: {col: 0 for col in columns} for label in labels + [BACKGROUND_LABEL]}

    for pi, gi, _ in matches:
        matrix[ground_truth[gi]["label"]][predictions[pi]["label"]] += 1
    for gi in unmatched_gts:
        matrix[ground_truth[gi]["label"]][MISSED_LABEL] += 1
    for pi in unmatched_preds:
        matrix[BACKGROUND_LABEL][predictions[pi]["label"]] += 1

    return matrix


def evaluate(
    predictions: list[Segment],
    ground_truth: list[Segment],
    iou_thresholds: tuple[float, ...] = (0.3, 0.5, 0.7),
) -> dict[str, Any]:
    return {
        "overall": {str(t): precision_recall_f1(predictions, ground_truth, t) for t in iou_thresholds},
        "per_class": {str(t): per_class_metrics(predictions, ground_truth, t) for t in iou_thresholds},
        "confusion_matrix": confusion_matrix(predictions, ground_truth),
        "num_predictions": len(predictions),
        "num_ground_truth": len(ground_truth),
    }
