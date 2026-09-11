"""Turns per-class score timelines from either branch into final
{start, end, label, confidence} segments.

Shared temporal post-processing pipeline (same steps for both branches, since by
the time scores reach here they're just (timestamp, score) series in [0,1] — see
"Score normalization" in the design doc for why no cross-branch calibration is
needed): smoothing -> thresholding -> consecutive-run grouping -> gap merging ->
min-duration filtering. Segments from different classes may overlap in time
(e.g. Crowd Formation concurrent with Running); segments of the same class never
overlap after merging.
"""
from __future__ import annotations

import statistics
from typing import Any


def moving_median(values: list[float], window: int) -> list[float]:
    if window <= 1 or not values:
        return list(values)
    n = len(values)
    half = window // 2
    return [statistics.median(values[max(0, i - half): min(n, i + half + 1)]) for i in range(n)]


def build_segments_for_class(
    series: list[tuple[float, float]],
    interval_s: float,
    label: str,
    threshold: float,
    smoothing_window: int,
    min_duration_s: float,
    merge_gap_s: float,
) -> list[dict[str, Any]]:
    """series: sorted (timestamp, score) pairs, one per sampled frame/window.
    interval_s: the time each sample "covers" (1/sample_fps for the geometry
    branch, stride_s for the appearance branch) — used to turn the last active
    sample's timestamp into a segment end time."""
    if not series:
        return []
    series = sorted(series)
    timestamps = [t for t, _ in series]
    scores = [s for _, s in series]
    smoothed = moving_median(scores, smoothing_window)
    active = [s >= threshold for s in smoothed]

    raw_segments: list[list[float]] = []
    i, n = 0, len(timestamps)
    while i < n:
        if active[i]:
            j = i
            while j < n and active[j]:
                j += 1
            raw_segments.append([timestamps[i], timestamps[j - 1] + interval_s, max(smoothed[i:j])])
            i = j
        else:
            i += 1

    merged: list[list[float]] = []
    for seg in raw_segments:
        if merged and seg[0] - merged[-1][1] <= merge_gap_s:
            merged[-1][1] = max(merged[-1][1], seg[1])
            merged[-1][2] = max(merged[-1][2], seg[2])
        else:
            merged.append(seg)

    return [
        {"start": round(s, 2), "end": round(e, 2), "label": label, "confidence": round(c, 3)}
        for s, e, c in merged
        if e - s >= min_duration_s
    ]


def build_all_segments(
    class_series: dict[str, list[tuple[float, float]]],
    class_intervals: dict[str, float],
    class_display_names: dict[str, str],
    post_processing_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """class_series: e.g. {"fighting": [(t, score), ...], "intrusion": [...], ...}."""
    all_segments: list[dict[str, Any]] = []
    for class_key, series in class_series.items():
        segments = build_segments_for_class(
            series,
            interval_s=class_intervals[class_key],
            label=class_display_names[class_key],
            threshold=post_processing_cfg["confidence_threshold"],
            smoothing_window=post_processing_cfg["smoothing_window_frames"],
            min_duration_s=post_processing_cfg["min_segment_duration_s"],
            merge_gap_s=post_processing_cfg["merge_gap_s"],
        )
        all_segments.extend(segments)
    all_segments.sort(key=lambda s: s["start"])
    return all_segments
