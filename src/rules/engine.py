"""Deterministic rule engine for the 6 geometry-defined classes.

Runs in batch over a fully-decoded track history for one video (not streaming) —
this is an offline baseline, not a real-time system, so falling detection can look
a couple seconds forward from a candidate event without any architectural cost.

Each rule below implements exactly one row of the "rule-to-class mapping" table in
docs/designs/activity-detection-baseline.md, so no two classes share a mechanism:

    intrusion         -> zone_occupancy   (presence in a "restricted" zone)
    loitering         -> dwell_time       (sustained presence in a "loitering" zone)
    running           -> velocity         (sustained speed inside a "no_running" zone)
    crowd_formation   -> density          (N+ people clustered, zone-independent)
    falling           -> bbox_shape_change (aspect-ratio flip + post-event stillness)
    abandoned_object  -> owner_separation (static object + no person nearby)

All numeric thresholds come from configs/classes.yaml — nothing here is hardcoded.
"""
from __future__ import annotations

import math
from collections import defaultdict
from statistics import median
from typing import Callable

from src.detection.detector import Track
from src.rules.geometry import Zone, ZoneConfig, euclidean, point_in_polygon, scale_polygon

Frame = tuple[float, list[Track]]  # (timestamp, tracks present at that timestamp)


def _centroid(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _height(bbox: tuple[float, float, float, float]) -> float:
    return max(bbox[3] - bbox[1], 1.0)


def _width(bbox: tuple[float, float, float, float]) -> float:
    return max(bbox[2] - bbox[0], 1.0)


def build_track_history(frames: list[Frame]) -> dict[int, list[tuple[float, Track]]]:
    """Groups the per-frame track lists into per-track-id chronological sequences."""
    history: dict[int, list[tuple[float, Track]]] = defaultdict(list)
    for ts, tracks in frames:
        for t in tracks:
            history[t.track_id].append((ts, t))
    return history


def _in_any_zone(point: tuple[float, float], zones: list[Zone], scale_from: tuple[int, int], scale_to: tuple[int, int]) -> bool:
    return any(
        point_in_polygon(point, scale_polygon(z.polygon, scale_from, scale_to))
        for z in zones
    )


def _aggregate_max(
    per_key_series: dict[int, list[tuple[float, float]]], global_timestamps: list[float]
) -> list[tuple[float, float]]:
    lookups = {key: dict(series) for key, series in per_key_series.items()}
    out = []
    for ts in global_timestamps:
        best = 0.0
        for table in lookups.values():
            if ts in table:
                best = max(best, table[ts])
        out.append((ts, best))
    return out


def _duration_series(
    timestamps: list[float], condition_at_index: Callable[[int], bool]
) -> list[float]:
    """Returns, for each index, how long (seconds) the condition has held true
    continuously up to and including that index. Resets to 0 the moment it's false."""
    durations = []
    since: float | None = None
    for i, ts in enumerate(timestamps):
        if condition_at_index(i):
            if since is None:
                since = ts
            durations.append(ts - since)
        else:
            since = None
            durations.append(0.0)
    return durations


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def zone_occupancy_scores(
    track_history: dict[int, list[tuple[float, Track]]],
    zones: list[Zone],
    frame_size: tuple[int, int],
    zone_frame_size: tuple[int, int],
    min_trigger_duration_s: float,
    global_timestamps: list[float],
) -> list[tuple[float, float]]:
    """Intrusion rule: presence-based, near-immediate trigger once inside the zone
    for `min_trigger_duration_s` (a short guard against one noisy detection frame)."""
    per_track: dict[int, list[tuple[float, float]]] = {}
    for tid, points in track_history.items():
        if points[0][1].cls_name != "person" or not zones:
            continue
        timestamps = [ts for ts, _ in points]
        in_zone = [
            _in_any_zone(_centroid(track.bbox), zones, zone_frame_size, frame_size)
            for _, track in points
        ]
        durations = _duration_series(timestamps, lambda i: in_zone[i])
        per_track[tid] = [
            (ts, min(1.0, d / min_trigger_duration_s) if min_trigger_duration_s > 0 else (1.0 if d > 0 else 0.0))
            for ts, d in zip(timestamps, durations)
        ]
    return _aggregate_max(per_track, global_timestamps)


def dwell_time_scores(
    track_history: dict[int, list[tuple[float, Track]]],
    zones: list[Zone],
    frame_size: tuple[int, int],
    zone_frame_size: tuple[int, int],
    dwell_threshold_s: float,
    score_sigmoid_scale_s: float,
    stillness_radius_px: float,
    global_timestamps: list[float],
) -> list[tuple[float, float]]:
    """Loitering rule: sigmoid ramp centered on dwell_threshold_s.

    Loitering means staying roughly in one place, not merely remaining
    somewhere inside a (possibly large) zone while walking around — those are
    different behaviors and were conflated in an earlier version of this rule,
    caught during evaluation (see report.md): pedestrians who never stopped
    moving were flagged as loitering for tens of seconds simply for staying
    inside the zone. The fix adds a positional-stillness requirement: the
    condition holds only while the track hasn't drifted more than
    `stillness_radius_px` from where its current streak began — real walking
    resets the streak, standing around doesn't.
    """
    per_track: dict[int, list[tuple[float, float]]] = {}
    for tid, points in track_history.items():
        if points[0][1].cls_name != "person" or not zones:
            continue
        timestamps = [ts for ts, _ in points]
        tracks = [track for _, track in points]
        centroids = [_centroid(t.bbox) for t in tracks]
        in_zone = [_in_any_zone(c, zones, zone_frame_size, frame_size) for c in centroids]

        durations: list[float] = []
        since_idx: int | None = None
        for i in range(len(points)):
            if not in_zone[i]:
                since_idx = None
                durations.append(0.0)
                continue
            if since_idx is None:
                since_idx = i
            elif euclidean(centroids[i], centroids[since_idx]) > stillness_radius_px:
                since_idx = i  # moved too far from the anchor — streak restarts here
            durations.append(timestamps[i] - timestamps[since_idx])

        per_track[tid] = [
            (ts, _sigmoid((d - dwell_threshold_s) / max(score_sigmoid_scale_s, 1e-6)) if d > 0 else 0.0)
            for ts, d in zip(timestamps, durations)
        ]
    return _aggregate_max(per_track, global_timestamps)


def velocity_scores(
    track_history: dict[int, list[tuple[float, Track]]],
    zones: list[Zone],
    frame_size: tuple[int, int],
    zone_frame_size: tuple[int, int],
    velocity_threshold_bbox_heights_per_s: float,
    min_trigger_duration_s: float,
    global_timestamps: list[float],
) -> list[tuple[float, float]]:
    """Running rule: sustained speed (normalized by the track's own bbox height,
    so it's roughly scale-invariant to camera distance) while inside a no_running
    zone. This is the signal that distinguishes Running from Intrusion — both use
    zone membership, but Running additionally requires sustained high velocity."""
    per_track: dict[int, list[tuple[float, float]]] = {}
    for tid, points in track_history.items():
        if points[0][1].cls_name != "person" or not zones:
            continue
        timestamps = [ts for ts, _ in points]
        tracks = [track for _, track in points]
        velocities = [0.0]
        for i in range(1, len(points)):
            dt = timestamps[i] - timestamps[i - 1]
            if dt <= 0:
                velocities.append(0.0)
                continue
            dist = euclidean(_centroid(tracks[i].bbox), _centroid(tracks[i - 1].bbox))
            velocities.append((dist / _height(tracks[i].bbox)) / dt)
        in_zone = [
            _in_any_zone(_centroid(t.bbox), zones, zone_frame_size, frame_size) for t in tracks
        ]
        condition = lambda i: in_zone[i] and velocities[i] >= velocity_threshold_bbox_heights_per_s
        durations = _duration_series(timestamps, condition)
        per_track[tid] = [
            (ts, min(1.0, d / min_trigger_duration_s) if min_trigger_duration_s > 0 else (1.0 if d > 0 else 0.0))
            for ts, d in zip(timestamps, durations)
        ]
    return _aggregate_max(per_track, global_timestamps)


def density_scores(
    frames: list[Frame],
    min_people: int,
    cluster_radius_px: float,
    sustained_duration_s: float,
) -> list[tuple[float, float]]:
    """Crowd Formation rule: largest connected cluster of person centroids within
    cluster_radius_px of each other, sustained for sustained_duration_s. Not
    zone-restricted — an unusual gathering is unusual anywhere in frame."""

    def largest_cluster_size(tracks: list[Track]) -> int:
        people = [t for t in tracks if t.cls_name == "person"]
        n = len(people)
        if n < min_people:
            return 0
        centroids = [_centroid(t.bbox) for t in people]
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            for j in range(i + 1, n):
                if euclidean(centroids[i], centroids[j]) <= cluster_radius_px:
                    union(i, j)
        counts: dict[int, int] = defaultdict(int)
        for i in range(n):
            counts[find(i)] += 1
        return max(counts.values(), default=0)

    timestamps = [ts for ts, _ in frames]
    cluster_ok = [largest_cluster_size(tracks) >= min_people for _, tracks in frames]
    durations = _duration_series(timestamps, lambda i: cluster_ok[i])
    return [
        (ts, min(1.0, d / sustained_duration_s) if sustained_duration_s > 0 else (1.0 if d > 0 else 0.0))
        for ts, d in zip(timestamps, durations)
    ]


def falling_scores(
    track_history: dict[int, list[tuple[float, Track]]],
    aspect_ratio_flip_ratio: float,
    post_event_stillness_window_s: float,
    post_event_movement_px: float,
    global_timestamps: list[float],
    baseline_lookback_s: float = 3.0,
) -> list[tuple[float, float]]:
    """Falling rule: a sudden width/height flip (person goes from standing/tall to
    lying/wide) followed by near-stillness, distinguishing a collapse from a crouch
    or a bend that recovers."""
    per_track: dict[int, list[tuple[float, float]]] = {}
    for tid, points in track_history.items():
        if points[0][1].cls_name != "person":
            continue
        timestamps = [ts for ts, _ in points]
        tracks = [track for _, track in points]
        aspect_ratios = [_width(t.bbox) / _height(t.bbox) for t in tracks]
        scores = []
        for i, ts in enumerate(timestamps):
            lookback = [
                aspect_ratios[j] for j in range(i) if ts - timestamps[j] <= baseline_lookback_s
            ]
            baseline = median(lookback) if lookback else aspect_ratios[i]
            flipped = baseline > 0 and aspect_ratios[i] >= baseline * aspect_ratio_flip_ratio
            still = False
            if flipped:
                future_idxs = [
                    j for j in range(i, len(points)) if timestamps[j] - ts <= post_event_stillness_window_s
                ]
                if len(future_idxs) >= 2:
                    disp = euclidean(_centroid(tracks[future_idxs[0]].bbox), _centroid(tracks[future_idxs[-1]].bbox))
                    still = disp <= post_event_movement_px
            scores.append((ts, 1.0 if (flipped and still) else 0.0))
        per_track[tid] = scores
    return _aggregate_max(per_track, global_timestamps)


def abandoned_object_scores(
    frames: list[Frame],
    track_history: dict[int, list[tuple[float, Track]]],
    static_object_classes: list[str],
    static_duration_s: float,
    static_movement_px: float,
    owner_proximity_px: float,
    owner_separation_duration_s: float,
    global_timestamps: list[float],
) -> list[tuple[float, float]]:
    """Abandoned Object rule: an object is a candidate once it's been stationary
    for static_duration_s; it's flagged once no person has been within
    owner_proximity_px of it for owner_separation_duration_s.

    Simplification, stated here rather than overclaimed: "owner" is approximated
    as "any nearby person", not a specific tracked identity — re-identifying a
    specific owner across occlusion is out of scope for this baseline. The report
    names this as a limitation.
    """
    person_positions_by_ts: dict[float, list[tuple[float, float]]] = {
        ts: [_centroid(t.bbox) for t in tracks if t.cls_name == "person"] for ts, tracks in frames
    }

    per_object: dict[int, list[tuple[float, float]]] = {}
    for tid, points in track_history.items():
        cls_name = points[0][1].cls_name
        if cls_name not in static_object_classes:
            continue
        timestamps = [ts for ts, _ in points]
        tracks = [track for _, track in points]

        def is_static(i: int) -> bool:
            ts = timestamps[i]
            ref_idx = None
            for j in range(i, -1, -1):
                if ts - timestamps[j] >= static_duration_s:
                    ref_idx = j
                    break
            if ref_idx is None:
                return False
            disp = euclidean(_centroid(tracks[i].bbox), _centroid(tracks[ref_idx].bbox))
            return disp <= static_movement_px

        def no_owner_nearby(i: int) -> bool:
            ts = timestamps[i]
            obj_pos = _centroid(tracks[i].bbox)
            people = person_positions_by_ts.get(ts, [])
            return all(euclidean(obj_pos, p) > owner_proximity_px for p in people)

        condition = lambda i: is_static(i) and no_owner_nearby(i)
        durations = _duration_series(timestamps, condition)
        per_object[tid] = [
            (ts, min(1.0, d / owner_separation_duration_s) if owner_separation_duration_s > 0 else (1.0 if d > 0 else 0.0))
            for ts, d in zip(timestamps, durations)
        ]
    return _aggregate_max(per_object, global_timestamps)


class RuleEngine:
    """Runs all 6 geometry rules over one video's decoded track history and returns
    a per-class score timeline aligned to the sampled frame timestamps."""

    GEOMETRY_CLASSES = (
        "intrusion",
        "loitering",
        "running",
        "crowd_formation",
        "falling",
        "abandoned_object",
    )

    def __init__(self, classes_config: dict, frame_size: tuple[int, int], zone_config: ZoneConfig | None):
        self.classes_config = classes_config
        self.frame_size = frame_size
        self.zone_config = zone_config

    def compute(self, frames: list[Frame]) -> dict[str, list[tuple[float, float]]]:
        global_timestamps = [ts for ts, _ in frames]
        track_history = build_track_history(frames)
        zone_frame_size = self.zone_config.frame_size if self.zone_config else (0, 0)
        scores: dict[str, list[tuple[float, float]]] = {}

        cfg = self.classes_config["intrusion"]["params"]
        zones = self.zone_config.zones_of_type("restricted") if self.zone_config else []
        scores["intrusion"] = zone_occupancy_scores(
            track_history, zones, self.frame_size, zone_frame_size,
            cfg["min_trigger_duration_s"], global_timestamps,
        )

        cfg = self.classes_config["loitering"]["params"]
        zones = self.zone_config.zones_of_type("loitering") if self.zone_config else []
        scores["loitering"] = dwell_time_scores(
            track_history, zones, self.frame_size, zone_frame_size,
            cfg["dwell_threshold_s"], cfg["score_sigmoid_scale_s"], cfg["stillness_radius_px"],
            global_timestamps,
        )

        cfg = self.classes_config["running"]["params"]
        zones = self.zone_config.zones_of_type("no_running") if self.zone_config else []
        scores["running"] = velocity_scores(
            track_history, zones, self.frame_size, zone_frame_size,
            cfg["velocity_threshold_bbox_heights_per_s"], cfg["min_trigger_duration_s"], global_timestamps,
        )

        cfg = self.classes_config["crowd_formation"]["params"]
        scores["crowd_formation"] = density_scores(
            frames, cfg["min_people"], cfg["cluster_radius_px"], cfg["sustained_duration_s"],
        )

        cfg = self.classes_config["falling"]["params"]
        scores["falling"] = falling_scores(
            track_history, cfg["aspect_ratio_flip_ratio"],
            cfg["post_event_stillness_window_s"], cfg["post_event_movement_px"],
            global_timestamps,
        )

        cfg = self.classes_config["abandoned_object"]["params"]
        scores["abandoned_object"] = abandoned_object_scores(
            frames, track_history, cfg["static_object_classes"], cfg["static_duration_s"],
            cfg["static_movement_px"], cfg["owner_proximity_px"], cfg["owner_separation_duration_s"],
            global_timestamps,
        )

        return scores
