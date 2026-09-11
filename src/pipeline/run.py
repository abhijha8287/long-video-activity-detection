"""Orchestrates the full pipeline: video -> geometry branch + appearance branch
-> shared temporal post-processing -> final segments. See
docs/designs/activity-detection-baseline.md for the architecture rationale.

Split into `compute_class_series` (the expensive part — detection, tracking,
CLIP inference) and `build_segments_from_series` (cheap post-processing) so
threshold/smoothing experiments (scripts/run_experiments.py) can sweep
post-processing parameters without re-running detection or CLIP for every
setting — only window/stride changes require a full re-run, since those change
what CLIP actually sees.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from src.classifiers.clip_zeroshot import CLIPZeroShotClassifier
from src.classifiers.video_windows import iter_windows
from src.detection.detector import TrackedDetector, get_video_info
from src.pipeline.segment_builder import build_all_segments
from src.rules.engine import RuleEngine
from src.rules.geometry import ZoneConfig, load_zone_config
from src.utils import load_classes_config, set_seed

logger = logging.getLogger(__name__)

GEOMETRY_CLASSES = (
    "intrusion",
    "loitering",
    "running",
    "crowd_formation",
    "falling",
    "abandoned_object",
)
APPEARANCE_CLASSES = ("fire_smoke", "fighting", "vandalism", "theft")

ZONE_DEPENDENT_CLASSES = {"intrusion", "running"}  # loitering can run zone-free too, but is weaker without one


def compute_class_series(
    video_path: str | Path,
    zones_name: str | None = None,
    classes_config_path: str = "configs/classes.yaml",
    zones_dir: str = "configs/zones",
    sample_fps: float = 8.0,
    window_size_s: float = 3.0,
    stride_s: float = 1.5,
    device: str = "auto",
    weights: str = "yolo11n.pt",
    detector: TrackedDetector | None = None,
    clip_classifier: CLIPZeroShotClassifier | None = None,
) -> dict[str, Any]:
    """Runs both branches once and returns everything needed to build segments
    (and to re-build them with different post-processing settings).

    `detector`/`clip_classifier` let a caller reuse already-loaded models across
    multiple videos (e.g. the Streamlit app) instead of reloading YOLO/CLIP
    weights on every call — CLI callers leave these None and get fresh instances.
    """
    classes_config, post_processing_cfg = load_classes_config(classes_config_path)
    video_path = Path(video_path)
    info = get_video_info(video_path)
    logger.info("Video: %s (%.1fs, %dx%d, %.1f fps)", video_path.name, info.duration_s, info.width, info.height, info.fps)

    zone_config: ZoneConfig | None = None
    if zones_name:
        zone_config = load_zone_config(Path(zones_dir) / f"{zones_name}.yaml")
    else:
        skipped = [c for c in ("intrusion", "running") if c in classes_config]
        if skipped:
            logger.warning(
                "No --zones given: %s will be reported as not applicable for this run.",
                ", ".join(classes_config[c]["display_name"] for c in skipped),
            )

    t0 = time.time()
    if detector is None:
        detector = TrackedDetector(weights=weights, device=device)
    frames = list(detector.iter_tracks(video_path, sample_fps=sample_fps))
    engine = RuleEngine(classes_config, (info.width, info.height), zone_config)
    geometry_scores = engine.compute([(ts, tracks) for ts, _, _, tracks in frames])
    geometry_interval = 1.0 / sample_fps
    if not zone_config:
        for c in ZONE_DEPENDENT_CLASSES:
            geometry_scores[c] = []
    t_geometry = time.time()
    logger.info("Geometry branch done in %.1fs", t_geometry - t0)

    if clip_classifier is None:
        clip_classifier = CLIPZeroShotClassifier(classes_config, device=device)
    appearance_scores: dict[str, list[tuple[float, float]]] = {c: [] for c in clip_classifier.class_keys}
    for window_start, window_end, window_frames in iter_windows(video_path, window_size_s, stride_s):
        window_mid = (window_start + window_end) / 2.0
        probs = clip_classifier.score_window(window_frames)
        for class_key, prob in probs.items():
            appearance_scores[class_key].append((window_mid, prob))
    t_appearance = time.time()
    logger.info("Appearance branch done in %.1fs", t_appearance - t_geometry)

    class_series: dict[str, list[tuple[float, float]]] = {**geometry_scores, **appearance_scores}
    class_intervals = {c: geometry_interval for c in GEOMETRY_CLASSES}
    class_intervals.update({c: stride_s for c in APPEARANCE_CLASSES})

    return {
        "class_series": class_series,
        "class_intervals": class_intervals,
        "classes_config": classes_config,
        "post_processing_cfg": post_processing_cfg,
        "video_info": info,
        "compute_time_s": time.time() - t0,
    }


def build_segments_from_series(
    class_series: dict[str, list[tuple[float, float]]],
    class_intervals: dict[str, float],
    classes_config: dict[str, Any],
    post_processing_cfg: dict[str, Any],
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    cfg = {**post_processing_cfg, "confidence_threshold": threshold} if threshold is not None else post_processing_cfg
    class_display_names = {
        k: v["display_name"] for k, v in classes_config.items() if v.get("branch") != "appearance_background"
    }
    return build_all_segments(class_series, class_intervals, class_display_names, cfg)


def run_pipeline(
    video_path: str | Path,
    output_path: str | Path | None = None,
    zones_name: str | None = None,
    classes_config_path: str = "configs/classes.yaml",
    zones_dir: str = "configs/zones",
    sample_fps: float = 8.0,
    window_size_s: float = 3.0,
    stride_s: float = 1.5,
    threshold: float | None = None,
    device: str = "auto",
    weights: str = "yolo11n.pt",
    detector: TrackedDetector | None = None,
    clip_classifier: CLIPZeroShotClassifier | None = None,
) -> dict[str, Any]:
    set_seed()
    raw = compute_class_series(
        video_path, zones_name, classes_config_path, zones_dir,
        sample_fps, window_size_s, stride_s, device, weights,
        detector, clip_classifier,
    )
    segments = build_segments_from_series(
        raw["class_series"], raw["class_intervals"], raw["classes_config"], raw["post_processing_cfg"], threshold,
    )

    info = raw["video_info"]
    # The last window/frame's interval-based end-time estimate can slightly
    # overshoot the video's actual duration (e.g. a short clip's final CLIP
    # window is truncated but still gets a full stride_s added) — clamp so a
    # segment's end time never exceeds the video it came from.
    for seg in segments:
        if seg["end"] > info.duration_s:
            seg["end"] = round(info.duration_s, 2)
    total_time = raw["compute_time_s"]
    logger.info(
        "Pipeline complete in %.1fs for a %.1fs video (%.2fx real-time). %d segments found.",
        total_time, info.duration_s, total_time / max(info.duration_s, 1e-6), len(segments),
    )

    result = {"video": str(video_path), "segments": segments}
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Predictions written to %s", output_path)

    return result
