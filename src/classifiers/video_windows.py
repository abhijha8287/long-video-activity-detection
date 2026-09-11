"""Sliding-window frame sampler for the appearance branch."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from src.detection.detector import get_video_info


def iter_windows(
    video_path: str | Path,
    window_size_s: float,
    stride_s: float,
    frames_per_window: int = 3,
) -> Iterator[tuple[float, float, list[np.ndarray]]]:
    """Yields (window_start_s, window_end_s, frames) for each sliding window.
    Samples `frames_per_window` evenly-spaced frames per window rather than one,
    to reduce single-frame noise — this is still static-frame sampling with no
    motion modeling (CLIP's documented weakness), just fewer unlucky single frames.
    """
    info = get_video_info(video_path)
    cap = cv2.VideoCapture(str(video_path))
    try:
        window_start = 0.0
        while window_start < info.duration_s:
            window_end = min(window_start + window_size_s, info.duration_s)
            span = window_end - window_start
            offsets = [
                window_start + span * (k + 0.5) / frames_per_window for k in range(frames_per_window)
            ]
            frames = []
            for ts in offsets:
                frame_idx = min(int(round(ts * info.fps)), info.frame_count - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_idx, 0))
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            if frames:
                yield window_start, window_end, frames
            window_start += stride_s
    finally:
        cap.release()
