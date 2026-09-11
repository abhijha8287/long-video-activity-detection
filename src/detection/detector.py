"""YOLO11n detection + ByteTrack tracking, sampled at a target fps to stay inside
the runtime budget on CPU/MPS hardware (see docs/designs/activity-detection-baseline.md).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from src.utils import resolve_device


@dataclass(frozen=True)
class Track:
    track_id: int
    cls_name: str
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    conf: float

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float


def get_video_info(video_path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration_s = frame_count / fps if fps > 0 else 0.0
    return VideoInfo(width, height, fps, frame_count, duration_s)


class TrackedDetector:
    """Wraps ultralytics YOLO detection + its built-in ByteTrack tracker.

    Frames are sampled at `sample_fps` (default 5.0, see the runtime-budget note in
    the design doc) rather than every native frame, trading some track-ID continuity
    for CPU-friendly throughput. Person-class detections are always kept; object
    classes relevant to Abandoned Object (backpack/suitcase/handbag) are kept too.
    """

    RELEVANT_CLASSES = {"person", "backpack", "suitcase", "handbag"}

    def __init__(self, weights: str = "yolo11n.pt", device: str = "auto", conf: float = 0.25):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device = resolve_device(device)
        self.conf = conf

    def reset(self) -> None:
        """Clear tracker state — call before processing a new video."""
        self.model.predictor = None

    def iter_tracks(
        self, video_path: str | Path, sample_fps: float = 5.0
    ) -> Iterator[tuple[float, int, np.ndarray, list[Track]]]:
        """Yields (timestamp_s, frame_idx, frame_bgr, tracks) for sampled frames."""
        self.reset()
        info = get_video_info(video_path)
        frame_interval = max(1, round(info.fps / sample_fps))

        cap = cv2.VideoCapture(str(video_path))
        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_interval == 0:
                    timestamp = frame_idx / info.fps
                    result = self.model.track(
                        frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        device=self.device,
                        conf=self.conf,
                        classes=None,
                        verbose=False,
                    )[0]
                    tracks = self._parse_result(result)
                    yield timestamp, frame_idx, frame, tracks
                frame_idx += 1
        finally:
            cap.release()

    def _parse_result(self, result) -> list[Track]:
        tracks: list[Track] = []
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return tracks
        names = result.names
        for i in range(len(boxes)):
            cls_name = names[int(boxes.cls[i].item())]
            if cls_name not in self.RELEVANT_CLASSES:
                continue
            track_id = int(boxes.id[i].item())
            conf = float(boxes.conf[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            tracks.append(Track(track_id, cls_name, (x1, y1, x2, y2), conf))
        return tracks
