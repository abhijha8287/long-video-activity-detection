"""Pixel-geometry primitives used by the rule engine: point-in-polygon, zone
loading, and distance helpers. No ML here — deterministic geometry only."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from src.utils import load_yaml


@dataclass(frozen=True)
class Zone:
    name: str
    zone_type: str  # "restricted" | "loitering" | "no_running"
    polygon: list[tuple[float, float]]


@dataclass(frozen=True)
class ZoneConfig:
    video_ref: str
    frame_size: tuple[int, int]  # width, height the polygons were authored against
    zones: list[Zone]

    def zones_of_type(self, zone_type: str) -> list[Zone]:
        return [z for z in self.zones if z.zone_type == zone_type]


def load_zone_config(path: str | Path) -> ZoneConfig:
    raw = load_yaml(path)
    zones = [
        Zone(z["name"], z["type"], [tuple(pt) for pt in z["polygon"]])
        for z in raw.get("zones", [])
    ]
    fw, fh = raw.get("frame_size", [0, 0])
    return ZoneConfig(raw.get("video", ""), (fw, fh), zones)


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    """Standard ray-casting point-in-polygon test."""
    x, y = point
    n = len(polygon)
    inside = False
    x1, y1 = polygon[0]
    for i in range(1, n + 1):
        x2, y2 = polygon[i % n]
        if y > min(y1, y2):
            if y <= max(y1, y2):
                if x <= max(x1, x2):
                    if y1 != y2:
                        x_intersect = (y - y1) * (x2 - x1) / (y2 - y1) + x1
                    else:
                        x_intersect = x1
                    if x1 == x2 or x <= x_intersect:
                        inside = not inside
        x1, y1 = x2, y2
    return inside


def scale_polygon(polygon: list[tuple[float, float]], from_size: tuple[int, int], to_size: tuple[int, int]) -> list[tuple[float, float]]:
    """Rescale a zone polygon authored on one frame size to another (letting the
    same zone config be reused if a video is resized/re-encoded)."""
    fw, fh = from_size
    tw, th = to_size
    if fw == 0 or fh == 0 or (fw, fh) == (tw, th):
        return polygon
    sx, sy = tw / fw, th / fh
    return [(x * sx, y * sy) for x, y in polygon]


def euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
