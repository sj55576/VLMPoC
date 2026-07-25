"""Pure geometry helpers used by pose and SOP conditions."""
from __future__ import annotations

import math

from .models import Keypoint


def sanitize_bbox(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float] | None:
    """Return a finite, ordered box or ``None`` when the input is unusable."""
    if len(bbox) != 4 or not all(math.isfinite(value) for value in bbox):
        return None
    x1, y1, x2, y2 = bbox
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    if x1 == x2 or y1 == y2:
        return None
    return x1, y1, x2, y2


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Return intersection over union for x1,y1,x2,y2 rectangles."""
    a, b = sanitize_bbox(a), sanitize_bbox(b)
    if a is None or b is None:
        return 0.0
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.dist(a, b)


def normalized_distance(a: tuple[float, float], b: tuple[float, float], width: float, height: float) -> float:
    return math.hypot((a[0]-b[0])/width, (a[1]-b[1])/height)


def calculate_joint_angle(point_a: Keypoint, point_b: Keypoint, point_c: Keypoint) -> float:
    """Return degrees at point_b, or 0 for a degenerate joint."""
    ba = (point_a.x - point_b.x, point_a.y - point_b.y)
    bc = (point_c.x - point_b.x, point_c.y - point_b.y)
    denom = math.hypot(*ba) * math.hypot(*bc)
    if denom == 0:
        return 0.0
    cosine = max(-1.0, min(1.0, (ba[0]*bc[0] + ba[1]*bc[1]) / denom))
    return math.degrees(math.acos(cosine))


def inside_region(point: tuple[float, float], region: dict[str, float]) -> bool:
    return region["x1"] <= point[0] <= region["x2"] and region["y1"] <= point[1] <= region["y2"]
