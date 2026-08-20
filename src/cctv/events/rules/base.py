"""Shared primitives for zone rules: three-valued presence and the
stationarity test loitering.py and episodes.py both need.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from math import ceil, hypot
from statistics import median


class Presence(Enum):
    INSIDE = "inside"
    OUTSIDE = "outside"
    MISSING = "missing"


class Stationarity(Enum):
    STATIONARY = "stationary"
    MOVING = "moving"
    UNKNOWN = "unknown"


@dataclass
class StationaritySample:
    ts: float
    fx: float
    fy: float
    h: float
    interpolated: bool


def stationarity(
    window: deque[StationaritySample],
    *,
    stationary_window_seconds: float,
    stationary_radius: float,
    stationary_scale_ratio_max: float,
    fps: float,
    stride: int,
) -> Stationarity:
    """Max radius about the MEDIAN foot point over a time-indexed window,
    thresholded in bbox-height units, plus a scale-change gate that closes
    radial motion (walking toward/away from the camera = near-zero image
    displacement, which the radius test alone cannot see).
    """
    if not window:
        return Stationarity.UNKNOWN

    now = window[-1].ts
    while window and (now - window[0].ts) > stationary_window_seconds:
        window.popleft()

    min_n = max(3, ceil(0.5 * stationary_window_seconds * fps / max(1, stride)))
    if len(window) < min_n:
        return Stationarity.UNKNOWN

    fxs = [s.fx for s in window]
    fys = [s.fy for s in window]
    hs = [s.h for s in window]
    cx, cy = median(fxs), median(fys)
    radius_px = max(hypot(fx - cx, fy - cy) for fx, fy in zip(fxs, fys))
    med_h = median(hs)
    thresh_px = stationary_radius * med_h if med_h > 0 else float("inf")
    h_ratio = (max(hs) / min(hs)) if min(hs) > 0 else float("inf")

    if radius_px <= thresh_px and h_ratio <= stationary_scale_ratio_max:
        return Stationarity.STATIONARY
    return Stationarity.MOVING
