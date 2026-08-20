"""Deterministic per-identity color from global_id, stable across processes.
Golden-ratio hue stepping keeps consecutive ids visually distinct even though
they are numerically adjacent.
"""
from __future__ import annotations

import colorsys

_GOLDEN_RATIO_CONJUGATE = 0.618033988749895


def color_for_id(gid: int) -> tuple[int, int, int]:
    """Returns a BGR tuple in [0, 255]. Not using Python's salted hash() --
    it must be identical across processes (dashboard vs pipeline)."""
    hue = ((gid * _GOLDEN_RATIO_CONJUGATE) % 1.0)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))
