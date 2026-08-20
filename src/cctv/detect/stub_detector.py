"""Scripted-boxes detector. No torch import anywhere in this module -- the
entire offline test suite (and CI) depends on --detector stub working without
a model download.
"""
from __future__ import annotations

import numpy as np

from cctv.detect.base import register_detector


@register_detector("stub")
class StubDetector:
    """Deterministic scripted detections, keyed by frame_idx.

    script: dict[int, list[dict]] mapping frame_idx -> list of
    {bbox_xyxy, conf, track_id} entries. Any frame not in the script yields
    zero detections. cls is always 0.
    """

    def __init__(self, script: dict[int, list[dict]] | None = None, **_kwargs):
        self.script = script or {}
        self._frame_idx = -1

    def begin_source(self, imgsz: int) -> None:
        self._frame_idx = -1

    def step(self, bgr: np.ndarray) -> tuple[list[dict], bool]:
        self._frame_idx += 1
        entries = self.script.get(self._frame_idx, [])
        dets = [
            {
                "bbox_xyxy": tuple(e["bbox_xyxy"]),
                "conf": float(e.get("conf", 0.9)),
                "cls": 0,
                "track_id": e.get("track_id"),
            }
            for e in entries
        ]
        is_track = True
        return dets, is_track
