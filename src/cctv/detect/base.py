"""Detector protocol + registry. PersonTracker receives an instance and never
constructs one -- this is what makes --detector stub possible for offline tests.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Detector(Protocol):
    """A Detector wraps model.track()-style stateful tracking, not bare detection,
    because ByteTrack/BoT-SORT state must persist across calls within one source.
    """

    def begin_source(self, imgsz: int) -> None: ...

    def step(self, bgr: np.ndarray) -> tuple[list[dict], bool]:
        """Returns (detections, is_track).
        Each detection dict: {bbox_xyxy, conf, cls, track_id (or None)}.
        is_track is a per-frame flag: False means boxes were produced but the
        tracker did not assign ids this frame (all track_id will be None).
        """
        ...


DETECTOR_REGISTRY: dict[str, type] = {}


def register_detector(name: str):
    def deco(cls):
        DETECTOR_REGISTRY[name] = cls
        return cls
    return deco
