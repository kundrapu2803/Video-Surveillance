"""PersonTracker: the pipeline-facing wrapper around an injected Detector.
Never constructs a Detector itself -- receives one, which is what makes
--detector stub swap in for tests without touching this module.
"""
from __future__ import annotations

import numpy as np

from cctv.detect.base import Detector


class PersonTracker:
    def __init__(self, detector: Detector, *, source_id: str, imgsz: int):
        self.detector = detector
        self.source_id = source_id
        self.imgsz = imgsz
        self.frames_seen = 0
        self.frames_without_ids = 0
        self.local_ids_seen: set[int] = set()

    def begin_source(self) -> None:
        self.detector.begin_source(self.imgsz)
        self.frames_seen = 0
        self.frames_without_ids = 0
        self.local_ids_seen = set()

    def step(self, bgr: np.ndarray) -> tuple[list[dict], bool]:
        dets, is_track = self.detector.step(bgr)
        self.frames_seen += 1
        if dets and not is_track:
            self.frames_without_ids += 1
        for d in dets:
            if d.get("track_id") is not None:
                self.local_ids_seen.add(d["track_id"])
        return dets, is_track

    @property
    def frames_without_ids_frac(self) -> float:
        return self.frames_without_ids / self.frames_seen if self.frames_seen else 0.0
