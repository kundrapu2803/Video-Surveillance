"""EMA box smoothing in (cx, cy, w, h) space, written to smooth_bbox only --
ANNOTATION ONLY. The event layer and evaluator always use the raw bbox_xyxy.
"""
from __future__ import annotations

import dataclasses

from cctv.geometry import xyxy_to_cxcywh
from cctv.schema import TrackRecord

TRAIL_TTL_FRAMES = 90


class EmaSmoother:
    def __init__(self, alpha: float = 0.6):
        self.alpha = alpha
        self._state: dict[int, tuple[float, float, float, float]] = {}
        self._last_seen: dict[int, int] = {}

    def smooth(self, records: list[TrackRecord]) -> list[TrackRecord]:
        out = []
        for rec in records:
            if rec.global_id is None:
                out.append(rec)
                continue
            cxcywh = xyxy_to_cxcywh(rec.bbox_xyxy)
            prev = self._state.get(rec.global_id)
            if prev is None:
                sm = cxcywh
            else:
                a = self.alpha
                sm = tuple(a * p + (1 - a) * c for p, c in zip(prev, cxcywh))
            self._state[rec.global_id] = sm
            self._last_seen[rec.global_id] = rec.frame_idx

            cx, cy, w, h = sm
            smooth_bbox = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            out.append(dataclasses.replace(rec, smooth_bbox=smooth_bbox))

        self._evict(records[0].frame_idx if records else None)
        return out

    def _evict(self, frame_idx: int | None) -> None:
        if frame_idx is None:
            return
        stale = [gid for gid, last in self._last_seen.items() if frame_idx - last > TRAIL_TTL_FRAMES]
        for gid in stale:
            self._state.pop(gid, None)
            self._last_seen.pop(gid, None)
