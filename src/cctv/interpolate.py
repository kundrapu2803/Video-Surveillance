"""--stride keyframe interpolation. 'linear' holds one keyframe of lookahead
and interpolates bbox_xyxy for global_ids present in BOTH keyframes -- never
extrapolates, which is how phantom boxes get born. Interpolated records feed
dwell but must never be a loitering trigger frame (see events/episodes.py).
"""
from __future__ import annotations

import dataclasses
from typing import Iterator

from cctv.schema import TrackRecord


class LinearInterpolator:
    def __init__(self, stride: int):
        self.stride = max(1, stride)
        self._prev_frame_idx: int | None = None
        self._prev_ts: float | None = None
        self._prev_by_gid: dict[int, TrackRecord] = {}
        self._pending: tuple[int, float, list[TrackRecord]] | None = None

    def feed(self, frame_idx: int, ts: float, records: list[TrackRecord]) -> Iterator[tuple[int, float, list[TrackRecord]]]:
        if self.stride == 1:
            yield frame_idx, ts, records
            return

        cur_by_gid = {r.global_id: r for r in records if r.global_id is not None}

        if self._pending is not None:
            prev_idx, prev_ts, prev_records = self._pending
            n_gaps = frame_idx - prev_idx
            for g in range(1, n_gaps):
                gap_idx = prev_idx + g
                gap_ts = prev_ts + (ts - prev_ts) * (g / n_gaps)
                interp_records = []
                for gid, prev_rec in self._prev_by_gid.items():
                    cur_rec = cur_by_gid.get(gid)
                    if cur_rec is None:
                        continue  # never extrapolate
                    frac = g / n_gaps
                    bbox = tuple(
                        p + (c - p) * frac for p, c in zip(prev_rec.bbox_xyxy, cur_rec.bbox_xyxy)
                    )
                    interp_records.append(dataclasses.replace(
                        prev_rec, frame_idx=gap_idx, timestamp_s=gap_ts, bbox_xyxy=bbox,
                        smooth_bbox=None, interpolated=True,
                    ))
                yield gap_idx, gap_ts, interp_records

        yield frame_idx, ts, records
        self._pending = (frame_idx, ts, records)
        self._prev_by_gid = cur_by_gid
