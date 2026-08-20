"""perf_counter-based stage timing. time.time() quantises to ~15.6ms on this
box (time.get_clock_info('time').resolution), which would round a 9ms tracker
stage to 0 or 15.6ms -- perf_counter is exclusively used instead. First 10
frames are excluded as warmup (first inference call is 3-10x steady state).
"""
from __future__ import annotations

import statistics
import time
from collections import defaultdict

WARMUP_FRAMES = 10


class Stage:
    __slots__ = ("profiler", "name", "_t0")

    def __init__(self, profiler: "Profiler", name: str):
        self.profiler = profiler
        self.name = name

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.profiler.record(self.name, time.perf_counter() - self._t0)


class Profiler:
    def __init__(self):
        self._samples: dict[str, list[float]] = defaultdict(list)
        self._frame_count = 0
        self._frame_start_ts: list[float] = []

    def stage(self, name: str) -> Stage:
        return Stage(self, name)

    def record(self, name: str, seconds: float) -> None:
        self._samples[name].append(seconds)

    def frame_boundary(self) -> None:
        self._frame_count += 1
        self._frame_start_ts.append(time.perf_counter())

    @staticmethod
    def _p(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        idx = min(len(s) - 1, int(round(q * (len(s) - 1))))
        return s[idx]

    def report(self) -> dict:
        out: dict = {}
        for name, values in self._samples.items():
            usable = values[WARMUP_FRAMES:] if len(values) > WARMUP_FRAMES else values
            if not usable:
                continue
            out[f"{name}_ms_p50"] = round(self._p(usable, 0.50) * 1000, 3)
            out[f"{name}_ms_p95"] = round(self._p(usable, 0.95) * 1000, 3)

        ts = self._frame_start_ts[WARMUP_FRAMES:]
        if len(ts) >= 2:
            total_s = ts[-1] - ts[0]
            out["end_to_end_fps"] = round((len(ts) - 1) / total_s, 3) if total_s > 0 else 0.0
        if len(ts) >= 101:
            first100 = ts[:100]
            out["fps_first_100"] = round(99 / (first100[-1] - first100[0]), 3) if first100[-1] > first100[0] else 0.0
        if len(ts) >= 100:
            last100 = ts[-100:]
            span = last100[-1] - last100[0]
            out["fps_last_100"] = round(99 / span, 3) if span > 0 else 0.0
        out["warmup_frames_excluded"] = WARMUP_FRAMES
        return out
