"""Applied in this order (semantic filters first, safety valve last):
1. min duration      -- drop intervals shorter than min_event_duration_s
2. cooldown          -- suppress repeat triggers on the same (gid, zone, type)
3. gap merge          -- a new episode within merge_gap_seconds of the previous
                         close extends it instead of opening a new event
4. token bucket        -- hard per-minute ceiling, global and per-zone
"""
from __future__ import annotations

import uuid
from dataclasses import replace
from typing import Optional


class RateLimiter:
    def __init__(self, per_minute: int, per_minute_per_zone: int):
        self.per_minute = per_minute
        self.per_minute_per_zone = per_minute_per_zone
        self._global_bucket: list[float] = []
        self._zone_buckets: dict[str, list[float]] = {}

    def allow(self, ts: float, zone_id: str) -> bool:
        self._global_bucket = [t for t in self._global_bucket if ts - t < 60.0]
        zb = self._zone_buckets.setdefault(zone_id, [])
        zb[:] = [t for t in zb if ts - t < 60.0]
        if len(self._global_bucket) >= self.per_minute or len(zb) >= self.per_minute_per_zone:
            return False
        self._global_bucket.append(ts)
        zb.append(ts)
        return True


class EventDeduper:
    """Owns cooldown/gap-merge/rate-limit state for one run. Thresholds are
    read per-call from each event's own resolved rule_params, since different
    zones/rules can carry different values -- only the global rate-limit
    bucket is shared, keyed off whichever zone last fired (file defaults
    normally make max_events_per_minute uniform across zones anyway).
    """

    def __init__(self, default_max_per_minute: int = 60, default_max_per_minute_per_zone: int = 30):
        self._cooldown_until: dict[tuple, float] = {}
        self._last_interval: dict[tuple, object] = {}  # key -> Event (interval) eligible for merge
        self._rate_limiter = RateLimiter(default_max_per_minute, default_max_per_minute_per_zone)

    def _key(self, source_id, global_id, zone_id, event_type):
        return (source_id, global_id, zone_id, event_type)

    def gate_alert(self, ts: float, source_id: str, global_id: int, zone_id: str,
                    event_type: str, cooldown_seconds: float) -> bool:
        key = self._key(source_id, global_id, zone_id, event_type)
        until = self._cooldown_until.get(key, -1.0)
        if ts < until:
            return False
        self._cooldown_until[key] = ts + cooldown_seconds
        return True

    def gate_interval(self, ev, source_id: str, min_event_duration_s: float,
                       merge_gap_seconds: float, max_events_per_minute: int,
                       max_events_per_minute_per_zone: int) -> Optional[object]:
        """Returns the Event that should actually be written (possibly a merged
        version of a previous still-recent interval, reusing its event_id so
        the writer overwrites the earlier row instead of appending a second
        one), or None if the interval is too short or rate-limited.
        """
        duration = ev.duration_s or 0.0
        if duration < min_event_duration_s:
            return None

        key = self._key(source_id, ev.global_id, ev.zone_id, ev.event_type)
        prev = self._last_interval.get(key)
        if prev is not None and (ev.timestamp_s - prev.end_timestamp_s) <= merge_gap_seconds:
            merged = replace(
                ev,
                event_id=prev.event_id,
                timestamp_s=prev.timestamp_s,
                timestamp_hms=prev.timestamp_hms,
                frame_idx=prev.frame_idx,
                frame_number=prev.frame_number,
                duration_s=(ev.end_timestamp_s - prev.timestamp_s),
                n_support_frames=prev.n_support_frames + ev.n_support_frames,
                merge_count=prev.merge_count + 1,
            )
            self._last_interval[key] = merged
            return merged

        self._rate_limiter.per_minute = max_events_per_minute
        self._rate_limiter.per_minute_per_zone = max_events_per_minute_per_zone
        if not self._rate_limiter.allow(ev.timestamp_s, ev.zone_id):
            return None

        self._last_interval[key] = ev
        return ev


def new_event_id() -> str:
    return uuid.uuid4().hex[:12]
