"""One debounced ZoneEpisode per (global_id, zone_id). Shared infrastructure
-- hysteresis and dropout handling are written and tested once here, and both
rules (intrusion, loitering) just ask an episode "has enter_seconds elapsed"
/ "is dwell+stationarity past threshold".
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from cctv.events.rules.base import Presence, Stationarity, StationaritySample, stationarity
from cctv.events.rules.intrusion import should_confirm_entry
from cctv.events.rules.loitering import should_fire as should_fire_loitering


@dataclass
class ZoneEpisode:
    zone_id: str
    global_id: int
    state: str = "entering"  # entering | active | exiting
    inside_run_s: float = 0.0
    outside_run_s: float = 0.0
    dwell_s: float = 0.0
    last_update_ts: float = 0.0
    last_seen_ts: float = 0.0
    entry_ts: Optional[float] = None
    entry_frame_idx: Optional[int] = None
    confirmed: bool = False
    loiter_fired: bool = False
    stationary_window: deque = field(default_factory=deque)
    support_confs: list = field(default_factory=list)
    last_bbox_xyxy: Optional[tuple] = None
    last_foot_point: Optional[tuple] = None
    observed_radius: Optional[float] = None


class ZoneOccupancyTracker:
    def __init__(self, fps: float, stride: int = 1):
        self.fps = fps
        self.stride = stride
        self.episodes: dict[tuple[int, str], ZoneEpisode] = {}

    def update(
        self, *, global_id: int, zone_id: str, params: dict, frame_idx: int, ts: float,
        presence: Presence, foot_point: tuple[float, float], bbox_xyxy: tuple,
        bbox_h: float, conf: float, interpolated: bool,
    ) -> list[dict]:
        key = (global_id, zone_id)
        ep = self.episodes.get(key)
        drafts: list[dict] = []

        if ep is None:
            if presence != Presence.INSIDE:
                return drafts
            ep = ZoneEpisode(zone_id=zone_id, global_id=global_id, last_update_ts=ts, last_seen_ts=ts)
            self.episodes[key] = ep

        dt = max(0.0, min(ts - ep.last_update_ts, 2.0 / self.fps))
        ep.last_update_ts = ts
        if presence != Presence.MISSING:
            ep.last_seen_ts = ts
        if bbox_xyxy is not None:  # presence == MISSING carries no real box -- never overwrite with it
            ep.last_bbox_xyxy = bbox_xyxy
            ep.last_foot_point = foot_point

        intrusion_params = params.get("intrusion", {})
        loiter_params = params.get("loitering", {})

        if ep.state == "entering":
            if presence == Presence.INSIDE:
                ep.inside_run_s += dt
                ep.dwell_s += dt
                ep.support_confs.append(conf)
                if not interpolated:
                    ep.stationary_window.append(
                        StationaritySample(ts, foot_point[0], foot_point[1], bbox_h, interpolated)
                    )
                # Hysteresis confirmation is presence-stability debouncing, independent of
                # whether the intrusion RULE happens to be enabled for this zone -- a
                # monitored zone (loitering only, intrusion disabled) must still be able
                # to reach "active" state, or loitering could never fire either.
                if not ep.confirmed and should_confirm_entry(ep.inside_run_s, intrusion_params):
                    ep.confirmed = True
                    ep.entry_ts = ts
                    ep.entry_frame_idx = frame_idx
                    ep.state = "active"
                    drafts.append(self._draft(ep, "enter_confirmed", frame_idx, ts, params))
            elif presence == Presence.MISSING:
                if (ts - ep.last_seen_ts) <= loiter_params.get("dwell_grace_seconds", 1.5):
                    pass  # hold, jitter tolerance
                else:
                    del self.episodes[key]
            else:  # OUTSIDE
                del self.episodes[key]

        elif ep.state == "active":
            if presence == Presence.INSIDE:
                ep.dwell_s += dt
                ep.outside_run_s = 0.0
                ep.support_confs.append(conf)
                if not interpolated:
                    ep.stationary_window.append(
                        StationaritySample(ts, foot_point[0], foot_point[1], bbox_h, interpolated)
                    )
                if loiter_params.get("enabled", False) and not ep.loiter_fired:
                    stat = stationarity(
                        ep.stationary_window,
                        stationary_window_seconds=loiter_params.get("stationary_window_seconds", 4.0),
                        stationary_radius=loiter_params.get("stationary_radius", 0.5),
                        stationary_scale_ratio_max=loiter_params.get("stationary_scale_ratio_max", 1.25),
                        fps=self.fps, stride=self.stride,
                    )
                    if should_fire_loitering(ep.dwell_s, stat, loiter_params):
                        ep.loiter_fired = True
                        if ep.stationary_window:
                            fxs = [s.fx for s in ep.stationary_window]
                            fys = [s.fy for s in ep.stationary_window]
                            cx = sorted(fxs)[len(fxs) // 2]
                            cy = sorted(fys)[len(fys) // 2]
                            ep.observed_radius = max(((fx - cx) ** 2 + (fy - cy) ** 2) ** 0.5 for fx, fy in zip(fxs, fys))
                        drafts.append(self._draft(ep, "loiter_fired", frame_idx, ts, params))
            elif presence == Presence.MISSING:
                gap = ts - ep.last_seen_ts
                if gap <= loiter_params.get("dwell_grace_seconds", 1.5):
                    pass  # dwell bridges, does not reset
                elif gap <= loiter_params.get("track_timeout_seconds", 3.0):
                    pass  # dwell freezes; stationarity window ages out on its own
                else:
                    drafts.append(self._draft(ep, "closed", frame_idx, ts, params))
                    del self.episodes[key]
            else:  # OUTSIDE
                ep.outside_run_s = 0.0
                ep.state = "exiting"

        elif ep.state == "exiting":
            if presence == Presence.INSIDE:
                ep.state = "active"
                ep.dwell_s += dt
                ep.outside_run_s = 0.0
            elif presence == Presence.OUTSIDE:
                ep.outside_run_s += dt
                if ep.outside_run_s >= intrusion_params.get("exit_seconds", 0.60):
                    drafts.append(self._draft(ep, "closed", frame_idx, ts, params))
                    del self.episodes[key]
            else:  # MISSING while exiting: keep the exit clock running
                ep.outside_run_s += dt
                if ep.outside_run_s >= intrusion_params.get("exit_seconds", 0.60):
                    drafts.append(self._draft(ep, "closed", frame_idx, ts, params))
                    del self.episodes[key]

        return drafts

    def flush(self, frame_idx: int, ts: float, params_by_zone: dict) -> list[dict]:
        """Force-close every still-open episode at end of stream, marked truncated."""
        drafts = []
        for key, ep in list(self.episodes.items()):
            if ep.confirmed:
                params = params_by_zone.get(ep.zone_id, {})
                drafts.append(self._draft(ep, "closed", frame_idx, ts, params, truncated=True))
            del self.episodes[key]
        return drafts

    @staticmethod
    def _draft(ep: ZoneEpisode, kind: str, frame_idx: int, ts: float, params: dict, truncated: bool = False) -> dict:
        return {
            "kind": kind,
            "zone_id": ep.zone_id,
            "global_id": ep.global_id,
            "frame_idx": frame_idx,
            "ts": ts,
            "entry_ts": ep.entry_ts,
            "entry_frame_idx": ep.entry_frame_idx,
            "dwell_s": ep.dwell_s,
            "bbox_xyxy": ep.last_bbox_xyxy,
            "foot_point": ep.last_foot_point,
            "confs": list(ep.support_confs),
            "observed_radius": ep.observed_radius,
            "params": params,
            "truncated": truncated,
            "confirmed": ep.confirmed,
            "loiter_fired": ep.loiter_fired,
        }
