"""EventEngine.update(packet) is the single entry point: advances all zone
state by exactly one frame and returns the Event objects (alerts and/or
interval closes) that should be written this call. frame_overlay_state() is
the renderer's ONLY interface to event logic, so the video and the log can
never disagree about what fired.
"""
from __future__ import annotations

from datetime import timedelta

from cctv.events.dedup import EventDeduper, new_event_id
from cctv.events.episodes import Presence, ZoneOccupancyTracker
from cctv.geometry import xyxy_to_tlwh_0based
from cctv.io.zones import ZoneSet, foot_point, zone_membership
from cctv.schema import EVENT_SCHEMA_VERSION, Event


class FrameOrderError(Exception):
    pass


class TrackerNotProducingIdsError(Exception):
    pass


def _hms(ts: float) -> str:
    td = timedelta(seconds=max(0.0, ts))
    total_ms = int(round(td.total_seconds() * 1000))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _norm_bbox(bbox_xyxy, w, h):
    x1, y1, x2, y2 = bbox_xyxy
    return (x1 / w, y1 / h, x2 / w, y2 / h)


class EventEngine:
    def __init__(self, zone_set: ZoneSet, *, run_id: str, source_id: str, camera_id: str,
                 frame_width: int, frame_height: int, fps: float, stride: int = 1,
                 strict_id_rate: bool = True):
        self.zone_set = zone_set
        self.zones_by_id = {z.id: z for z in zone_set.zones}
        self.occupancy = ZoneOccupancyTracker(fps, stride)
        self.run_id = run_id
        self.source_id = source_id
        self.camera_id = camera_id
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.sequence = 0
        self._last_frame_idx = -1
        self._deduper = EventDeduper()
        self._last_zone_state: dict[str, dict] = {}
        self._never_triggered = {z.id for z in zone_set.zones}
        self._frames_seen = 0
        self._frames_with_gid = 0
        self.strict_id_rate = strict_id_rate
        self.low_id_rate_warning: str | None = None

    def update(self, *, frame_idx: int, ts: float, records: list, interpolated_ok: bool = True) -> list[Event]:
        if frame_idx < self._last_frame_idx:
            raise FrameOrderError(f"non-increasing frame_idx: {frame_idx} after {self._last_frame_idx}")
        self._last_frame_idx = frame_idx

        self._frames_seen += 1
        present_gids = {r.global_id for r in records if r.global_id is not None}
        if present_gids:
            self._frames_with_gid += 1
        if self._frames_seen == 100 and self._frames_with_gid / self._frames_seen < 0.80:
            msg = (
                f"{self._frames_with_gid}/{self._frames_seen} of the first 100 detection-bearing "
                "frames had a global_id. On a source with ground truth this almost always means "
                "the tracker is misconfigured (check --tracker / conf thresholds), not the zone "
                "config -- so it's a hard error there. Without ground truth to verify against "
                "(e.g. exploratory footage), sparse/uncertain detections legitimately failing to "
                "confirm into a track is also a plausible real explanation, so this is a warning."
            )
            if self.strict_id_rate:
                raise TrackerNotProducingIdsError(msg)
            self.low_id_rate_warning = msg

        by_gid = {r.global_id: r for r in records if r.global_id is not None}
        events: list[Event] = []
        overlay: dict[str, dict] = {}

        for zone in self.zone_set.zones:
            if not zone.enabled:
                continue
            min_conf = zone.params.get("intrusion", {}).get("min_confidence",
                        zone.params.get("loitering", {}).get("min_confidence", 0.35))

            inside_gids = set()
            for gid, rec in by_gid.items():
                if rec.conf < min_conf:
                    continue
                fp = foot_point(rec.bbox_xyxy)
                if zone_membership(fp, zone):
                    inside_gids.add(gid)

            watch_gids = inside_gids | {gid for (gid, zid) in self.occupancy.episodes if zid == zone.id}

            for gid in watch_gids:
                rec = by_gid.get(gid)
                if rec is not None:
                    presence = Presence.INSIDE if gid in inside_gids else Presence.OUTSIDE
                    bbox_h = rec.bbox_xyxy[3] - rec.bbox_xyxy[1]
                    fp = foot_point(rec.bbox_xyxy)
                    conf = rec.conf
                    bbox = rec.bbox_xyxy
                    interpolated = rec.interpolated
                else:
                    presence = Presence.MISSING
                    bbox_h, fp, conf, bbox, interpolated = 0.0, (0.0, 0.0), 0.0, None, False

                drafts = self.occupancy.update(
                    global_id=gid, zone_id=zone.id, params=zone.params, frame_idx=frame_idx, ts=ts,
                    presence=presence, foot_point=fp, bbox_xyxy=bbox, bbox_h=bbox_h, conf=conf,
                    interpolated=interpolated,
                )
                for draft in drafts:
                    events.extend(self._events_from_draft(zone, draft))

            if inside_gids:
                self._never_triggered.discard(zone.id)

            active_types = set()
            for gid in inside_gids:
                ep = self.occupancy.episodes.get((gid, zone.id))
                if ep is None:
                    continue
                if ep.confirmed and zone.params.get("intrusion", {}).get("enabled", True):
                    active_types.add("zone_intrusion")
                if ep.loiter_fired and zone.params.get("loitering", {}).get("enabled", False):
                    active_types.add("loitering")

            overlay[zone.id] = {
                "polygon_px": zone.polygon_px,
                "color": zone.color,
                "name": zone.name,
                "occupancy": len(inside_gids),
                "breached": bool(inside_gids),
                "member_gids": sorted(inside_gids),
                "active_event_types": active_types,
                "intrusion_enabled": zone.params.get("intrusion", {}).get("enabled", True),
                "loitering_enabled": zone.params.get("loitering", {}).get("enabled", False),
            }

        self._last_zone_state = overlay
        return events

    def _events_from_draft(self, zone, draft: dict) -> list[Event]:
        out: list[Event] = []
        gid = draft["global_id"]
        bbox = draft["bbox_xyxy"]
        fp = draft["foot_point"]
        confs = draft["confs"] or [0.0]
        conf_median = sorted(confs)[len(confs) // 2]
        conf_min = min(confs)
        frame_idx = draft["frame_idx"]
        ts = draft["ts"]
        rule_params = draft["params"]
        priority = zone.priority

        def _base_kwargs(event_type: str, status: str, record_kind: str) -> dict:
            bbox_tlwh = xyxy_to_tlwh_0based(bbox) if bbox else (0.0, 0.0, 0.0, 0.0)
            bbox_norm = _norm_bbox(bbox, self.frame_width, self.frame_height) if bbox else (0.0, 0.0, 0.0, 0.0)
            return dict(
                schema_version=EVENT_SCHEMA_VERSION,
                record_kind=record_kind,
                sequence=self._next_sequence(),
                run_id=self.run_id, source_id=self.source_id, camera_id=self.camera_id,
                event_type=event_type, zone_id=zone.id, zone_name=zone.name,
                severity=priority, status=status,
                track_id=None, global_id=gid,
                frame_idx=frame_idx, frame_number=frame_idx + 1,
                timestamp_s=ts, timestamp_hms=_hms(ts), detected_at_s=ts,
                end_frame_idx=None, end_timestamp_s=None, duration_s=None,
                bbox_xyxy=bbox or (0.0, 0.0, 0.0, 0.0), bbox_tlwh=bbox_tlwh, bbox_norm=bbox_norm,
                foot_point=fp, frame_width=self.frame_width, frame_height=self.frame_height,
                confidence=conf_median, det_conf_median=conf_median, det_conf_min=conf_min,
                n_support_frames=len(confs),
                geom_margin_px=0.0, dwell_s=draft.get("dwell_s"),
                observed_radius=draft.get("observed_radius"),
                observed_radius_units="bbox_heights" if draft.get("observed_radius") is not None else None,
                rule_params=rule_params, zones_config_digest=self.zone_set.digest,
                suppressed_count=0, merge_count=0, truncated=draft.get("truncated", False),
                notes="",
            )

        if draft["kind"] == "enter_confirmed":
            intrusion_params = rule_params.get("intrusion", {})
            cooldown = intrusion_params.get("cooldown_seconds", 30.0)
            if intrusion_params.get("enabled", True) and self._deduper.gate_alert(
                ts, self.source_id, gid, zone.id, "zone_intrusion", cooldown
            ):
                kwargs = _base_kwargs("zone_intrusion", "open", "alert")
                kwargs["event_id"] = new_event_id()
                out.append(Event(**kwargs))

        elif draft["kind"] == "loiter_fired":
            cooldown = rule_params.get("loitering", {}).get("cooldown_seconds", 30.0)
            if self._deduper.gate_alert(ts, self.source_id, gid, zone.id, "loitering", cooldown):
                kwargs = _base_kwargs("loitering", "open", "alert")
                kwargs["event_id"] = new_event_id()
                out.append(Event(**kwargs))

        elif draft["kind"] == "closed":
            entry_ts = draft.get("entry_ts") or ts
            entry_frame_idx = draft.get("entry_frame_idx") or frame_idx
            for event_type, fired in (("zone_intrusion", draft["confirmed"]), ("loitering", draft["loiter_fired"])):
                if not fired:
                    continue
                type_params = rule_params.get("intrusion" if event_type == "zone_intrusion" else "loitering", {})
                if not type_params.get("enabled", True):
                    continue
                kwargs = _base_kwargs(event_type, "truncated" if draft["truncated"] else "closed", "interval")
                kwargs["event_id"] = new_event_id()
                kwargs["frame_idx"] = entry_frame_idx
                kwargs["frame_number"] = entry_frame_idx + 1
                kwargs["timestamp_s"] = entry_ts
                kwargs["timestamp_hms"] = _hms(entry_ts)
                kwargs["end_frame_idx"] = frame_idx
                kwargs["end_timestamp_s"] = ts
                kwargs["duration_s"] = max(0.0, ts - entry_ts)
                ev = Event(**kwargs)
                gated = self._deduper.gate_interval(
                    ev, self.source_id,
                    min_event_duration_s=type_params.get("min_event_duration_s", 0.30),
                    merge_gap_seconds=type_params.get("merge_gap_seconds", 5.0),
                    max_events_per_minute=type_params.get("max_events_per_minute", 60),
                    max_events_per_minute_per_zone=type_params.get("max_events_per_minute_per_zone", 30),
                )
                if gated is not None:
                    out.append(gated)
        return out

    def _next_sequence(self) -> int:
        self.sequence += 1
        return self.sequence

    def frame_overlay_state(self) -> dict:
        return self._last_zone_state

    def flush(self, frame_idx: int, ts: float) -> list[Event]:
        params_by_zone = {z.id: z.params for z in self.zone_set.zones}
        drafts = self.occupancy.flush(frame_idx, ts, params_by_zone)
        events: list[Event] = []
        for draft in drafts:
            zone = self.zones_by_id[draft["zone_id"]]
            events.extend(self._events_from_draft(zone, draft))
        return events

    def never_triggered_zones(self) -> set[str]:
        return set(self._never_triggered)
