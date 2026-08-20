"""Targeted event-engine tests beyond the full pipeline e2e smoke test:
resolution-invariance of the loitering stationarity test (bbox-height units
must give the same verdict regardless of source resolution), the radial-
walker scale gate (walking toward the camera must not read as stationary),
and the strict/relaxed id-rate guard.
"""
from __future__ import annotations

import json

import pytest

from cctv.events.engine import EventEngine, TrackerNotProducingIdsError
from cctv.io.zones import ZoneSet
from cctv.schema import TrackRecord

FPS = 10.0


def _zones_file(tmp_path, width, height, *, loiter_seconds=0.3, stationary_window_seconds=1.0,
                 stationary_radius=0.5, stationary_scale_ratio_max=1.25):
    zones = {
        "schema_version": "1.0",
        "source": {"name": "t", "kind": "video", "width": width, "height": height, "fps": FPS,
                    "camera_motion": "static"},
        "defaults": {
            "normalized": True, "min_confidence": 0.3, "enter_seconds": 0.1, "exit_seconds": 0.2,
            "dwell_grace_seconds": 0.5, "track_timeout_seconds": 1.0,
            "loiter_seconds": loiter_seconds, "stationary_window_seconds": stationary_window_seconds,
            "stationary_radius": stationary_radius, "stationary_scale_ratio_max": stationary_scale_ratio_max,
            "cooldown_seconds": 5.0, "merge_gap_seconds": 1.0, "min_event_duration_s": 0.0,
            "max_events_per_minute": 60, "max_events_per_minute_per_zone": 60, "min_area_frac": 0.0001,
        },
        "zones": [{
            "id": "z1", "name": "Z1", "zone_class": "monitored", "enabled": True, "priority": 1,
            "color": [0, 0, 255], "polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            "rules": {"intrusion": {"enabled": False}, "loitering": {"enabled": True}},
        }],
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "z.json"
    p.write_text(json.dumps(zones), encoding="utf-8")
    return p


def _run_track(engine, positions, *, gid=1, conf=0.9, box_wh=None):
    """positions: list of (x1, y1, x2, y2) in pixels, one per frame at FPS."""
    events = []
    for i, box in enumerate(positions):
        rec = TrackRecord(frame_idx=i, timestamp_s=i / FPS, global_id=gid, track_id=gid,
                           bbox_xyxy=box, smooth_bbox=None, conf=conf, cls=0, source_id="t")
        events.extend(engine.update(frame_idx=i, ts=i / FPS, records=[rec]))
    events.extend(engine.flush(len(positions) - 1, (len(positions) - 1) / FPS))
    return events


def _make_engine(zones_path, width, height, **kw):
    zs = ZoneSet.load(zones_path, frame_width=width, frame_height=height, camera_motion="static")
    return EventEngine(zs, run_id="t", source_id="t", camera_id="cam01",
                        frame_width=width, frame_height=height, fps=FPS, **kw)


def test_bbox_heights_invariant_to_resolution(tmp_path):
    """Identical synthetic jitter at two very different resolutions (a 6x gap,
    like MOT17 1080p vs UCF 240p) must give the SAME loitering verdict --
    the whole point of measuring stationarity in bbox-height units."""
    def verdict(width, height):
        scale = width / 320.0
        zp = _zones_file(tmp_path / f"r{width}", width, height)
        engine = _make_engine(zp, width, height)
        box_w, box_h = 30 * scale, 70 * scale
        cx, cy = width * 0.5, height * 0.5
        positions = []
        for i in range(20):
            jitter = (i % 3 - 1) * 1.0 * scale  # small proportional jitter
            x1 = cx - box_w / 2 + jitter
            y1 = cy - box_h / 2
            positions.append((x1, y1, x1 + box_w, y1 + box_h))
        events = _run_track(engine, positions)
        return any(e.event_type == "loitering" for e in events)

    assert verdict(320, 240) == verdict(1920, 1080)


def test_radial_walker_does_not_loiter(tmp_path):
    """A person walking straight toward the camera has near-zero (x,y) image
    displacement but a large scale change -- the scale gate must catch this,
    or a walker crossing a forward-facing street view reads as stationary."""
    zp = _zones_file(tmp_path, 640, 480, loiter_seconds=0.3, stationary_window_seconds=1.0)
    engine = _make_engine(zp, 640, 480)
    cx = 320
    positions = []
    for i in range(20):
        h = 40 + i * 6  # approaching the camera: height grows fast (>25% over the window)
        w = h * 0.4
        y2 = 480
        positions.append((cx - w / 2, y2 - h, cx + w / 2, y2))
    events = _run_track(engine, positions)
    assert not any(e.event_type == "loitering" for e in events)


def test_empty_frames_do_not_crash_or_emit_events(tmp_path):
    """Zero detections per frame (a common real case -- nobody in view) must
    flow through cleanly: no crash, no spurious events."""
    zp = _zones_file(tmp_path, 320, 240)
    engine = _make_engine(zp, 320, 240, strict_id_rate=False)
    events = []
    for i in range(30):
        events.extend(engine.update(frame_idx=i, ts=i / FPS, records=[]))
    events.extend(engine.flush(29, 29 / FPS))
    assert events == []


def test_rapid_reentry_within_cooldown_emits_one_alert(tmp_path):
    """A person bouncing in/out of a restricted zone faster than the cooldown
    window must produce exactly one intrusion ALERT, not one per re-entry --
    this is what events/dedup.py's cooldown gate exists for."""
    zones = {
        "schema_version": "1.0",
        "source": {"name": "t", "kind": "video", "width": 320, "height": 240, "fps": FPS,
                    "camera_motion": "static"},
        "defaults": {
            "normalized": True, "min_confidence": 0.3, "enter_seconds": 0.1, "exit_seconds": 0.1,
            "dwell_grace_seconds": 0.1, "track_timeout_seconds": 0.5, "cooldown_seconds": 2.0,
            "merge_gap_seconds": 0.1, "min_event_duration_s": 0.0,
            "max_events_per_minute": 60, "max_events_per_minute_per_zone": 60, "min_area_frac": 0.0001,
        },
        "zones": [{
            "id": "z1", "name": "Z1", "zone_class": "restricted", "enabled": True, "priority": 1,
            "color": [0, 0, 255], "polygon": [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]],
            "rules": {"intrusion": {"enabled": True}, "loitering": {"enabled": False}},
        }],
    }
    zp = tmp_path / "z.json"
    zp.write_text(json.dumps(zones), encoding="utf-8")
    engine = _make_engine(zp, 320, 240)

    inside = (140.0, 100.0, 180.0, 140.0)   # centre of the zone
    outside = (10.0, 10.0, 30.0, 30.0)      # well outside it
    # 3 rapid in/out/in/out/in cycles, all inside the 2s cooldown window
    positions = ([inside] * 3 + [outside] * 3) * 3

    events = _run_track(engine, positions)
    alerts = [e for e in events if e.record_kind == "alert" and e.event_type == "zone_intrusion"]
    assert len(alerts) == 1, f"expected exactly 1 alert under cooldown, got {len(alerts)}"


def test_strict_id_rate_raises_when_gt_expected(tmp_path):
    zp = _zones_file(tmp_path, 320, 240)
    engine = _make_engine(zp, 320, 240, strict_id_rate=True)
    with pytest.raises(TrackerNotProducingIdsError):
        for i in range(100):
            gid = 1 if i % 5 == 0 else None  # 20% id rate, well under the 80% floor
            rec = TrackRecord(frame_idx=i, timestamp_s=i / FPS, global_id=gid, track_id=gid,
                               bbox_xyxy=(10, 10, 20, 20), smooth_bbox=None, conf=0.9, cls=0, source_id="t")
            engine.update(frame_idx=i, ts=i / FPS, records=[rec])


def test_low_id_rate_warns_instead_of_raising_without_gt(tmp_path):
    zp = _zones_file(tmp_path, 320, 240)
    engine = _make_engine(zp, 320, 240, strict_id_rate=False)
    for i in range(100):
        gid = 1 if i % 5 == 0 else None
        rec = TrackRecord(frame_idx=i, timestamp_s=i / FPS, global_id=gid, track_id=gid,
                           bbox_xyxy=(10, 10, 20, 20), smooth_bbox=None, conf=0.9, cls=0, source_id="t")
        engine.update(frame_idx=i, ts=i / FPS, records=[rec])  # must not raise
    assert engine.low_id_rate_warning is not None
