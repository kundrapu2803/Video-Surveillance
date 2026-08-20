"""draw_overlay(): the ONLY rendering function, shared by the live video writer
and the dashboard scrubber, so the video and the dashboard can never disagree
about what a frame looked like.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Optional

import cv2
import numpy as np

from cctv.render.palette import color_for_id

TRAIL_TTL_FRAMES = 90
TRAIL_MAXLEN = 30

_trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=TRAIL_MAXLEN))
_trail_last_seen: dict[int, int] = {}


def _evict_stale_trails(frame_idx: int) -> None:
    stale = [gid for gid, last in _trail_last_seen.items() if frame_idx - last > TRAIL_TTL_FRAMES]
    for gid in stale:
        _trail_last_seen.pop(gid, None)
        _trails.pop(gid, None)


def draw_overlay(
    bgr: np.ndarray,
    records: list,
    zone_overlay_state: Optional[dict] = None,
    *,
    frame_idx: int = 0,
    hud: Optional[dict] = None,
    show_trails: bool = True,
    show_ids: bool = True,
    update_trails: bool = True,
) -> np.ndarray:
    frame = bgr.copy()

    if zone_overlay_state:
        for zid, state in zone_overlay_state.items():
            poly = state.get("polygon_px")
            if poly is None:
                continue
            color = tuple(int(c) for c in state.get("color", (0, 0, 255)))
            breached = state.get("breached", False)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [poly.astype(np.int32)], color)
            alpha = 0.28 if breached else 0.14
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            thickness = 3 if breached else 1
            cv2.polylines(frame, [poly.astype(np.int32)], True, color, thickness)
            label_pt = poly.reshape(-1, 2)[0].astype(int)
            active_types = state.get("active_event_types") or set()
            label = state.get("name", zid)
            if active_types:
                label += "  [" + " + ".join(sorted(t.replace("_", " ").upper() for t in active_types)) + "]"
            cv2.putText(frame, label, tuple(label_pt), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, color, 2 if active_types else 1, cv2.LINE_AA)

    _evict_stale_trails(frame_idx)

    # Per-person active event types (zone_intrusion/loitering), from the SAME
    # engine state that already drives the zone-polygon highlight -- so a
    # breached zone and the specific member(s) causing it can never disagree.
    member_active_types: dict = defaultdict(set)
    if zone_overlay_state:
        for zstate in zone_overlay_state.values():
            types = zstate.get("active_event_types") or set()
            if not types:
                continue
            for m_gid in zstate.get("member_gids") or ():
                member_active_types[m_gid] |= types

    for rec in records:
        gid = rec.global_id if rec.global_id is not None else rec.track_id
        person_event_types = member_active_types.get(gid, set()) if gid is not None else set()

        if not show_ids or gid is None:
            color = (0, 200, 0) if not show_ids else (160, 160, 160)
        elif person_event_types:
            color = (0, 0, 255)  # alert red: this specific person is the one triggering the event
        else:
            color = color_for_id(gid)

        box = rec.smooth_bbox if rec.smooth_bbox is not None else rec.bbox_xyxy
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if person_event_types else 2)

        label = f"id{gid} {rec.conf:.2f}" if show_ids and gid is not None else f"{rec.conf:.2f}"
        if show_ids and rec.reid_event == "resumed":
            label += " [RE-ID]"
        if person_event_types:
            label += "  [" + " + ".join(sorted(t.replace("_", " ").upper() for t in person_event_types)) + "]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)

        if gid is not None and update_trails:
            fx, fy = (x1 + x2) / 2.0, y2
            _trails[gid].append((int(fx), int(fy)))
            _trail_last_seen[gid] = frame_idx
        if show_trails and gid is not None:
            pts = list(_trails[gid])
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], color, 2, cv2.LINE_AA)

    if hud:
        y = 20
        for key, val in hud.items():
            cv2.putText(frame, f"{key}: {val}", (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y += 18

    return frame


def reset_trails() -> None:
    """Called at the start of each source so trails from a previous run/source
    (which reuse global_id space differently) don't bleed into a new video."""
    _trails.clear()
    _trail_last_seen.clear()
