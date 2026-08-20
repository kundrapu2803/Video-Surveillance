"""The ONE zone editor. cv2 window: left-click adds a point to the current
zone, right-click undoes the last point, 'n' finishes the current zone and
starts a new one, 's' saves, 'q' quits. Always writes normalized: true, then
immediately reloads through ZoneSet.load() and prints the validation report
-- a broken config is never left on disk (.bak restore on failure).

--from-points is a headless fallback for machines without a display.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _default_defaults() -> dict:
    return {
        "normalized": True, "reference_point": "bottom_center", "min_confidence": 0.35,
        "enter_seconds": 0.20, "exit_seconds": 0.60, "track_timeout_seconds": 3.0,
        "dwell_grace_seconds": 1.5, "loiter_seconds": 8.0, "stationary_window_seconds": 4.0,
        "stationary_radius": 0.5, "stationary_scale_ratio_max": 1.25,
        "cooldown_seconds": 30.0, "merge_gap_seconds": 5.0, "min_event_duration_s": 0.30,
        "max_events_per_minute": 60, "max_events_per_minute_per_zone": 30,
    }


def interactive_editor(image_path: str, out_path: str, scale: float) -> dict:
    import cv2

    img = cv2.imread(image_path)
    if img is None:
        raise IOError(f"could not read {image_path}")
    h, w = img.shape[:2]
    disp = cv2.resize(img, (int(w * scale), int(h * scale))) if scale != 1.0 else img.copy()

    zones: list[list[tuple[float, float]]] = [[]]
    zone_idx = 0

    def on_mouse(event, x, y, flags, param):
        # The scale trap: every click is in DISPLAY space; divide by scale
        # before storing, so the saved polygon is in ORIGINAL image space.
        if event == cv2.EVENT_LBUTTONDOWN:
            zones[zone_idx].append((x / scale, y / scale))
        elif event == cv2.EVENT_RBUTTONDOWN and zones[zone_idx]:
            zones[zone_idx].pop()

    win = "draw_zones (L-click add, R-click undo, n=next zone, s=save, q=quit)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    print("Click to add points (in ORIGINAL image coordinates, title bar shown scaled).")
    while True:
        frame = disp.copy()
        for zi, pts in enumerate(zones):
            pts_disp = [(int(px * scale), int(py * scale)) for px, py in pts]
            for p in pts_disp:
                cv2.circle(frame, p, 3, (0, 255, 0), -1)
            if len(pts_disp) > 1:
                cv2.polylines(frame, [__import__("numpy").array(pts_disp)], zi < zone_idx, (0, 0, 255), 1)
        cv2.putText(frame, f"zone {zone_idx}: {len(zones[zone_idx])} pts (orig space)",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(win, frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("n"):
            zones.append([])
            zone_idx += 1
        elif key == ord("s"):
            break
        elif key == ord("q"):
            cv2.destroyAllWindows()
            sys.exit(0)
    cv2.destroyAllWindows()

    return _build_zones_json(zones, w, h)


def _build_zones_json(polygons: list[list[tuple[float, float]]], width: int, height: int) -> dict:
    zones_out = []
    for i, pts in enumerate(polygons):
        if len(pts) < 3:
            continue
        zones_out.append({
            "id": f"z_{i}", "name": f"Zone {i}", "zone_class": "restricted",
            "enabled": True, "priority": i + 1, "color": [0, 0, 255],
            "polygon": [[round(px / width, 4), round(py / height, 4)] for px, py in pts],
            "rules": {"intrusion": {"enabled": True}, "loitering": {"enabled": False}},
        })
    return {
        "schema_version": "1.0",
        "source": {"name": "unnamed", "kind": "video", "width": width, "height": height,
                    "fps": 25.0, "camera_motion": "static"},
        "defaults": _default_defaults(),
        "zones": zones_out,
    }


def save_and_validate(zones_dict: dict, out_path: str, *, frame_width: int, frame_height: int) -> None:
    out = Path(out_path)
    backup = None
    if out.exists():
        backup = out.with_suffix(out.suffix + ".bak")
        shutil.copyfile(out, backup)

    out.write_text(json.dumps(zones_dict, indent=2), encoding="utf-8")

    from cctv.io.zones import ZoneSet

    try:
        loaded = ZoneSet.load(out, frame_width=frame_width, frame_height=frame_height, camera_motion="static")
        print(f"saved {out} -- validated OK, {len(loaded.zones)} zone(s):")
        for z in loaded.zones:
            print(f"  {z.id}: {z.name} ({z.zone_class})")
    except Exception as e:
        if backup:
            shutil.copyfile(backup, out)
            print(f"VALIDATION FAILED, restored previous version from {backup}: {e}", file=sys.stderr)
        else:
            out.unlink(missing_ok=True)
            print(f"VALIDATION FAILED, removed broken file: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="a representative frame to draw zones on")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=1.0, help="display scale for oversized frames")
    ap.add_argument("--from-points", default=None,
                     help="headless fallback: JSON file [[[x,y],...], [[x,y],...]] in ORIGINAL pixel coords")
    args = ap.parse_args()

    import cv2
    img = cv2.imread(args.image)
    if img is None:
        print(f"could not read {args.image}", file=sys.stderr)
        return 1
    h, w = img.shape[:2]

    if args.from_points:
        polygons = json.loads(Path(args.from_points).read_text(encoding="utf-8"))
        zones_dict = _build_zones_json([[tuple(p) for p in poly] for poly in polygons], w, h)
    else:
        try:
            zones_dict = interactive_editor(args.image, args.out, args.scale)
        except Exception as e:
            print(f"interactive editor unavailable ({e}); use --from-points for a headless fallback", file=sys.stderr)
            return 1

    save_and_validate(zones_dict, args.out, frame_width=w, frame_height=h)
    return 0


if __name__ == "__main__":
    sys.exit(main())
