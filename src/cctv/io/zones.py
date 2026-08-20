"""zones.json load/validate/scale -> zones.resolved.json. See plan section 4.5."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cctv.schema import ZONES_JSON_SCHEMA

ZONE_CLASS_PRESETS = {
    "restricted": {
        "intrusion": {"enabled": True},
        "loitering": {"enabled": False},
    },
    "monitored": {
        "intrusion": {"enabled": False},
        "loitering": {"enabled": True},
    },
}

BUILTIN_DEFAULTS = {
    "normalized": True,
    "reference_point": "bottom_center",
    "foot_inset_frac": 0.0,
    "hysteresis_band_px": 0.0,
    "min_confidence": 0.35,
    "enter_seconds": 0.20,
    "exit_seconds": 0.60,
    "track_timeout_seconds": 3.0,
    "dwell_grace_seconds": 1.5,
    "loiter_seconds": 8.0,
    "stationary_window_seconds": 4.0,
    "stationary_radius": 0.5,
    "stationary_units": "bbox_heights",
    "stationary_scale_ratio_max": 1.25,
    "min_stationary_samples": 0,
    "cooldown_seconds": 30.0,
    "merge_gap_seconds": 5.0,
    "min_event_duration_s": 0.30,
    "max_events_per_minute": 60,
    "max_events_per_minute_per_zone": 30,
}


class MovingCameraZoneError(Exception):
    pass


class ZoneValidationError(Exception):
    pass


@dataclass
class Zone:
    id: str
    name: str
    zone_class: str
    enabled: bool
    priority: int
    color: tuple[int, int, int]
    polygon_px: np.ndarray  # (N,1,2) float32, this source's pixel space
    params: dict = field(default_factory=dict)  # fully resolved per-rule params


@dataclass
class ZoneSet:
    zones: list[Zone]
    source_meta: dict
    digest: str

    def as_resolved_dict(self) -> dict:
        return {
            "schema_version": "1.0",
            "source": self.source_meta,
            "zones": [
                {
                    "id": z.id, "name": z.name, "zone_class": z.zone_class,
                    "enabled": z.enabled, "priority": z.priority, "color": list(z.color),
                    "polygon": z.polygon_px.reshape(-1, 2).tolist(),
                    "rules": z.params,
                }
                for z in self.zones
            ],
        }

    @staticmethod
    def load(
        zones_json_path: str | Path,
        *,
        frame_width: int,
        frame_height: int,
        camera_motion: str = "unknown",
        allow_moving_camera_zones: bool = False,
        cli_overrides: dict | None = None,
    ) -> "ZoneSet":
        import jsonschema

        path = Path(zones_json_path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        jsonschema.validate(raw, ZONES_JSON_SCHEMA)

        if camera_motion == "moving" and not allow_moving_camera_zones:
            raise MovingCameraZoneError(
                f"{path}: source camera_motion='moving' -- an image-space polygon "
                "stops describing the same physical region once the camera pans. "
                "Pass --allow-moving-camera-zones to override."
            )

        src = raw["source"]
        src_w, src_h = src.get("width", frame_width), src.get("height", frame_height)
        normalized = raw.get("defaults", {}).get("normalized", True)

        if not normalized:
            scale_ratio = max(frame_width / src_w, src_w / frame_width)
            if scale_ratio > 1.5:
                raise ZoneValidationError(
                    f"{path}: absolute-pixel zones authored for {src_w}x{src_h} "
                    f"applied to a {frame_width}x{frame_height} source "
                    f"(ratio {scale_ratio:.1f}x). Pass --zone-autoscale, or "
                    "author normalized polygons instead."
                )

        digest = hashlib.sha1(path.read_bytes()).hexdigest()

        file_defaults = {**BUILTIN_DEFAULTS, **raw.get("defaults", {})}
        if cli_overrides:
            file_defaults = {**file_defaults, **cli_overrides}

        zones = []
        for zraw in raw["zones"]:
            preset = ZONE_CLASS_PRESETS.get(zraw["zone_class"], {})
            resolved_rules = {}
            for rule_name in ("intrusion", "loitering"):
                merged = {**file_defaults, **preset.get(rule_name, {}), **zraw.get("rules", {}).get(rule_name, {})}
                resolved_rules[rule_name] = merged

            pts = np.array(zraw["polygon"], dtype=np.float64)
            if len(pts) < 3:
                raise ZoneValidationError(f"zone {zraw['id']}: needs >=3 points")
            if not np.all(np.isfinite(pts)):
                raise ZoneValidationError(f"zone {zraw['id']}: NaN/inf coordinate")

            if normalized:
                pts_px = pts * np.array([frame_width, frame_height])
            else:
                pts_px = pts

            contour = np.ascontiguousarray(pts_px, dtype=np.float32).reshape(-1, 1, 2)

            import cv2

            area = cv2.contourArea(contour)
            frame_area = frame_width * frame_height
            min_area_frac = file_defaults.get("min_area_frac", 0.0005)
            if area < min_area_frac * frame_area:
                raise ZoneValidationError(
                    f"zone {zraw['id']}: area {area:.0f}px^2 is below "
                    f"{min_area_frac:.4%} of frame area -- likely a zero-overlap "
                    "config authored at the wrong scale/resolution."
                )

            zones.append(
                Zone(
                    id=zraw["id"], name=zraw["name"], zone_class=zraw["zone_class"],
                    enabled=zraw.get("enabled", True), priority=zraw.get("priority", 1),
                    color=tuple(zraw.get("color", [0, 0, 255])),
                    polygon_px=contour, params=resolved_rules,
                )
            )

        return ZoneSet(zones=zones, source_meta=src, digest=digest)

    def save_resolved(self, out_path: str | Path) -> None:
        Path(out_path).write_text(
            json.dumps(self.as_resolved_dict(), indent=2), encoding="utf-8"
        )


def foot_point(bbox_xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    return ((x1 + x2) / 2.0, y2)


def zone_membership(point: tuple[float, float], zone: Zone) -> bool:
    import cv2

    pt = (float(point[0]), float(point[1]))
    return cv2.pointPolygonTest(zone.polygon_px, pt, False) >= 0
