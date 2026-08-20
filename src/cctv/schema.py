"""THE contract. No module may define its own version of any of this --
tests/test_schema_frozen.py and tests/test_no_duplicate_geometry.py enforce it.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal, Optional

BBox = tuple[float, float, float, float]


# --------------------------------------------------------------------------- #
# SourceRecord
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class SourceRecord:
    id: str
    dataset: Literal["mot17", "ucf", "user"]
    kind: Literal["video", "imgseq"]
    path: str  # ABSOLUTE
    frames: int  # -1 if genuinely unknown
    fps: float
    fps_source: Literal["seqinfo", "container", "cli", "default"]
    width: int
    height: int
    frame_index_base: int  # 1 for MOT17 imgseq, 0 for video
    gt_path: Optional[str]
    seqinfo_path: Optional[str]
    camera_motion: Literal["static", "moving", "unknown"]
    zone_capable: bool
    profile: str
    imgsz_override: Optional[int] = None
    meta: dict = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.meta is None:
            object.__setattr__(self, "meta", {})


# --------------------------------------------------------------------------- #
# TrackRecord
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class TrackRecord:
    frame_idx: int  # 0-based
    timestamp_s: float
    global_id: Optional[int]
    track_id: Optional[int]
    bbox_xyxy: BBox  # RAW -- used for MOT txt, metrics, ALL zone geometry
    smooth_bbox: Optional[BBox]  # ANNOTATION ONLY -- never zones/scoring
    conf: float
    cls: int  # always 0 (person)
    source_id: str
    interpolated: bool = False
    zone_ids: tuple[str, ...] = ()
    reid_event: Literal["none", "new", "resumed"] = "none"
    reid_score: Optional[float] = None
    reid_gap_s: Optional[float] = None


# --------------------------------------------------------------------------- #
# Event
# --------------------------------------------------------------------------- #

EVENT_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class Event:
    schema_version: str
    event_id: str
    record_kind: Literal["alert", "interval"]
    sequence: int
    run_id: str
    source_id: str
    camera_id: str
    event_type: Literal["zone_intrusion", "loitering"]
    zone_id: str
    zone_name: str
    severity: int
    status: Literal["open", "closed", "truncated"]
    track_id: Optional[int]
    global_id: int
    frame_idx: int  # 0-based
    frame_number: int  # 1-based
    timestamp_s: float
    timestamp_hms: str
    detected_at_s: float
    end_frame_idx: Optional[int]
    end_timestamp_s: Optional[float]
    duration_s: Optional[float]
    bbox_xyxy: BBox
    bbox_tlwh: BBox
    bbox_norm: BBox
    foot_point: tuple[float, float]
    frame_width: int
    frame_height: int
    confidence: float
    det_conf_median: float
    det_conf_min: float
    n_support_frames: int
    geom_margin_px: float
    dwell_s: Optional[float]
    observed_radius: Optional[float]
    observed_radius_units: Optional[str]
    rule_params: dict
    zones_config_digest: str
    suppressed_count: int
    merge_count: int
    truncated: bool
    notes: str


EVENT_SCHEMA_KEYS = frozenset(f.name for f in fields(Event))

EVENT_CSV_COLUMNS = [
    "event_id", "record_kind", "sequence", "run_id", "source_id", "camera_id",
    "event_type", "zone_id", "zone_name", "severity", "status",
    "track_id", "global_id",
    "frame_idx", "frame_number", "timestamp_s", "timestamp_hms", "detected_at_s",
    "end_frame_idx", "end_timestamp_s", "duration_s",
    "bbox_xyxy", "bbox_tlwh", "bbox_norm", "foot_point",
    "frame_width", "frame_height",
    "confidence", "det_conf_median", "det_conf_min", "n_support_frames",
    "geom_margin_px", "dwell_s", "observed_radius", "observed_radius_units",
    "zones_config_digest", "suppressed_count", "merge_count", "truncated", "notes",
]


# --------------------------------------------------------------------------- #
# zones.json JSON Schema (authoring schema, v1.0)
# --------------------------------------------------------------------------- #

ZONE_RULE_SCHEMA = {
    "type": "object",
    "properties": {"enabled": {"type": "boolean"}},
    "additionalProperties": True,
}

ZONES_JSON_SCHEMA = {
    "type": "object",
    "required": ["schema_version", "source", "defaults", "zones"],
    "properties": {
        "schema_version": {"type": "string"},
        "source": {
            "type": "object",
            "required": ["name", "kind", "width", "height", "fps", "camera_motion"],
        },
        "defaults": {"type": "object"},
        "zones": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "name", "zone_class", "enabled", "polygon", "rules"],
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "zone_class": {"enum": ["restricted", "monitored"]},
                    "enabled": {"type": "boolean"},
                    "priority": {"type": "integer"},
                    "color": {"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3},
                    "polygon": {
                        "type": "array",
                        "minItems": 3,
                        "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                    },
                    "rules": {
                        "type": "object",
                        "properties": {
                            "intrusion": ZONE_RULE_SCHEMA,
                            "loitering": ZONE_RULE_SCHEMA,
                        },
                    },
                },
            },
        },
    },
}

ZONES_RESOLVED_JSON_SCHEMA = ZONES_JSON_SCHEMA  # same shape; polygons are pixel-space, params pre-resolved

RUN_JSON_SCHEMA = {
    "type": "object",
    "required": ["schema_version", "run_id", "status", "sources", "cameras"],
    "properties": {
        "schema_version": {"type": "integer"},
        "run_id": {"type": "string"},
        "status": {"enum": ["running", "ok", "crashed", "interrupted"]},
        "sources": {"type": "array"},
        "cameras": {"type": "object"},
    },
}


# --------------------------------------------------------------------------- #
# Output directory layout
# --------------------------------------------------------------------------- #

ARTIFACT_LAYOUT = {
    "root_events_json": "events.json",
    "root_events_csv": "events.csv",
    "root_summary_json": "summary.json",
    "root_index_json": "index.json",
    "root_latest_run_txt": "latest_run.txt",
    "run_manifest": "run.json",
    "run_effective_config": "effective_config.yaml",
    "run_tracker_resolved": "tracker_resolved.yaml",
    "run_requirements_lock": "requirements.lock.txt",
    "run_log": "logs/run.log",
    "camera_events_jsonl": "cameras/{camera_id}/events.jsonl",
    "camera_events_csv": "cameras/{camera_id}/events.csv",
    "camera_alerts_jsonl": "cameras/{camera_id}/alerts.jsonl",
    "camera_events_summary": "cameras/{camera_id}/events_summary.json",
    "camera_detections_parquet": "cameras/{camera_id}/detections.parquet",
    "camera_detections_jsonl": "cameras/{camera_id}/detections.jsonl",
    "camera_tracks_mot": "cameras/{camera_id}/tracks_mot.txt",
    "camera_dets_jsonl": "cameras/{camera_id}/dets.jsonl",
    "camera_zones_resolved": "cameras/{camera_id}/zones.resolved.json",
    "camera_video": "cameras/{camera_id}/annotated.mp4",
}
