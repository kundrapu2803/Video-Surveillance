import jsonschema

from cctv.schema import (
    Event,
    EVENT_SCHEMA_KEYS,
    ZONES_JSON_SCHEMA,
    RUN_JSON_SCHEMA,
)


def test_event_schema_keys_exact():
    from dataclasses import fields

    assert set(f.name for f in fields(Event)) == EVENT_SCHEMA_KEYS


def test_zones_json_schema_validates_golden_fixture():
    fixture = {
        "schema_version": "1.0",
        "source": {
            "name": "MOT17-09-FRCNN", "kind": "imgseq",
            "width": 1920, "height": 1080, "fps": 30.0, "camera_motion": "static",
        },
        "defaults": {"normalized": True, "loiter_seconds": 8.0},
        "zones": [
            {
                "id": "z1", "name": "Zone 1", "zone_class": "restricted",
                "enabled": True, "priority": 1, "color": [0, 0, 255],
                "polygon": [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]],
                "rules": {"intrusion": {"enabled": True}, "loitering": {"enabled": False}},
            }
        ],
    }
    jsonschema.validate(fixture, ZONES_JSON_SCHEMA)


def test_run_json_schema_validates_golden_fixture():
    fixture = {
        "schema_version": 1,
        "run_id": "abc123",
        "status": "ok",
        "sources": [],
        "cameras": {},
    }
    jsonschema.validate(fixture, RUN_JSON_SCHEMA)
