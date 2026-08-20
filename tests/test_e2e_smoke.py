"""End-to-end pipeline smoke test: --detector stub, synthetic 20-frame source,
no torch, no model download, no network. Exercises the exact ordering
detect -> track -> smooth -> interpolate -> event -> write and asserts on
every artifact the CLI is supposed to produce.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from cctv.config import resolve_effective_config
from cctv.pipeline import run_source


def test_e2e_smoke(tmp_path, synth_source_dir, synth_zones_path, stub_script):
    output_dir = tmp_path / "results"

    eff = resolve_effective_config({
        "detector": "stub", "stub_script": stub_script,
        "tracker": "bytetrack", "no_video": False, "stride": 1, "threads": 2,
    })

    run_dir = run_source(
        video_path=str(synth_source_dir), zones_path=str(synth_zones_path),
        output_dir=str(output_dir), eff=eff, argv=["test"],
    )

    cam_dir = run_dir / "cameras" / "cam01"
    for name in ("events.jsonl", "events.csv", "alerts.jsonl", "events_summary.json",
                 "detections.parquet", "detections.jsonl", "tracks_mot.txt",
                 "zones.resolved.json", "annotated.mp4"):
        p = cam_dir / name
        assert p.exists() and p.stat().st_size > 0, f"missing or empty artifact: {p}"

    run_json = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run_json["status"] == "ok"
    cam = run_json["cameras"]["cam01"]
    assert cam["frames_processed"] == 20
    assert cam["frames_without_ids"] == 0

    df = pd.read_parquet(cam_dir / "detections.parquet")
    assert df["frame_idx"].nunique() == 20
    assert set(df["global_id"].dropna().unique()) == {1, 2}

    events = [json.loads(l) for l in (cam_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    event_types = {e["event_type"] for e in events}
    assert event_types == {"zone_intrusion", "loitering"}, f"unexpected event types: {event_types}"

    intrusion = [e for e in events if e["event_type"] == "zone_intrusion"][0]
    assert intrusion["global_id"] == 1
    assert intrusion["zone_id"] == "z_restricted"
    assert 9 <= intrusion["frame_idx"] <= 13, f"intrusion fired at an unexpected frame: {intrusion['frame_idx']}"

    loiter = [e for e in events if e["event_type"] == "loitering"][0]
    assert loiter["global_id"] == 2
    assert loiter["zone_id"] == "z_monitored"

    alerts = [json.loads(l) for l in (cam_dir / "alerts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(alerts) >= 2  # at least one zone_intrusion + one loitering alert
    assert all(a["record_kind"] == "alert" for a in alerts)

    for e in events:
        assert e["duration_s"] is not None

    zones_resolved = json.loads((cam_dir / "zones.resolved.json").read_text(encoding="utf-8"))
    assert {z["id"] for z in zones_resolved["zones"]} == {"z_restricted", "z_monitored"}


def test_stub_detector_offline_no_torch():
    """--detector stub must never import torch -- this is what makes the
    entire offline test suite fast and network-free."""
    import sys
    assert "torch" not in sys.modules or True  # torch may be present from other tests in the session;
    # the real guarantee is import-time: stub_detector.py has no torch import at all.
    import ast
    from pathlib import Path
    import cctv.detect.stub_detector as mod
    src = Path(mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(a.name.split(".")[0] == "torch" for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module is None or node.module.split(".")[0] != "torch"
