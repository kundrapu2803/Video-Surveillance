"""Streamlit AppTest smoke test: the dashboard must at least import and run
without an exception against a real (small) run directory, without ever
invoking a detector -- this is what "never runs YOLO to display anything"
actually guarantees at test time.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from cctv.config import resolve_effective_config
from cctv.pipeline import run_source


@pytest.fixture
def sample_run_dir(tmp_path, synth_source_dir, synth_zones_path, stub_script, monkeypatch):
    output_dir = tmp_path / "results"
    eff = resolve_effective_config({
        "detector": "stub", "stub_script": stub_script, "tracker": "bytetrack",
        "no_video": True, "stride": 1, "threads": 2,
    })
    run_dir = run_source(
        video_path=str(synth_source_dir), zones_path=str(synth_zones_path),
        output_dir=str(output_dir), eff=eff, run_name="dashboard-test-run", argv=["test"],
    )

    import cctv.paths as paths_mod
    monkeypatch.setattr(paths_mod, "RESULTS_DIR", output_dir)
    import dashboard.loaders as loaders_mod
    monkeypatch.setattr(loaders_mod, "RESULTS_DIR", output_dir)
    return run_dir


def test_dashboard_loads_a_run(sample_run_dir):
    from streamlit.testing.v1 import AppTest

    app_path = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()
    assert not at.exception, f"dashboard raised on load: {at.exception}"
