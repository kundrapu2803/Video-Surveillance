"""All file I/O for the dashboard funnels through here so caching has one
place to live. @st.cache_data keys on arguments only -- it has no idea a file
changed on disk -- so every loader takes an explicit mtime float to bust the
cache when the run directory is updated underneath it.
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from cctv.paths import CONFIGS_DIR, DATASET_DIR, PROJECT_ROOT, RESULTS_DIR, ZONE_CONFIGS_DIR

UPLOADS_DIR = RESULTS_DIR / "_uploads"


def list_runs() -> list[dict]:
    index_path = RESULTS_DIR / "index.json"
    if index_path.exists():
        return json.loads(index_path.read_text(encoding="utf-8"))
    return [{"run_id": p.name, "status": "?", "created_utc": None}
             for p in sorted(RESULTS_DIR.glob("*/"), key=lambda p: p.stat().st_mtime, reverse=True)
             if (p / "run.json").exists()]


def run_dir_for(run_id: str) -> Path:
    return RESULTS_DIR / run_id


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def load_run_json(run_dir_str: str, mtime: float) -> dict:
    return json.loads((Path(run_dir_str) / "run.json").read_text(encoding="utf-8"))


def load_run(run_id: str) -> dict:
    run_dir = run_dir_for(run_id)
    return load_run_json(str(run_dir), _mtime(run_dir / "run.json"))


@st.cache_data(show_spinner=False)
def load_events(cam_dir_str: str, mtime: float) -> pd.DataFrame:
    p = Path(cam_dir_str) / "events.jsonl"
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_detections(cam_dir_str: str, mtime: float) -> pd.DataFrame:
    p = Path(cam_dir_str) / "detections.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    for col in ("bbox_xyxy", "smooth_bbox", "zone_ids"):
        if col in df.columns:
            df[col] = df[col].apply(lambda v: json.loads(v) if isinstance(v, str) else v)
    return df


@st.cache_data(show_spinner=False)
def load_zones_resolved(cam_dir_str: str, mtime: float) -> dict:
    p = Path(cam_dir_str) / "zones.resolved.json"
    if not p.exists():
        return {"zones": []}
    return json.loads(p.read_text(encoding="utf-8"))


def cam_dir_for(run_id: str, camera_id: str = "cam01") -> Path:
    return run_dir_for(run_id) / "cameras" / camera_id


@st.cache_resource(show_spinner=False)
def open_frame_source(source_path: str):
    """A VideoCapture/ImageSequence handle is unhashable -- cache_resource,
    not cache_data. Keyed on the path string."""
    from cctv.io.frame_source import open_source
    return open_source(source_path, allow_lowres=True)


def read_frame_at(source, frame_idx: int):
    """Random access into a FrameSource. Image sequences index straight into
    the sorted file list; video seeks then verifies the landed position (a
    naive set() can land on the preceding keyframe with B-frame-heavy x264).
    """
    import cv2

    if hasattr(source, "_files"):
        frame_idx = max(0, min(frame_idx, len(source._files) - 1))
        return cv2.imread(str(source._files[frame_idx]))

    cap = cv2.VideoCapture(str(source.path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            return None
        return frame
    finally:
        cap.release()


# -- New-run controls ---------------------------------------------------------
# Everything below supports the dashboard's "Start/Run" flow: picking a
# source + zones, validating, shelling out to run.py/evaluate.py as a
# subprocess (never importing torch/the detector into this process), and
# offering the result for download.


def list_sample_sources() -> list[dict]:
    """Existing dataset sources, enumerated (not copied) from Dataset/.
    Paths are absolute and point straight at the original files/dirs."""
    sources = []
    mot17_dir = DATASET_DIR / "mot17"
    if mot17_dir.is_dir():
        for seq_dir in sorted(mot17_dir.iterdir()):
            img1 = seq_dir / "img1"
            if img1.is_dir():
                sources.append({
                    "label": f"{seq_dir.name} (MOT17)",
                    "path": str(img1.resolve()),
                    "dataset": "mot17",
                    "config": str((CONFIGS_DIR / "sources" / "mot17.yaml").resolve()),
                })
    ucf_dir = DATASET_DIR / "ucf-crime"
    if ucf_dir.is_dir():
        for clip in sorted(ucf_dir.glob("*.mp4")):
            sources.append({
                "label": f"{clip.stem} (UCF-Crime)",
                "path": str(clip.resolve()),
                "dataset": "ucf",
                "config": str((CONFIGS_DIR / "sources" / "ucf.yaml").resolve()),
            })
    return sources


def list_zone_configs() -> list[Path]:
    return sorted(ZONE_CONFIGS_DIR.glob("*.json")) if ZONE_CONFIGS_DIR.is_dir() else []


def probe_source(path: str) -> dict:
    """Cheap header-only probe (no full decode) -- same open_source() the
    frame scrubber already uses -- for width/height/fps before a run."""
    from cctv.io.frame_source import open_source

    src = open_source(path, allow_lowres=True)
    return {"width": src.width, "height": src.height, "fps": src.fps, "total_frames": getattr(src, "total_frames", -1)}


def validate_zones(zones_path: str, frame_width: int, frame_height: int):
    """Reuses the pipeline's own zones.json loader/validator -- the dashboard
    never parses or re-implements zone schema rules itself. Returns
    (zone_set_or_none, error_message_or_none)."""
    import jsonschema

    from cctv.io.zones import MovingCameraZoneError, ZoneSet, ZoneValidationError

    try:
        zone_set = ZoneSet.load(zones_path, frame_width=frame_width, frame_height=frame_height)
        return zone_set, None
    except jsonschema.ValidationError as e:
        return None, f"zones JSON does not match the required schema: {e.message}"
    except (ZoneValidationError, MovingCameraZoneError) as e:
        return None, str(e)


def save_uploaded_file(uploaded_file, subdir: str = "") -> Path:
    """Writes an st.file_uploader buffer to disk exactly once, under
    results/_uploads/ -- never under Dataset/ or configs/, so existing
    dataset/config files are never touched by a dashboard-driven run."""
    target_dir = UPLOADS_DIR / subdir if subdir else UPLOADS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / uploaded_file.name
    target.write_bytes(uploaded_file.getvalue())
    return target


def run_pipeline(argv: list[str]) -> subprocess.CompletedProcess:
    """Shells out to run.py as a subprocess -- same as a terminal invocation
    -- so this process never imports torch/the detector."""
    cmd = [sys.executable, str(PROJECT_ROOT / "run.py"), *argv]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))


def run_evaluate(argv: list[str]) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "evaluate.py"), *argv]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))


def zip_run_dir(run_id: str, camera_id: str = "cam01") -> bytes:
    """Bundles a run's annotated video, event logs, and manifest into an
    in-memory zip -- no temp files on disk."""
    run_dir = run_dir_for(run_id)
    cam_dir = cam_dir_for(run_id, camera_id)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for extra in ("run.json", "effective_config.yaml", "tracker_resolved.yaml"):
            p = run_dir / extra
            if p.exists():
                zf.write(p, arcname=extra)
        if cam_dir.is_dir():
            for p in cam_dir.iterdir():
                if p.is_file():
                    zf.write(p, arcname=f"cameras/{camera_id}/{p.name}")
    return buf.getvalue()
