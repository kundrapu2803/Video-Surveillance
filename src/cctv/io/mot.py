"""MOTChallenge gt.txt / det.txt readers, protocol-exact. Distinct from
geometry.parse_mot_row (which handles a single tracks_mot.txt row) because
gt.txt/det.txt carry two extra columns (class, visibility) that our own
tracks_mot.txt does not.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

GT_COLUMNS = ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height",
              "conf", "class", "visibility"]
DET_COLUMNS = ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf"]


def load_gt(path: str | Path) -> pd.DataFrame:
    """Returns a DataFrame with 0-based frame_idx and 0-based xyxy columns,
    plus the raw MOT-space columns for protocol filtering (class, conf, visibility).
    """
    df = pd.read_csv(path, header=None, names=GT_COLUMNS, usecols=range(9), engine="python")
    df = df.rename(columns={"conf": "conf_flag"})
    df["frame_idx"] = df["frame"] - 1
    df["x1"] = df["bb_left"] - 1.0
    df["y1"] = df["bb_top"] - 1.0
    df["x2"] = df["x1"] + df["bb_width"]
    df["y2"] = df["y1"] + df["bb_height"]
    return df


def load_det(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=DET_COLUMNS, usecols=range(7), engine="python")
    df["frame_idx"] = df["frame"] - 1
    df["x1"] = df["bb_left"] - 1.0
    df["y1"] = df["bb_top"] - 1.0
    df["x2"] = df["x1"] + df["bb_width"]
    df["y2"] = df["y1"] + df["bb_height"]
    return df


def load_tracks_mot(path: str | Path) -> pd.DataFrame:
    """Reads our own tracks_mot.txt (10-column MOTChallenge format) back in,
    via the single geometry.py conversion point."""
    from cctv.geometry import parse_mot_row

    rows = []
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return pd.DataFrame(columns=["frame_idx", "track_id", "x1", "y1", "x2", "y2", "conf"])
    for line in text.splitlines():
        frame_idx, track_id, xyxy, conf = parse_mot_row(line)
        rows.append((frame_idx, track_id, *xyxy, conf))
    return pd.DataFrame(rows, columns=["frame_idx", "track_id", "x1", "y1", "x2", "y2", "conf"])
