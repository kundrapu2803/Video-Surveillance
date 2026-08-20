"""MOT17 GT loading for scoring: keep pedestrian rows only (class 1,
conf_flag != 0). min_visibility defaults to 0.0 -- filtering on visibility
removes the hard occluded GT and inflates MOTA by several points; MOT17's own
evaluation protocol does not filter on it either.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from cctv.io.mot import load_gt

PEDESTRIAN_CLASS = 1


def load_protocol_gt(gt_path: str | Path, *, min_visibility: float = 0.0) -> pd.DataFrame:
    df = load_gt(gt_path)
    filtered = df[(df["conf_flag"] != 0) & (df["class"] == PEDESTRIAN_CLASS) & (df["visibility"] >= min_visibility)]
    return filtered.reset_index(drop=True)
