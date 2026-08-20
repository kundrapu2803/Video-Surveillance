"""MOTChallenge tracking metrics via motmetrics. Loops the full [first, last]
frame range INCLUSIVE (not just frames present in either file) -- empty-GT
frames still need their FPs counted, empty-hyp frames still need their misses.
"""
from __future__ import annotations

import motmetrics as mm
import numpy as np
import pandas as pd

from cctv.geometry import iou_distance


def accumulate_sequence(gt_df: pd.DataFrame, hyp_df: pd.DataFrame,
                         first_frame_idx: int, last_frame_idx: int) -> mm.MOTAccumulator:
    acc = mm.MOTAccumulator(auto_id=False)

    gt_by_frame = {f: g for f, g in gt_df.groupby("frame_idx")}
    hyp_by_frame = {f: g for f, g in hyp_df.groupby("frame_idx")}

    for frame_idx in range(first_frame_idx, last_frame_idx + 1):
        g = gt_by_frame.get(frame_idx)
        h = hyp_by_frame.get(frame_idx)

        gt_ids = g["id"].tolist() if g is not None else []
        hyp_ids = h["track_id"].tolist() if h is not None and "track_id" in h.columns else (
            h["id"].tolist() if h is not None else []
        )

        if g is not None and len(g):
            gt_tlwh = np.stack([g["x1"], g["y1"], g["bb_width"], g["bb_height"]], axis=1)
        else:
            gt_tlwh = np.empty((0, 4))
        if h is not None and len(h):
            hx1, hy1 = h["x1"].to_numpy(), h["y1"].to_numpy()
            hw, hh = (h["x2"] - h["x1"]).to_numpy(), (h["y2"] - h["y1"]).to_numpy()
            hyp_tlwh = np.stack([hx1, hy1, hw, hh], axis=1)
        else:
            hyp_tlwh = np.empty((0, 4))

        dists = iou_distance(gt_tlwh, hyp_tlwh, max_iou=0.5)
        # acc.update(GT_ids, HYP_ids, dists, frameid=f) -- argument order matters:
        # swapping GT/HYP silently swaps FP and FN while MOTA stays plausible.
        acc.update(gt_ids, hyp_ids, dists, frameid=frame_idx)

    return acc


MOTP_RENAME = {"motp": "motp_d"}


def compute_summary(accs: dict[str, mm.MOTAccumulator]) -> pd.DataFrame:
    mh = mm.metrics.create()
    summary = mh.compute_many(
        list(accs.values()), metrics=mm.metrics.motchallenge_metrics,
        generate_overall=True, names=list(accs.keys()),
    )
    summary = summary.rename(columns=MOTP_RENAME)
    # MOTChallenge convention: higher is better, as a percentage.
    summary["motp_pct"] = 100.0 * (1.0 - summary["motp_d"])
    return summary


def render_report(summary: pd.DataFrame) -> str:
    mh = mm.metrics.create()
    namemap = dict(mm.io.motchallenge_metric_names)
    namemap.pop("motp", None)  # stock namemap mislabels the distance column as "MOTP"; we report both explicitly
    formatters = dict(mh.formatters)
    return mm.io.render_summary(
        summary.drop(columns=["motp_pct"]), formatters=formatters, namemap=namemap,
    ) + "\n\nmotp_pct (MOTChallenge convention, higher=better):\n" + summary["motp_pct"].round(2).to_string()
