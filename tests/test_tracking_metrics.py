"""Day-one hand-computed fixture: 2 objects x 3 frames = 6 GT appearances,
one miss (frame 1 hyp for gt id 2 absent) and a full id swap at frame 2
(hyp 10<->11 exchange which GT object they track). A swap is TWO mismatch
events, one per GT object whose matched hyp id changed (motmetrics tracks
mismatches per-GT-object, not per-swap) -> mota == 1 - (1 miss + 2 switches)/6.
"""
from __future__ import annotations

import pandas as pd
import pytest

from cctv.eval.tracking_metrics import accumulate_sequence, compute_summary


def _row(frame_idx, oid, x1, y1, w, h):
    return {"frame_idx": frame_idx, "id": oid, "x1": x1, "y1": y1, "bb_width": w, "bb_height": h}


def _hrow(frame_idx, tid, x1, y1, x2, y2):
    return {"frame_idx": frame_idx, "track_id": tid, "x1": x1, "y1": y1, "x2": x2, "y2": y2}


def test_hand_computed_mota_one_miss_one_switch():
    gt = pd.DataFrame([
        _row(0, 1, 0, 0, 10, 10), _row(0, 2, 100, 100, 10, 10),
        _row(1, 1, 0, 0, 10, 10), _row(1, 2, 100, 100, 10, 10),
        _row(2, 1, 0, 0, 10, 10), _row(2, 2, 100, 100, 10, 10),
    ])
    hyp = pd.DataFrame([
        _hrow(0, 10, 0, 0, 10, 10), _hrow(0, 11, 100, 100, 110, 110),
        _hrow(1, 10, 0, 0, 10, 10),  # id 2's hypothesis missing this frame -> 1 miss
        _hrow(2, 11, 0, 0, 10, 10), _hrow(2, 10, 100, 100, 110, 110),  # ids swapped -> 1 switch
    ])

    acc = accumulate_sequence(gt, hyp, 0, 2)
    summary = compute_summary({"seq": acc})
    mota = summary.loc["seq", "mota"]
    assert mota == pytest.approx(1 - 3 / 6, abs=1e-6)


def test_motp_is_a_distance_not_a_percentage():
    gt = pd.DataFrame([_row(0, 1, 0, 0, 10, 10)])
    hyp = pd.DataFrame([_hrow(0, 10, 0, 0, 10, 10)])  # identical box -> IoU 1.0 -> distance 0.0

    acc = accumulate_sequence(gt, hyp, 0, 0)
    summary = compute_summary({"seq": acc})
    assert summary.loc["seq", "motp_d"] == pytest.approx(0.0, abs=1e-6)
    assert summary.loc["seq", "motp_pct"] == pytest.approx(100.0, abs=1e-4)


def test_empty_gt_and_hyp_frames_are_scored():
    gt = pd.DataFrame([_row(0, 1, 0, 0, 10, 10)])
    hyp = pd.DataFrame([_hrow(0, 10, 0, 0, 10, 10)])
    # frames 1 and 2 exist in neither -- the full range must still be walked so an
    # empty-hyp frame's misses (if any GT existed) and an empty-gt frame's FPs (if
    # any hyp existed) would be counted; here both are empty so MOTA stays perfect.
    acc = accumulate_sequence(gt, hyp, 0, 2)
    summary = compute_summary({"seq": acc})
    assert summary.loc["seq", "mota"] == pytest.approx(1.0, abs=1e-6)
