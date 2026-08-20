import random

import numpy as np
import pytest

from cctv.geometry import (
    xyxy_to_tlwh_0based,
    xyxy_to_tlwh_1based,
    tlwh_1based_to_xyxy,
    xyxy_to_cxcywh,
    to_mot_row,
    parse_mot_row,
    iou_distance,
    iou_similarity,
)


def test_xyxy_tlwh_roundtrip():
    rng = random.Random(0)
    for _ in range(1000):
        x1 = rng.uniform(0, 1000)
        y1 = rng.uniform(0, 1000)
        w = rng.uniform(1, 500)
        h = rng.uniform(1, 500)
        box = (x1, y1, x1 + w, y1 + h)
        tlwh = xyxy_to_tlwh_0based(box)
        back = (tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3])
        assert back == pytest.approx(box, abs=1e-9)


def test_center_xywh_is_not_tlwh():
    box = (10, 20, 50, 80)
    tlwh = xyxy_to_tlwh_0based(box)
    cxcywh = xyxy_to_cxcywh(box)
    assert tlwh == pytest.approx((10, 20, 40, 60))
    assert cxcywh == pytest.approx((30, 50, 40, 60))
    assert tlwh != cxcywh, (
        "top-left tlwh and center-based cxcywh must never be interchanged -- "
        "mixing Ultralytics' .xywh (center-based) into MOT/motmetrics tlwh "
        "code silently offsets every box by half its size."
    )


def test_mot_roundtrip_golden():
    # frame=1 (1-based), id=1, bb_left=912, bb_top=484, w=97, h=109, conf=0.86
    line = "1,1,912,484,97,109,1,1,0.86"
    frame_idx0, track_id, xyxy, conf = parse_mot_row(line)
    assert frame_idx0 == 0
    assert xyxy == pytest.approx((911.0, 483.0, 1008.0, 592.0))

    reserialized = to_mot_row(frame_idx0, track_id, xyxy, 1.0)
    frame_idx0_2, track_id_2, xyxy_2, _ = parse_mot_row(reserialized)
    assert frame_idx0_2 == frame_idx0
    assert track_id_2 == track_id
    assert xyxy_2 == pytest.approx(xyxy)


def test_mot_roundtrip_offsets_independent():
    line = "1,1,912,484,97,109,1,1,0.86"
    frame_idx0, _, xyxy, _ = parse_mot_row(line)
    # frame offset checked independently of pixel offset
    assert frame_idx0 == 0
    # pixel offset checked independently of frame offset
    assert xyxy[0] == pytest.approx(911.0)
    assert xyxy[1] == pytest.approx(483.0)


def test_iou_matrix_max_iou_enforced():
    # Two boxes at IoU ~0.4 -- must be rejected (NaN) at max_iou=0.5? No: 0.4 < 0.5
    # so it should be ACCEPTED. Use a pair below the gate instead: construct boxes
    # at IoU exactly ~0.1 and assert max_iou=0.5 doesn't wrongly reject it, then
    # construct a non-overlapping pair (IoU=0) far apart is trivially rejected by
    # the permissive default too, so instead prove the wrapper actually PASSES
    # max_iou by using a pair whose IoU is *below* an artificially tight max_iou.
    gt = np.array([[0.0, 0.0, 10.0, 10.0]])
    hyp = np.array([[3.0, 0.0, 10.0, 10.0]])  # IoU = 7*10 / (100+100-70) = 0.538
    dist_permissive = iou_distance(gt, hyp, max_iou=0.9)
    assert not np.isnan(dist_permissive[0, 0])

    dist_strict = iou_distance(gt, hyp, max_iou=0.1)
    assert np.isnan(dist_strict[0, 0]), (
        "max_iou must actually gate the distance -- proves the permissive "
        "signature default (max_iou=1.0) is never silently inherited."
    )


def test_iou_similarity_identical_boxes():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[0.0, 0.0, 10.0, 10.0]])
    sim = iou_similarity(a, b)
    assert sim[0, 0] == pytest.approx(1.0)


def test_iou_similarity_empty():
    a = np.zeros((0, 4))
    b = np.array([[0.0, 0.0, 10.0, 10.0]])
    sim = iou_similarity(a, b)
    assert sim.shape == (0, 1)
