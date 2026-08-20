"""Detection quality independent of tracking: AP50 only (never AP50-95 --
MOT17 GT boxes are amodal while COCO-trained YOLO predicts visible extent, so
tight-IoU thresholds penalise a correct detector for a labeling convention
mismatch) plus P/R/F1 at the pipeline's actual operating confidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cctv.geometry import iou_similarity


def _match_frame(gt_frame: pd.DataFrame, det_frame: pd.DataFrame, iou_thresh: float) -> tuple[int, int, int]:
    """Returns (tp, fp, fn) for one frame via greedy IoU matching, descending confidence."""
    if det_frame.empty:
        return 0, 0, len(gt_frame)
    if gt_frame.empty:
        return 0, len(det_frame), 0

    gt_tlwh = np.stack([gt_frame["x1"], gt_frame["y1"],
                         gt_frame["x2"] - gt_frame["x1"], gt_frame["y2"] - gt_frame["y1"]], axis=1)
    det_sorted = det_frame.sort_values("conf", ascending=False)
    det_tlwh = np.stack([det_sorted["x1"], det_sorted["y1"],
                          det_sorted["x2"] - det_sorted["x1"], det_sorted["y2"] - det_sorted["y1"]], axis=1)

    iou = iou_similarity(det_tlwh, gt_tlwh)
    matched_gt = set()
    tp = 0
    for i in range(len(det_sorted)):
        j = np.argmax(iou[i]) if iou.shape[1] else -1
        if j >= 0 and iou[i, j] >= iou_thresh and j not in matched_gt:
            matched_gt.add(j)
            tp += 1
    fp = len(det_sorted) - tp
    fn = len(gt_frame) - len(matched_gt)
    return tp, fp, fn


def precision_recall_f1(gt_df: pd.DataFrame, det_df: pd.DataFrame, *, conf_thresh: float,
                         iou_thresh: float = 0.5) -> dict:
    det_at_conf = det_df[det_df["conf"] >= conf_thresh]
    tp = fp = fn = 0
    frames = set(gt_df["frame_idx"]) | set(det_at_conf["frame_idx"])
    gt_by_frame = {f: g for f, g in gt_df.groupby("frame_idx")}
    det_by_frame = {f: d for f, d in det_at_conf.groupby("frame_idx")}
    for f in frames:
        g = gt_by_frame.get(f, gt_df.iloc[0:0])
        d = det_by_frame.get(f, det_at_conf.iloc[0:0])
        t, p, n = _match_frame(g, d, iou_thresh)
        tp += t; fp += p; fn += n
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"conf_thresh": conf_thresh, "iou_thresh": iou_thresh, "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def ap50(gt_df: pd.DataFrame, det_df: pd.DataFrame, iou_thresh: float = 0.5) -> float:
    """Standard single-pass AP: match detections to GT in GLOBAL
    confidence-descending order (each frame's GT boxes matched at most once),
    then integrate the resulting precision/recall curve. O(detections +
    frames), not O(thresholds x frames) -- recomputing a full frame sweep per
    unique confidence value is the kind of thing that looks fine on a 20-frame
    test fixture and then burns 7 minutes of CPU on a 525-frame real sequence.
    """
    if det_df.empty or gt_df.empty:
        return 0.0

    total_gt = len(gt_df)
    gt_by_frame: dict = {f: g[["x1", "y1", "x2", "y2"]].to_numpy() for f, g in gt_df.groupby("frame_idx")}
    matched: dict = {f: np.zeros(len(boxes), dtype=bool) for f, boxes in gt_by_frame.items()}

    det_sorted = det_df.sort_values("conf", ascending=False)
    tp = np.zeros(len(det_sorted))
    fp = np.zeros(len(det_sorted))

    for i, row in enumerate(det_sorted.itertuples()):
        gt_boxes = gt_by_frame.get(row.frame_idx)
        if gt_boxes is None or len(gt_boxes) == 0:
            fp[i] = 1
            continue
        det_box = np.array([[row.x1, row.y1, row.x2 - row.x1, row.y2 - row.y1]])
        gt_tlwh = np.hstack([gt_boxes[:, :2], gt_boxes[:, 2:] - gt_boxes[:, :2]])
        iou = iou_similarity(det_box, gt_tlwh)[0]
        j = int(np.argmax(iou)) if len(iou) else -1
        frame_matched = matched[row.frame_idx]
        if j >= 0 and iou[j] >= iou_thresh and not frame_matched[j]:
            frame_matched[j] = True
            tp[i] = 1
        else:
            fp[i] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / total_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)

    # monotonic envelope (precision never increases as recall increases, per VOC convention)
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    r = np.concatenate([[0.0], recall, [1.0]])
    p = np.concatenate([[precision[0] if len(precision) else 0.0], precision, [0.0]])
    return float(np.trapz(p, r))
