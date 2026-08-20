"""The ONLY place coordinate/index conversions happen in this codebase.

Internal canonical form, everywhere outside this module and the MOT file
boundary: frame_idx is 0-based; boxes are float xyxy, 0-based pixels, in the
source's original resolution (never letterboxed coordinates).

Three traps this module exists to contain:
  - Ultralytics' Boxes.xywh is CENTRE-based; MOT/motmetrics want TOP-LEFT
    (tlwh). Mixing them offsets every box by half its size -> IoU ~= 0 ->
    strongly negative MOTA. Only Boxes.xyxy is ever consumed.
  - MOT files are 1-indexed in both frame numbers and pixel coordinates.
    The +1 pixel offset is IoU-neutral and therefore silent -- it will never
    surface as a visible bug, only as an untested convention violation.
  - The +1 frame offset is NOT neutral: one frame of misalignment at 30fps
    in a crowded scene costs several MOTA points and reads as "a mediocre
    tracker" rather than "an off-by-one".
"""
from __future__ import annotations

import numpy as np

MOT_FRAME_BASE = 1
MOT_COORD_BASE = 1.0

BBox = tuple[float, float, float, float]


def xyxy_to_tlwh_0based(box: BBox) -> BBox:
    x1, y1, x2, y2 = box
    return (x1, y1, x2 - x1, y2 - y1)


def tlwh_0based_to_xyxy(box: BBox) -> BBox:
    x, y, w, h = box
    return (x, y, x + w, y + h)


def xyxy_to_tlwh_1based(box: BBox) -> BBox:
    x, y, w, h = xyxy_to_tlwh_0based(box)
    return (x + MOT_COORD_BASE, y + MOT_COORD_BASE, w, h)


def tlwh_1based_to_xyxy(box: BBox) -> BBox:
    x, y, w, h = box
    x0 = x - MOT_COORD_BASE
    y0 = y - MOT_COORD_BASE
    return (x0, y0, x0 + w, y0 + h)


def xyxy_to_cxcywh(box: BBox) -> BBox:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return (x1 + w / 2.0, y1 + h / 2.0, w, h)


def to_mot_row(frame_idx0: int, track_id: int, xyxy: BBox, conf: float) -> str:
    """0-based frame_idx, 0-based xyxy in -> a MOTChallenge tracks_mot.txt row out."""
    frame = frame_idx0 + MOT_FRAME_BASE
    x, y, w, h = xyxy_to_tlwh_1based(xyxy)
    return f"{frame},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1"


def parse_mot_row(line: str) -> tuple[int, int, BBox, float]:
    """A MOTChallenge row in -> (frame_idx0, track_id, xyxy 0-based, conf) out."""
    parts = line.strip().split(",")
    frame = int(float(parts[0]))
    track_id = int(float(parts[1]))
    x, y, w, h = (float(p) for p in parts[2:6])
    conf = float(parts[6]) if len(parts) > 6 else 1.0
    xyxy = tlwh_1based_to_xyxy((x, y, w, h))
    return frame - MOT_FRAME_BASE, track_id, xyxy, conf


def iou_similarity(a_tlwh: np.ndarray, b_tlwh: np.ndarray) -> np.ndarray:
    """IoU overlap matrix (higher = more overlap) between two arrays of tlwh boxes.
    Own implementation (not motmetrics) so preprocessing does not depend on the
    scoring library's internals. Shape (len(a), len(b))."""
    a = np.asarray(a_tlwh, dtype=np.float64)
    b = np.asarray(b_tlwh, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)

    ax1, ay1 = a[:, 0], a[:, 1]
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1 = b[:, 0], b[:, 1]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]

    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])

    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih

    area_a = np.clip(a[:, 2], 0, None) * np.clip(a[:, 3], 0, None)
    area_b = np.clip(b[:, 2], 0, None) * np.clip(b[:, 3], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter

    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou


def iou_distance(gt_tlwh, hyp_tlwh, *, max_iou: float = 0.5) -> np.ndarray:
    """The ONLY call site for a motmetrics-style distance matrix in this codebase.
    Wraps mm.distances.iou_matrix but pins max_iou explicitly, because that
    function's own signature default is max_iou=1.0 (which accepts every pair,
    silently disabling the distance gate) despite its docstring describing 0.5
    as the MOTChallenge convention.
    """
    import motmetrics as mm

    return mm.distances.iou_matrix(gt_tlwh, hyp_tlwh, max_iou=max_iou)
