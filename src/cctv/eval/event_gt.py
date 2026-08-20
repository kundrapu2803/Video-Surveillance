"""Derives event ground truth for free: run the SAME EventEngine over MOT17
gt.txt tracks (perfect boxes, perfect ids). This isolates event-logic
correctness from tracker error and costs zero labelling -- the only
quantitative claim available for the core zone/event requirement.
"""
from __future__ import annotations

import dataclasses

import pandas as pd

from cctv.events.engine import EventEngine
from cctv.io.mot import load_gt
from cctv.io.zones import ZoneSet
from cctv.schema import TrackRecord


def run_engine_over_gt(gt_path: str, zones_path: str, *, source_id: str, fps: float,
                        frame_width: int, frame_height: int) -> list[dict]:
    gt = load_gt(gt_path)
    gt = gt[(gt["conf_flag"] != 0) & (gt["class"] == 1)]

    zone_set = ZoneSet.load(zones_path, frame_width=frame_width, frame_height=frame_height,
                             camera_motion="static")
    engine = EventEngine(
        zone_set, run_id="gt-derived", source_id=source_id, camera_id="cam01",
        frame_width=frame_width, frame_height=frame_height, fps=fps,
    )

    all_events: list[dict] = []
    for frame_idx, frame in gt.groupby("frame_idx"):
        ts = frame_idx / fps
        records = [
            TrackRecord(
                frame_idx=int(frame_idx), timestamp_s=ts, global_id=int(row.id), track_id=int(row.id),
                bbox_xyxy=(row.x1, row.y1, row.x2, row.y2), smooth_bbox=None, conf=1.0, cls=0,
                source_id=source_id,
            )
            for row in frame.itertuples()
        ]
        events = engine.update(frame_idx=int(frame_idx), ts=ts, records=records)
        all_events.extend(dataclasses.asdict(e) for e in events)

    last_frame = int(gt["frame_idx"].max())
    final = engine.flush(last_frame, last_frame / fps)
    all_events.extend(dataclasses.asdict(e) for e in final)
    return [e for e in all_events if e["record_kind"] == "interval"]


def match_events(pred_events: list[dict], gt_events: list[dict], *, tolerance_s: float = 2.0) -> dict:
    """Matches predicted vs GT-derived intervals on (zone_id, event_type),
    global_id-agnostic (tracker ids don't align with GT ids 1:1), Hungarian
    over |Δtimestamp_s| within tolerance. Returns precision/recall/F1 and
    per-type breakdown plus mean trigger-time error for matched pairs.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    results = {}
    types = sorted({e["event_type"] for e in pred_events} | {e["event_type"] for e in gt_events})
    total_tp = total_fp = total_fn = 0
    all_time_errors = []

    for et in types:
        p = [e for e in pred_events if e["event_type"] == et]
        g = [e for e in gt_events if e["event_type"] == et]
        by_zone_p = {}
        by_zone_g = {}
        for e in p:
            by_zone_p.setdefault(e["zone_id"], []).append(e)
        for e in g:
            by_zone_g.setdefault(e["zone_id"], []).append(e)

        tp = fp = fn = 0
        time_errors = []
        zones = sorted(set(by_zone_p) | set(by_zone_g))
        for zid in zones:
            pz = by_zone_p.get(zid, [])
            gz = by_zone_g.get(zid, [])
            if not pz:
                fn += len(gz); continue
            if not gz:
                fp += len(pz); continue
            cost = np.full((len(pz), len(gz)), 1e6)
            for i, pe in enumerate(pz):
                for j, ge in enumerate(gz):
                    dt = abs(pe["timestamp_s"] - ge["timestamp_s"])
                    if dt <= tolerance_s:
                        cost[i, j] = dt
            r, c = linear_sum_assignment(cost)
            matched_p, matched_g = set(), set()
            for ri, ci in zip(r, c):
                if cost[ri, ci] < 1e6:
                    matched_p.add(ri); matched_g.add(ci)
                    time_errors.append(cost[ri, ci])
            tp += len(matched_p)
            fp += len(pz) - len(matched_p)
            fn += len(gz) - len(matched_g)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        results[et] = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1,
                        "mean_trigger_error_s": float(np.mean(time_errors)) if time_errors else None}
        total_tp += tp; total_fp += fp; total_fn += fn
        all_time_errors.extend(time_errors)

    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) else 0.0
    results["overall"] = {
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "precision": overall_p, "recall": overall_r, "f1": overall_f1,
        "mean_trigger_error_s": float(sum(all_time_errors) / len(all_time_errors)) if all_time_errors else None,
    }
    return results
