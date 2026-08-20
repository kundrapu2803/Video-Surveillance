"""Scores a completed run against MOT17 ground truth: MOTChallenge tracking
metrics (MOTA/MOTP/IDF1/...), AP50 detection quality, and event P/R/F1
against GT-derived events. Protocol statement is always printed alongside
the numbers -- a subset run is never presented as if it were the full
sequence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from cctv.eval.detection_metrics import ap50, precision_recall_f1
from cctv.eval.event_gt import match_events, run_engine_over_gt
from cctv.eval.mot_gt import load_protocol_gt
from cctv.eval.tracking_metrics import accumulate_sequence, compute_summary, render_report
from cctv.io.mot import load_tracks_mot
from cctv.paths import RESULTS_DIR


def _zones_path_from_argv(argv: list[str]) -> str | None:
    for i, a in enumerate(argv):
        if a == "--zones" and i + 1 < len(argv):
            return argv[i + 1]
    return None


def find_run_dir(run_arg: str) -> Path:
    p = Path(run_arg)
    if p.exists():
        return p
    candidate = RESULTS_DIR / run_arg
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"no run directory found for {run_arg!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--sequences", nargs="*", default=None, help="restrict to these source ids")
    ap.add_argument("--with-events", action="store_true")
    ap.add_argument("--zones", default=None, help="zones.json for --with-events GT derivation")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_dir = find_run_dir(args.run)
    run_json = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    # frames_processed / first_frame_idx0 clip GT to what was actually processed --
    # without this, an unprocessed tail of GT counts as a miss and MOTA collapses
    # toward 0 with no error message, since a CPU-only box's runs are commonly subsets.
    for cam in run_json.get("cameras", {}).values():
        if "frames_processed" not in cam:
            print(f"WARNING: run.json camera entry has no frames_processed -- "
                  "GT cannot be reliably clipped to what was actually processed.", file=sys.stderr)

    report_lines = []
    accs = {}
    det_rows = []
    event_rows = []

    for src in run_json["sources"]:
        if args.sequences and src["id"] not in args.sequences:
            continue
        if not src.get("gt_path"):
            print(f"[skip] {src['id']}: no gt_path")
            continue

        cam = next(iter(run_json["cameras"].values()))
        first = cam.get("first_frame_idx0", 0)
        last = first + cam.get("frames_processed", 0) - 1
        subset = cam.get("frames_processed", 0) < src.get("frames", -1)

        gt = load_protocol_gt(src["gt_path"])
        gt = gt[(gt["frame_idx"] >= first) & (gt["frame_idx"] <= last)]

        cam_dir = run_dir / "cameras" / "cam01"
        hyp = load_tracks_mot(cam_dir / "tracks_mot.txt")
        hyp = hyp[(hyp["frame_idx"] >= first) & (hyp["frame_idx"] <= last)]

        acc = accumulate_sequence(gt, hyp, first, last)
        accs[src["id"]] = acc

        pr = precision_recall_f1(gt, hyp, conf_thresh=cam.get("conf", 0.05))
        ap_score = ap50(gt, hyp)
        det_rows.append({"sequence": src["id"], "subset": subset, **pr, "ap50": ap_score})

        if args.with_events:
            zones_path = args.zones or _zones_path_from_argv(run_json.get("argv", []))
            if not zones_path:
                print(f"[skip events] {src['id']}: no --zones path recorded in run.json argv and none passed via --zones")
                continue
            events_path = cam_dir / "events.jsonl"
            pred_events = [json.loads(l) for l in events_path.read_text(encoding="utf-8").splitlines()] if events_path.exists() else []
            gt_events = run_engine_over_gt(
                src["gt_path"], zones_path,
                source_id=src["id"], fps=src["fps"], frame_width=src["width"], frame_height=src["height"],
            )
            match = match_events(pred_events, gt_events)
            event_rows.append({"sequence": src["id"], **match["overall"]})
            report_lines.append(f"\n== {src['id']} event P/R/F1 (GT-derived) ==")
            for et, m in match.items():
                report_lines.append(f"  {et}: P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f} "
                                     f"tp={m['tp']} fp={m['fp']} fn={m['fn']} "
                                     f"mean_trigger_err_s={m['mean_trigger_error_s']}")

    if accs:
        summary = compute_summary(accs)
        report_lines.insert(0, render_report(summary))

    protocol_stmt = (
        "PROTOCOL: our own tracker + our own detector (yolo11n), scored against MOT17 "
        "ground truth with conf_flag!=0, class==1 (pedestrian), no visibility filter. "
        "Subset runs are frame-range-clipped and MARKED as subset, not full-sequence."
    )
    report_lines.append("\n" + protocol_stmt)

    out_dir = Path(args.out) if args.out else (RESULTS_DIR / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    if det_rows:
        pd.DataFrame(det_rows).to_csv(out_dir / "eval_summary.csv", index=False)
    if event_rows:
        pd.DataFrame(event_rows).to_csv(out_dir / "events_summary.csv", index=False)
    (out_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print("\n".join(report_lines))
    print(f"\nwritten to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
