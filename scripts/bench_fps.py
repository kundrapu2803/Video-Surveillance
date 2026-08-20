"""THE one benchmark: a handful of hand-picked cells (not a combinatorial
sweep), each timing decode+infer end-to-end over --n-frames real frames on
an otherwise-idle machine. Reports first-half vs second-half FPS so 15W
thermal decay is a reported number, not a hidden confound.

Each cell runs in its OWN subprocess: OMP_NUM_THREADS/MKL_NUM_THREADS only
take effect if set before torch's native libraries first initialize inside a
process, so a threads-sweep sharing one interpreter would silently measure
the FIRST cell's thread count for every subsequent cell.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _run_one_cell(video_path: str, *, imgsz: int, stride: int, n_frames: int,
                   tracker: str = "botsort_reid", device: str = "cpu") -> dict:
    """Runs INSIDE a cell's own subprocess (see __main__ --cell mode below)."""
    import tempfile

    from cctv.detect.base import DETECTOR_REGISTRY
    import cctv.detect.yolo_detector  # noqa: F401  (registers "yolo")
    from cctv.io.frame_source import open_source
    from cctv.tracker_config import resolve_tracker_yaml

    source = open_source(video_path, stride=stride, allow_lowres=True)
    with tempfile.TemporaryDirectory() as tmp:
        tracker_yaml = resolve_tracker_yaml(tracker, None, tmp)
        detector = DETECTOR_REGISTRY["yolo"](
            model="yolo11n.pt", tracker_yaml=str(tracker_yaml), device=device, conf=0.05,
        )
        detector.begin_source(imgsz)

        times = []
        n = 0
        for frame_idx, ts, bgr in source:
            t0 = time.perf_counter()
            detector.step(bgr)
            times.append(time.perf_counter() - t0)
            n += 1
            if n >= n_frames:
                break

    warmup = min(10, len(times) // 4)
    usable = times[warmup:]
    if not usable:
        return {"fps_end_to_end": 0.0, "fps_first_half": 0.0, "fps_second_half": 0.0, "n_frames": n}

    half = len(usable) // 2
    fps_all = 1.0 / (sum(usable) / len(usable))
    fps_first = 1.0 / (sum(usable[:half]) / half) if half else fps_all
    fps_second = 1.0 / (sum(usable[half:]) / (len(usable) - half)) if len(usable) - half else fps_all
    return {"fps_end_to_end": round(fps_all, 3), "fps_first_half": round(fps_first, 3),
            "fps_second_half": round(fps_second, 3), "n_frames": n}


def _spawn_cell(python_exe: str, video_path: str, cell: dict, n_frames: int) -> dict:
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(cell["threads"])
    env["MKL_NUM_THREADS"] = str(cell["threads"])
    env["OPENBLAS_NUM_THREADS"] = str(cell["threads"])
    args = [python_exe, __file__, "--cell", "--video", video_path,
            "--imgsz", str(cell["imgsz"]), "--stride", str(cell["stride"]),
            "--n-frames", str(n_frames)]
    out = subprocess.run(args, env=env, capture_output=True, text=True, timeout=600)
    if out.returncode != 0:
        return {"error": out.stderr[-500:]}
    return json.loads(out.stdout.strip().splitlines()[-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--n-frames", type=int, default=100)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cell", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--imgsz", type=int, default=960, help=argparse.SUPPRESS)
    ap.add_argument("--stride", type=int, default=1, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.cell:
        result = _run_one_cell(args.video, imgsz=args.imgsz, stride=args.stride, n_frames=args.n_frames)
        print(json.dumps(result))
        return 0

    import pandas as pd
    from cctv.paths import RESULTS_DIR

    cells = [
        {"imgsz": 640, "threads": 4, "stride": 1},
        {"imgsz": 960, "threads": 4, "stride": 1},
        {"imgsz": 1280, "threads": 4, "stride": 1},
        {"imgsz": 960, "threads": 4, "stride": 2},
        {"imgsz": 960, "threads": 2, "stride": 1},
        {"imgsz": 960, "threads": 6, "stride": 1},
    ]

    rows = []
    for cell in cells:
        print(f"[bench] {cell}")
        result = _spawn_cell(sys.executable, args.video, cell, args.n_frames)
        print(f"  -> {result}")
        rows.append({**cell, **result})

    df = pd.DataFrame(rows)
    out_dir = Path(args.out) if args.out else (RESULTS_DIR / "bench")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "fps.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nwritten to {out_dir / 'fps.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
