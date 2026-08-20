"""RunWriters: every artifact under results/<run_id>/ goes through this one
class. Streaming where it matters (alerts.jsonl, run.json heartbeat) so a
killed run is still evaluable over the frames it completed; buffered + one
flush at finalize() for detections.parquet and tracks_mot.txt (need the full
set to sort).
"""
from __future__ import annotations

import csv
import dataclasses
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from cctv.geometry import to_mot_row
from cctv.schema import EVENT_CSV_COLUMNS, SourceRecord, TrackRecord


def _json_default(o):
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    if isinstance(o, (set, frozenset)):
        return list(o)
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    return str(o)


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5
        )
        return bool(out.stdout.strip()) if out.returncode == 0 else False
    except Exception:
        return False


class RunWriters:
    def __init__(self, run_dir: str | Path, camera_id: str = "cam01"):
        self.run_dir = Path(run_dir)
        self.camera_id = camera_id
        self.camera_dir = self.run_dir / "cameras" / camera_id
        self.camera_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)

        self._detection_records: list[dict] = []
        self._mot_rows: list[tuple[int, int, str]] = []  # (frame, id, row) for sort
        self._interval_events: dict[str, dict] = {}  # keyed by event_id -- gap-merge overwrites in place
        self._sequence = 0

        self._detections_jsonl_fh = open(
            self.camera_dir / "detections.jsonl", "w", encoding="utf-8"
        )
        self._alerts_jsonl_fh = open(
            self.camera_dir / "alerts.jsonl", "w", encoding="utf-8"
        )

        self.run_json: dict[str, Any] = {}

    # -- manifest -----------------------------------------------------------

    def write_manifest(self, *, run_id: str, argv: list[str], sources: list[SourceRecord],
                        config_sha256: str, status: str = "running") -> None:
        self.run_json = {
            "schema_version": 1,
            "run_id": run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_sha": _git_sha(),
            "git_dirty": _git_dirty(),
            "argv": argv,
            "config_sha256": config_sha256,
            "status": status,
            "sources": [dataclasses.asdict(s) for s in sources],
            "cameras": {},
            "runtime": {},
            "timings": {},
        }
        self._write_run_json()

    def set_camera_meta(self, meta: dict) -> None:
        self.run_json.setdefault("cameras", {})[self.camera_id] = meta
        self._write_run_json()

    def heartbeat(self, **camera_updates) -> None:
        cam = self.run_json.setdefault("cameras", {}).setdefault(self.camera_id, {})
        cam.update(camera_updates)
        self._write_run_json()

    def _write_run_json(self) -> None:
        (self.run_dir / "run.json").write_text(
            json.dumps(self.run_json, indent=2, default=_json_default, ensure_ascii=False),
            encoding="utf-8",
        )

    # -- per-frame ------------------------------------------------------------

    def on_frame(self, records: list[TrackRecord]) -> None:
        for rec in records:
            d = dataclasses.asdict(rec)
            self._detection_records.append(d)
            self._detections_jsonl_fh.write(json.dumps(d, default=_json_default, ensure_ascii=False) + "\n")

            if rec.track_id is not None:
                row = to_mot_row(rec.frame_idx, rec.track_id, rec.bbox_xyxy, rec.conf)
                self._mot_rows.append((rec.frame_idx, rec.track_id, row))

    def on_events(self, events: list) -> None:
        for ev in events:
            d = dataclasses.asdict(ev)
            if ev.record_kind == "alert":
                self._alerts_jsonl_fh.write(json.dumps(d, default=_json_default, ensure_ascii=False) + "\n")
                self._alerts_jsonl_fh.flush()
            else:
                self._interval_events[ev.event_id] = d

    # -- finalize -------------------------------------------------------------

    def finalize(self, *, status: str, bench: Optional[dict] = None) -> None:
        self._detections_jsonl_fh.close()
        self._alerts_jsonl_fh.close()

        self._write_detections_parquet()
        self._write_mot_txt()
        self._write_events_files()

        if bench:
            self.run_json.setdefault("timings", {}).update(bench)
        self.run_json["status"] = status
        self._write_run_json()

        self._mirror_to_root()

    def _write_detections_parquet(self) -> None:
        import pandas as pd

        df = pd.DataFrame(self._detection_records)
        if df.empty:
            df = pd.DataFrame(columns=[f.name for f in dataclasses.fields(TrackRecord)])
        for col in ("bbox_xyxy", "smooth_bbox", "zone_ids"):
            if col in df.columns:
                df[col] = df[col].apply(lambda v: json.dumps(v, default=_json_default) if v is not None else None)
        df.to_parquet(self.camera_dir / "detections.parquet", index=False)

    def _write_mot_txt(self) -> None:
        rows_sorted = sorted(self._mot_rows, key=lambda t: (t[0], t[1]))
        (self.camera_dir / "tracks_mot.txt").write_text(
            "\n".join(r[2] for r in rows_sorted) + ("\n" if rows_sorted else ""), encoding="utf-8"
        )

    def _write_events_files(self) -> None:
        rows = sorted(self._interval_events.values(), key=lambda d: d["timestamp_s"])

        events_jsonl = self.camera_dir / "events.jsonl"
        with open(events_jsonl, "w", encoding="utf-8") as f:
            for d in rows:
                f.write(json.dumps(d, default=_json_default, ensure_ascii=False) + "\n")

        events_csv = self.camera_dir / "events.csv"
        with open(events_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=EVENT_CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for d in rows:
                writer.writerow(d)

        by_type: dict[str, int] = {}
        by_zone: dict[str, int] = {}
        for d in rows:
            by_type[d["event_type"]] = by_type.get(d["event_type"], 0) + 1
            by_zone[d["zone_id"]] = by_zone.get(d["zone_id"], 0) + 1
        summary = {
            "n_events": len(rows),
            "by_type": by_type,
            "by_zone": by_zone,
        }
        (self.camera_dir / "events_summary.json").write_text(
            json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
        )

    def _mirror_to_root(self) -> None:
        root = self.run_dir.parent
        if root == self.run_dir:
            return
        try:
            root.mkdir(parents=True, exist_ok=True)
            rows = sorted(self._interval_events.values(), key=lambda d: d["timestamp_s"])
            (root / "events.json").write_text(
                json.dumps(rows, indent=2, default=_json_default, ensure_ascii=False),
                encoding="utf-8",
            )
            import shutil
            shutil.copyfile(self.camera_dir / "events.csv", root / "events.csv")
            summary = json.loads((self.camera_dir / "events_summary.json").read_text(encoding="utf-8"))
            (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            (root / "latest_run.txt").write_text(self.run_dir.name, encoding="utf-8")
            self._update_index(root)
        except OSError:
            pass

    def _update_index(self, root: Path) -> None:
        index_path = root / "index.json"
        index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else []
        index = [e for e in index if e.get("run_id") != self.run_dir.name]
        index.insert(0, {
            "run_id": self.run_dir.name,
            "status": self.run_json.get("status"),
            "created_utc": self.run_json.get("created_utc"),
        })
        index_path.write_text(json.dumps(index[:200], indent=2), encoding="utf-8")
