"""Resolve a named tracker + CLI overrides into a concrete YAML file.

Ultralytics yaml-loads tracker configs and silently ignores unrecognised
keys, so a typo'd override (e.g. track_buffer_frames instead of track_buffer)
produces a run that looks fine and is silently misconfigured. This module
validates every override key against an allow-list per tracker_type and
raises on anything unknown.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from cctv.paths import TRACKER_CONFIGS_DIR

_COMMON_KEYS = {
    "tracker_type", "track_high_thresh", "track_low_thresh", "new_track_thresh",
    "track_buffer", "match_thresh", "fuse_score",
}
_BOTSORT_KEYS = _COMMON_KEYS | {
    "gmc_method", "proximity_thresh", "appearance_thresh", "with_reid", "model",
}
ALLOWED_KEYS = {
    "bytetrack": _COMMON_KEYS,
    "botsort": _BOTSORT_KEYS,
}


def resolve_tracker_yaml(name: str, overrides: dict | None, out_dir: str | Path) -> Path:
    """name is a key into configs/trackers/ (without .yaml), e.g. 'botsort_reid'.
    Returns the path to the resolved YAML written into out_dir.
    """
    src = TRACKER_CONFIGS_DIR / f"{name}.yaml"
    if not src.exists():
        raise FileNotFoundError(f"unknown tracker config {name!r}: {src} does not exist")

    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    tracker_type = cfg["tracker_type"]
    allowed = ALLOWED_KEYS.get(tracker_type, _COMMON_KEYS)

    if overrides:
        unknown = set(overrides) - allowed
        if unknown:
            raise ValueError(
                f"tracker override key(s) {sorted(unknown)} not recognised for "
                f"tracker_type={tracker_type!r} (allowed: {sorted(allowed)}). "
                "Ultralytics silently ignores unknown keys instead of erroring, "
                "so this is a hard error here."
            )
        cfg.update(overrides)

    out_path = Path(out_dir) / "tracker_resolved.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out_path
