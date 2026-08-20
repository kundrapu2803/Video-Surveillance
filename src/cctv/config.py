"""5-layer config merge: CLI > --config YAML > per-source profile > default.yaml >
builtin defaults. argparse flags default to None (an "explicitly passed" sentinel)
so this module -- not argparse -- decides precedence.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

import yaml

from cctv.paths import CONFIGS_DIR, SOURCE_CONFIGS_DIR

BUILTIN_DEFAULTS: dict[str, Any] = {
    "detector": "yolo",
    "model": "yolo11n.pt",
    "backend": "pytorch",
    "device": "cpu",
    "imgsz": 960,
    "iou_nms": 0.7,
    "threads": 4,
    "half": False,
    "tracker": "botsort_reid",
    "stride": 1,
    "fps": None,
    "min_confidence": 0.35,
    "loiter_seconds": None,
    "enter_seconds": None,
    "exit_seconds": None,
    "cooldown_seconds": None,
    "merge_gap_seconds": None,
    "max_events_per_minute": None,
    "no_video": False,
    "dets_only": False,
    "save_mot": None,
    "output_format": "mp4",
    "show_trails": True,
    "allow_lowres": False,
    "zone_autoscale": False,
    "allow_moving_camera_zones": False,
    "seed": 0,
}


@dataclass
class EffectiveConfig:
    values: dict = field(default_factory=dict)

    def __getattr__(self, name: str):
        try:
            return self.values[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def as_dict(self) -> dict:
        return dict(self.values)


def _load_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_effective_config(cli_args: dict, source_profile: Optional[str] = None,
                              config_path: Optional[str] = None) -> EffectiveConfig:
    """Merge, lowest -> highest precedence:
    builtin defaults -> configs/default.yaml -> per-source profile ->
    --config YAML -> CLI (non-None values only).
    """
    merged: dict[str, Any] = dict(BUILTIN_DEFAULTS)
    merged.update(_load_yaml(CONFIGS_DIR / "default.yaml"))

    if source_profile:
        profile_path = SOURCE_CONFIGS_DIR / f"{source_profile}.yaml"
        merged.update(_load_yaml(profile_path))

    if config_path:
        merged.update(_load_yaml(config_path))

    for key, val in cli_args.items():
        if val is not None:
            merged[key] = val

    return EffectiveConfig(values=merged)
