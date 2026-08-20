"""PROJECT_ROOT and every derived path. No module elsewhere in this codebase
constructs a cwd-relative path -- everything is anchored here so behavior does
not change with the caller's working directory.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "Dataset"  # user-managed staging area for raw MOT17/UCF-Crime
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIGS_DIR = PROJECT_ROOT / "configs"
TRACKER_CONFIGS_DIR = CONFIGS_DIR / "trackers"
SOURCE_CONFIGS_DIR = CONFIGS_DIR / "sources"
ZONE_CONFIGS_DIR = CONFIGS_DIR / "zones"
DOCS_DIR = PROJECT_ROOT / "docs"
