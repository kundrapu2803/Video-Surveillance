#!/usr/bin/env python
"""Spec-mandated entrypoint:
    python run.py --video input.mp4 --zones zones.json --output results/
"""
import sys
from pathlib import Path

if not sys.prefix.replace("/", "\\").endswith("envs\\cctv"):
    print(
        "wrong interpreter active (sys.prefix=%s).\n"
        "Activate the cctv conda env first:\n"
        "  & \"$env:USERPROFILE\\anaconda3\\shell\\condabin\\conda-hook.ps1\"; conda activate cctv"
        % sys.prefix,
        file=sys.stderr,
    )
    sys.exit(2)

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cctv.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
