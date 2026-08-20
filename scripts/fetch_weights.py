"""Downloads the model weights the pipeline needs into models/.
Safe to re-run: skips anything already present.
"""
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def fetch_yolo(name):
    from ultralytics import YOLO

    dest = MODELS_DIR / name
    if dest.exists():
        print(f"[skip] {name} already present")
        return
    print(f"[fetch] {name}")
    model = YOLO(name)  # ultralytics downloads to cwd/name if not cached
    src = Path(name)
    if src.exists() and src.resolve() != dest.resolve():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        src.replace(dest)
    elif not dest.exists():
        # ultralytics may have cached it elsewhere; locate via model.ckpt_path
        ckpt = getattr(model, "ckpt_path", None)
        if ckpt and Path(ckpt).exists():
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            Path(ckpt).replace(dest) if Path(ckpt).parent != MODELS_DIR else None


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    fetch_yolo("yolo11n.pt")
    fetch_yolo("yolo11s.pt")
    print("fetch_weights.py: done.")


if __name__ == "__main__":
    sys.exit(main())
