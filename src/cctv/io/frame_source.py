"""FrameSource protocol + concrete implementations.

Contract: iteration yields (frame_idx, timestamp_s, bgr_ndarray) where
frame_idx is 0-based and NATIVE under stride (0, 2, 4, ... for stride=2, so
timestamps stay wall-clock-true), and timestamp_s == frame_idx / fps always.
Indices are not guaranteed contiguous -- callers must accumulate time from
timestamp_s deltas, never from a running frame counter.
"""
from __future__ import annotations

import configparser
import re
from pathlib import Path
from typing import Iterator, Protocol

import numpy as np

MIN_WIDTH = 240  # below this, a mirror is almost certainly shipping downscaled thumbnails


class LowResolutionSourceError(Exception):
    pass


def _natural_key(path: Path):
    # MOT17's zero-padded names sort correctly either way; this exists so a
    # generic (non-zero-padded) frame dump does not get silently shuffled by
    # lexicographic sort, e.g. "10.jpg" before "2.jpg".
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p for p in parts]


class FrameSource(Protocol):
    fps: float
    fps_source: str
    width: int
    height: int
    total_frames: int
    name: str

    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]: ...


class ImageSequenceFrameSource:
    def __init__(
        self,
        directory: str | Path,
        *,
        fps_override: float | None = None,
        stride: int = 1,
        min_width: int = MIN_WIDTH,
        allow_lowres: bool = False,
    ):
        import cv2

        self.directory = Path(directory)
        self.name = self.directory.parent.name if self.directory.name == "img1" else self.directory.name

        exts = {".jpg", ".jpeg", ".png"}
        files = sorted(
            (p for p in self.directory.iterdir() if p.suffix.lower() in exts),
            key=_natural_key,
        )
        if not files:
            raise FileNotFoundError(f"no images found in {self.directory}")
        self._files = files
        self.total_frames = len(files)

        first = cv2.imread(str(files[0]))
        if first is None:
            raise IOError(f"failed to read first frame {files[0]}")
        self.height, self.width = first.shape[:2]

        if self.width < min_width and not allow_lowres:
            raise LowResolutionSourceError(
                f"{self.directory}: frames are {self.width}x{self.height}, below the "
                f"min_width={min_width} sanity gate. This mirror is very likely "
                "shipping downscaled thumbnails rather than full-resolution frames. "
                "Pass --allow-lowres to proceed anyway."
            )

        seqinfo_path = self.directory.parent / "seqinfo.ini"
        if seqinfo_path.exists():
            if fps_override is not None:
                raise ValueError(
                    f"{seqinfo_path} declares an authoritative frame rate; "
                    "--fps must not be passed for a source that has a seqinfo.ini "
                    f"(seqinfo says frameRate={self._read_seqinfo_fps(seqinfo_path)})."
                )
            self.fps = self._read_seqinfo_fps(seqinfo_path)
            self.fps_source = "seqinfo"
            self.seqinfo_path = str(seqinfo_path)
        elif fps_override is not None:
            self.fps = fps_override
            self.fps_source = "cli"
            self.seqinfo_path = None
        else:
            self.fps = 25.0
            self.fps_source = "default"
            self.seqinfo_path = None

        if not (1.0 <= self.fps <= 240.0):
            raise ValueError(f"resolved fps={self.fps} out of sane range [1, 240]")

        self.stride = max(1, stride)
        self.gt_path = str(self.directory.parent / "gt" / "gt.txt") if (self.directory.parent / "gt" / "gt.txt").exists() else None

    @staticmethod
    def _read_seqinfo_fps(seqinfo_path: Path) -> float:
        cfg = configparser.ConfigParser()
        cfg.read(seqinfo_path)
        return float(cfg["Sequence"]["frameRate"])

    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]:
        import cv2

        for frame_idx in range(0, self.total_frames, self.stride):
            path = self._files[frame_idx]
            bgr = cv2.imread(str(path))
            if bgr is None:
                continue
            yield frame_idx, frame_idx / self.fps, bgr


class VideoFrameSource:
    def __init__(
        self,
        path: str | Path,
        *,
        fps_override: float | None = None,
        stride: int = 1,
        min_width: int = MIN_WIDTH,
        allow_lowres: bool = False,
    ):
        import cv2

        self.path = Path(path)
        self.name = self.path.stem
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise IOError(f"failed to open video {self.path}")

        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        container_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.width < min_width and not allow_lowres:
            raise LowResolutionSourceError(
                f"{self.path}: {self.width}x{self.height}, below min_width={min_width}. "
                "Pass --allow-lowres to proceed anyway."
            )

        if fps_override is not None:
            self.fps = fps_override
            self.fps_source = "cli"
        elif container_fps and container_fps > 0:
            self.fps = container_fps
            self.fps_source = "container"
        else:
            self.fps = 25.0
            self.fps_source = "default"

        self.stride = max(1, stride)
        self.gt_path = None
        self.seqinfo_path = None

    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]:
        import cv2

        cap = cv2.VideoCapture(str(self.path))
        try:
            frame_idx = 0
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                if frame_idx % self.stride == 0:
                    yield frame_idx, frame_idx / self.fps, bgr
                frame_idx += 1
        finally:
            cap.release()


def open_source(
    path: str | Path,
    *,
    fps_override: float | None = None,
    stride: int = 1,
    allow_lowres: bool = False,
):
    p = Path(path)
    if p.is_dir():
        img1 = p / "img1"
        target = img1 if img1.is_dir() else p
        return ImageSequenceFrameSource(
            target, fps_override=fps_override, stride=stride, allow_lowres=allow_lowres
        )
    return VideoFrameSource(
        p, fps_override=fps_override, stride=stride, allow_lowres=allow_lowres
    )
