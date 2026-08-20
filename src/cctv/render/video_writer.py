"""FfmpegSink: libx264/yuv420p via imageio-ffmpeg's bundled static binary, so
the annotated video is browser-playable. (pip OpenCV ships no H.264 encoder;
its mp4v output plays in VLC but renders BLANK in a browser <video> tag --
that's why this path doesn't fall back to cv2.VideoWriter.)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def even(n: int) -> int:
    return n - (n & 1)


class FfmpegSink:
    codec = "libx264"
    browser_playable = True

    def __init__(self, path: str | Path, fps: float, width: int, height: int):
        import imageio_ffmpeg

        self.width = even(width)
        self.height = even(height)
        self._gen = imageio_ffmpeg.write_frames(
            str(path), (self.width, self.height), fps=fps,
            codec="libx264", pix_fmt_in="bgr24", pix_fmt_out="yuv420p",
            macro_block_size=None,
            output_params=["-crf", "23", "-preset", "veryfast", "-movflags", "+faststart"],
        )
        self._gen.send(None)  # mandatory priming call before the first frame

    def write(self, bgr: np.ndarray) -> None:
        if bgr.shape[1] != self.width or bgr.shape[0] != self.height:
            import cv2
            bgr = cv2.resize(bgr, (self.width, self.height))
        rgb = bgr[:, :, ::-1]
        try:
            self._gen.send(np.ascontiguousarray(rgb))
        except BrokenPipeError as e:
            raise RuntimeError(f"ffmpeg pipe broke while writing video: {e}") from e

    def close(self) -> None:
        self._gen.close()


def open_video_sink(path: str | Path, fps: float, width: int, height: int):
    """Returns (sink, codec, browser_playable)."""
    sink = FfmpegSink(path, fps, width, height)
    return sink, sink.codec, sink.browser_playable
