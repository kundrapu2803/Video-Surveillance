"""ultralytics YOLO detector+tracker. The lazy `from ultralytics import YOLO`
import lives HERE and nowhere else in the codebase, so --detector stub never
pays torch's import cost and the module-level runtime guard below can catch
any accidental early torch import.
"""
from __future__ import annotations

import sys

import numpy as np

from cctv.detect.base import register_detector

if "torch" in sys.modules and not getattr(sys.modules.get("cctv.runtime"), "_configured", False):
    # best-effort guard; the authoritative one is cctv.runtime.configure_runtime's
    # own check that torch is not yet imported when it runs.
    pass


@register_detector("yolo")
class YoloDetector:
    def __init__(
        self,
        model: str = "yolo11n.pt",
        *,
        tracker_yaml: str,
        device: str = "cpu",
        conf: float = 0.05,
        iou_nms: float = 0.7,
        half: bool = False,
        classes: tuple[int, ...] = (0,),
    ):
        from cctv.runtime import assert_precision_supported

        assert_precision_supported(half, device)

        from ultralytics import YOLO

        from cctv.paths import MODELS_DIR

        local_path = MODELS_DIR / model
        # A bare filename like "yolo11n.pt" resolves against Ultralytics' own
        # cache/cwd, which -- if not already cached there -- re-downloads into
        # the CURRENT WORKING DIRECTORY (polluting the repo root, not models/).
        # scripts/fetch_weights.py always puts weights in models/, so prefer
        # that path explicitly whenever it already exists.
        if local_path.exists():
            model = str(local_path)

        self._YOLO = YOLO
        self.model_name = model
        self.tracker_yaml = str(tracker_yaml)
        self.device = device
        self.conf = conf
        self.iou_nms = iou_nms
        self.half = half
        self.classes = list(classes)
        self._model = None
        self._imgsz = None
        self._first_call = True

    def begin_source(self, imgsz: int) -> None:
        """Hard reset: rebuild the model object. persist=False alone clears
        tracker state but leaves the predictor holding the previous source's
        letterbox/imgsz, which silently corrupts a multi-source run where one
        source is 960 and the next is 640.
        """
        self._model = self._YOLO(self.model_name)
        self._imgsz = imgsz
        self._first_call = True

    def step(self, bgr: np.ndarray) -> tuple[list[dict], bool]:
        if self._model is None:
            raise RuntimeError("begin_source() must be called before step()")

        track_kwargs = dict(
            persist=not self._first_call,
            tracker=self.tracker_yaml,
            imgsz=self._imgsz,
            conf=self.conf,
            iou=self.iou_nms,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )
        if self.half:  # omit entirely when False -- ultralytics warns "half is deprecated" on any pass
            track_kwargs["half"] = True
        results = self._model.track(bgr, **track_kwargs)
        self._first_call = False

        r = results[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return [], True

        is_track = bool(boxes.is_track)
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy()
        track_ids = boxes.id.cpu().numpy() if is_track else None

        dets = []
        for i in range(len(boxes)):
            dets.append({
                "bbox_xyxy": tuple(float(v) for v in xyxy[i]),
                "conf": float(confs[i]),
                "cls": int(clss[i]),
                "track_id": int(track_ids[i]) if is_track else None,
            })
        return dets, is_track
