"""The ONLY frame loop in the codebase. Ordering is load-bearing:
detect -> track -> smooth -> interpolate -> event -> write.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from cctv.bench.profiler import Profiler
from cctv.detect.base import DETECTOR_REGISTRY
from cctv.detect.tracker import PersonTracker
from cctv.events.engine import EventEngine, TrackerNotProducingIdsError
from cctv.geometry import iou_distance  # noqa: F401  (import surface kept for downstream eval reuse)
from cctv.interpolate import LinearInterpolator
from cctv.io.frame_source import LowResolutionSourceError, open_source
from cctv.io.writers import RunWriters
from cctv.io.zones import MovingCameraZoneError, ZoneSet
from cctv.render.annotator import draw_overlay, reset_trails
from cctv.render.video_writer import open_video_sink
from cctv.schema import SourceRecord, TrackRecord
from cctv.smoothing import EmaSmoother
from cctv.tracker_config import resolve_tracker_yaml

logger = logging.getLogger("cctv.pipeline")


def _guess_source_record(path: Path, *, fps_override, stride, allow_lowres) -> tuple[SourceRecord, object]:
    src = open_source(path, fps_override=fps_override, stride=stride, allow_lowres=allow_lowres)
    is_mot = "MOT17" in str(path).upper() or Path(getattr(src, "seqinfo_path", "") or "").exists()
    dataset = "mot17" if is_mot else "user"
    gt_path = getattr(src, "gt_path", None)
    rec = SourceRecord(
        id=src.name,
        dataset=dataset,
        kind="imgseq" if hasattr(src, "directory") else "video",
        path=str(Path(path).resolve()),
        frames=src.total_frames,
        fps=src.fps,
        fps_source=src.fps_source,
        width=src.width,
        height=src.height,
        frame_index_base=1 if dataset == "mot17" else 0,
        gt_path=gt_path,
        seqinfo_path=getattr(src, "seqinfo_path", None),
        camera_motion="static",
        zone_capable=True,
        profile="mot17" if dataset == "mot17" else "default",
        meta={},
    )
    return rec, src


def _config_sha256(eff_dict: dict) -> str:
    blob = json.dumps(eff_dict, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def run_source(
    *,
    video_path: str,
    zones_path: Optional[str],
    output_dir: str,
    eff,
    run_name: Optional[str] = None,
    argv: Optional[list[str]] = None,
) -> Path:
    from cctv.runtime import configure_runtime
    configure_runtime(threads=eff.threads)

    path = Path(video_path)
    rec, source = _guess_source_record(
        path, fps_override=eff.fps, stride=eff.stride, allow_lowres=eff.allow_lowres
    )

    max_frames = getattr(eff, "max_frames", None)

    model_tag = Path(eff.model).stem if eff.detector == "yolo" else "stub"
    tracker_name = eff.tracker
    run_id = run_name or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{rec.id}_{model_tag}_{tracker_name}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    writers = RunWriters(run_dir, camera_id="cam01")
    eff_dict = eff.as_dict()
    writers.write_manifest(
        run_id=run_id, argv=argv or [], sources=[rec],
        config_sha256=_config_sha256(eff_dict), status="running",
    )
    (run_dir / "effective_config.yaml").write_text(
        "\n".join(f"{k}: {v!r}" for k, v in sorted(eff_dict.items())), encoding="utf-8"
    )

    tracker_overrides = getattr(eff, "tracker_set", None)
    tracker_yaml_path = resolve_tracker_yaml(tracker_name, tracker_overrides, run_dir)
    import yaml
    resolved_tracker_cfg = yaml.safe_load(tracker_yaml_path.read_text(encoding="utf-8"))

    if zones_path:
        try:
            zone_set = ZoneSet.load(
                zones_path, frame_width=rec.width, frame_height=rec.height,
                camera_motion=rec.camera_motion,
                allow_moving_camera_zones=eff.allow_moving_camera_zones,
                cli_overrides=_zone_cli_overrides(eff),
            )
        except MovingCameraZoneError:
            raise
        zone_set.save_resolved(writers.camera_dir / "zones.resolved.json")
    else:
        zone_set = ZoneSet(zones=[], source_meta={
            "name": rec.id, "kind": rec.kind, "width": rec.width, "height": rec.height,
            "fps": rec.fps, "camera_motion": rec.camera_motion,
        }, digest="none")

    conf = min(0.05, resolved_tracker_cfg.get("track_low_thresh", 0.10))

    if eff.detector == "stub":
        import cctv.detect.stub_detector  # noqa: F401  (registers "stub")
        detector = DETECTOR_REGISTRY["stub"](script=getattr(eff, "stub_script", {}))
    else:
        import cctv.detect.yolo_detector  # noqa: F401  (registers "yolo"; lazy so stub runs never import torch)
        detector = DETECTOR_REGISTRY["yolo"](
            model=eff.model, tracker_yaml=str(tracker_yaml_path), device=eff.device,
            conf=conf, iou_nms=eff.iou_nms, half=eff.half,
        )
    tracker = PersonTracker(detector, source_id=rec.id, imgsz=eff.imgsz)
    tracker.begin_source()

    engine = EventEngine(
        zone_set, run_id=run_id, source_id=rec.id, camera_id="cam01",
        frame_width=rec.width, frame_height=rec.height, fps=rec.fps, stride=eff.stride,
        strict_id_rate=(rec.gt_path is not None),
    )
    smoother = EmaSmoother(alpha=0.6)
    interpolator = LinearInterpolator(eff.stride)
    reset_trails()
    prof = Profiler()

    video_sink = None
    codec, browser_playable = None, None
    detection_sink = tracking_sink = intrusion_sink = loitering_sink = None
    has_intrusion_zone = any(z.params.get("intrusion", {}).get("enabled", True) for z in zone_set.zones)
    has_loitering_zone = any(z.params.get("loitering", {}).get("enabled", False) for z in zone_set.zones)
    if not eff.no_video and eff.output_format != "none":
        out_size = _render_out_size(rec.width, rec.height)
        video_fps = rec.fps  # interpolation keeps the annotated video at source fps
        video_path_out = writers.camera_dir / "annotated.mp4"
        video_sink, codec, browser_playable = open_video_sink(video_path_out, video_fps, *out_size)

        if getattr(eff, "stage_videos", False):
            detection_sink, _, _ = open_video_sink(writers.camera_dir / "detection.mp4", video_fps, *out_size)
            tracking_sink, _, _ = open_video_sink(writers.camera_dir / "tracking.mp4", video_fps, *out_size)
            if has_intrusion_zone:
                intrusion_sink, _, _ = open_video_sink(writers.camera_dir / "intrusion.mp4", video_fps, *out_size)
            if has_loitering_zone:
                loitering_sink, _, _ = open_video_sink(writers.camera_dir / "loitering.mp4", video_fps, *out_size)

    n_frames_processed = 0
    frames_without_ids_running = 0
    last_ts = 0.0
    last_frame_idx = 0
    global_id_counter_max = 0

    for frame_idx, ts, bgr in source:
        if max_frames is not None and n_frames_processed >= max_frames:
            break

        prof.frame_boundary()
        with prof.stage("track"):
            dets, is_track = tracker.step(bgr)

        records = []
        for d in dets:
            gid = d["track_id"]  # local == global: tracker-internal Re-ID only (with_reid on the tracker yaml)
            if gid is not None:
                global_id_counter_max = max(global_id_counter_max, gid)
            records.append(TrackRecord(
                frame_idx=frame_idx, timestamp_s=ts, global_id=gid, track_id=gid,
                bbox_xyxy=d["bbox_xyxy"], smooth_bbox=None, conf=d["conf"], cls=d["cls"],
                source_id=rec.id, interpolated=False,
            ))

        with prof.stage("rules"):
            records = smoother.smooth(records)
            for out_idx, out_ts, out_records in interpolator.feed(frame_idx, ts, records):
                events = engine.update(frame_idx=out_idx, ts=out_ts, records=out_records)
                writers.on_frame(out_records)
                writers.on_events(events)

                if video_sink is not None:
                    with prof.stage("render"):
                        out_size = _render_out_size(rec.width, rec.height)
                        import cv2

                        def _fit(img):
                            if img.shape[1] != out_size[0] or img.shape[0] != out_size[1]:
                                return cv2.resize(img, out_size)
                            return img

                        annotated = draw_overlay(
                            bgr, out_records, engine.frame_overlay_state(), frame_idx=out_idx,
                            hud={"frame": out_idx, "t": f"{out_ts:.1f}s"},
                            show_trails=eff.show_trails,
                        )
                        video_sink.write(_fit(annotated))

                        if detection_sink is not None:
                            det_frame = draw_overlay(
                                bgr, out_records, None, frame_idx=out_idx,
                                hud={"stage": "detection", "frame": out_idx}, show_trails=False, show_ids=False,
                                update_trails=False,
                            )
                            detection_sink.write(_fit(det_frame))
                        if tracking_sink is not None:
                            trk_frame = draw_overlay(
                                bgr, out_records, None, frame_idx=out_idx,
                                hud={"stage": "tracking", "frame": out_idx}, show_trails=eff.show_trails, show_ids=True,
                                update_trails=False,
                            )
                            tracking_sink.write(_fit(trk_frame))

                        overlay_state = engine.frame_overlay_state()
                        if intrusion_sink is not None:
                            intrusion_zones = {zid: s for zid, s in overlay_state.items() if s.get("intrusion_enabled")}
                            intr_frame = draw_overlay(
                                bgr, out_records, intrusion_zones, frame_idx=out_idx,
                                hud={"stage": "zone_intrusion", "frame": out_idx}, show_trails=False, show_ids=True,
                                update_trails=False,
                            )
                            intrusion_sink.write(_fit(intr_frame))
                        if loitering_sink is not None:
                            loiter_zones = {zid: s for zid, s in overlay_state.items() if s.get("loitering_enabled")}
                            loiter_frame = draw_overlay(
                                bgr, out_records, loiter_zones, frame_idx=out_idx,
                                hud={"stage": "loitering", "frame": out_idx}, show_trails=eff.show_trails, show_ids=True,
                                update_trails=False,
                            )
                            loitering_sink.write(_fit(loiter_frame))

        n_frames_processed += 1
        last_ts, last_frame_idx = ts, frame_idx

        if n_frames_processed % 50 == 0:
            writers.heartbeat(
                first_frame_idx0=0, frames_processed=n_frames_processed,
                stride=eff.stride, fps=rec.fps, fps_source=rec.fps_source,
                width=rec.width, height=rec.height, source_id=rec.id,
                model=eff.model, backend=eff.backend, device=eff.device, imgsz=eff.imgsz,
                conf=conf, iou_nms=eff.iou_nms, classes=[0], tracker=tracker_name,
                tracker_params=resolved_tracker_cfg, with_reid=resolved_tracker_cfg.get("with_reid", False),
                n_detections=len(writers._detection_records), frames_without_ids=tracker.frames_without_ids,
            )

    final_events = engine.flush(last_frame_idx, last_ts)
    writers.on_events(final_events)

    if video_sink is not None:
        video_sink.close()
    if detection_sink is not None:
        detection_sink.close()
    if tracking_sink is not None:
        tracking_sink.close()
    if intrusion_sink is not None:
        intrusion_sink.close()
    if loitering_sink is not None:
        loitering_sink.close()

    never_triggered = engine.never_triggered_zones()

    writers.heartbeat(
        first_frame_idx0=0, frames_processed=n_frames_processed,
        stride=eff.stride, fps=rec.fps, fps_source=rec.fps_source,
        width=rec.width, height=rec.height, source_id=rec.id,
        model=eff.model, backend=eff.backend, device=eff.device, imgsz=eff.imgsz,
        conf=conf, iou_nms=eff.iou_nms, classes=[0], tracker=tracker_name,
        tracker_params=resolved_tracker_cfg, with_reid=resolved_tracker_cfg.get("with_reid", False),
        n_detections=len(writers._detection_records), n_local_ids=len(tracker.local_ids_seen),
        frames_without_ids=tracker.frames_without_ids,
        video_codec=codec, browser_playable=browser_playable,
        never_triggered_zones=sorted(never_triggered),
    )

    writers.run_json.setdefault("runtime", {}).update({"threads": eff.threads})
    writers.finalize(status="ok", bench=prof.report())

    if tracker.frames_without_ids_frac > 0.01:
        logger.warning(
            "frames_without_ids=%.1f%% exceeds the 1%% budget -- tracker is dropping ids frequently",
            tracker.frames_without_ids_frac * 100,
        )
    for zid in never_triggered:
        logger.warning("zone %s never triggered any event in this run", zid)
    if engine.low_id_rate_warning:
        logger.warning(engine.low_id_rate_warning)

    return run_dir


def _render_out_size(src_w: int, src_h: int) -> tuple[int, int]:
    scale = min(1.0, max(640, min(src_w, 1280)) / src_w) if src_w > 1280 else 1.0
    if src_w < 640:
        scale = 640 / src_w
    w, h = int(src_w * scale), int(src_h * scale)
    return (w - (w % 2), h - (h % 2))


def _zone_cli_overrides(eff) -> dict:
    overrides = {}
    for key in ("loiter_seconds", "enter_seconds", "exit_seconds", "cooldown_seconds",
                "merge_gap_seconds", "max_events_per_minute", "min_confidence"):
        val = getattr(eff, key, None)
        if val is not None:
            overrides[key] = val
    return overrides
