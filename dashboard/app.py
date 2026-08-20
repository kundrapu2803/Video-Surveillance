"""Streamlit dashboard. Browsing a run never runs YOLO -- scrubbing re-draws
from detections.parquet + zones.resolved.json onto the raw source frame via
the SAME render.draw_overlay the pipeline's video writer uses, so the
dashboard and the annotated.mp4 can never show something different.

The one exception is the "Start / Run processing" button in the Video input
section: that shells out to run.py as a *subprocess* (exactly like a
terminal invocation), so this process itself still never imports torch or
calls the detector -- it only reads whatever artifacts the subprocess wrote,
same as browsing any other run.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import streamlit as st

from cctv.render.annotator import draw_overlay
from cctv.schema import TrackRecord
from dashboard import loaders, state

st.set_page_config(page_title="cctv", layout="wide")

state.apply_pending_jump()
state.apply_pending_run_select()

st.title("CCTV person tracking & event detection")


def _fmt_duration(seconds) -> str:
    if seconds is None:
        return "-"
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


# =============================================================================
# 1. Video input -- start a new run
# =============================================================================

existing_runs = loaders.list_runs()

with st.expander("Start a new run", expanded=not existing_runs):
    col_src, col_zone = st.columns(2)

    video_path = None
    config_path = None
    with col_src:
        st.markdown("**Source video**")
        source_mode = st.radio("Source", ["Use existing sample", "Upload video"],
                                horizontal=True, label_visibility="collapsed", key="src_mode")
        if source_mode == "Use existing sample":
            samples = loaders.list_sample_sources()
            if not samples:
                st.info("No sample sources found under Dataset/.")
            else:
                choice = st.selectbox("Sample source", samples, format_func=lambda s: s["label"])
                video_path, config_path = choice["path"], choice["config"]
        else:
            uploaded_video = st.file_uploader(
                "Upload a CCTV video (.mp4, .avi, .mov, .mkv)",
                type=["mp4", "avi", "mov", "mkv"],
            )
            if uploaded_video is not None:
                video_path = str(loaders.save_uploaded_file(uploaded_video))
                st.caption(f"saved to `{video_path}`")

    zones_path = None
    with col_zone:
        st.markdown("**Zone configuration (JSON)**")
        zone_mode = st.radio("Zones", ["Use existing config", "Upload zones JSON"],
                              horizontal=True, label_visibility="collapsed", key="zone_mode")
        if zone_mode == "Use existing config":
            zone_files = loaders.list_zone_configs()
            if not zone_files:
                st.info("No zone configs found under configs/zones/.")
            else:
                zones_choice = st.selectbox("Zone config", zone_files, format_func=lambda p: p.name)
                zones_path = str(zones_choice)
        else:
            uploaded_zones = st.file_uploader("Upload zones.json", type=["json"])
            if uploaded_zones is not None:
                zones_path = str(loaders.save_uploaded_file(uploaded_zones, subdir="zones"))
                st.caption(f"saved to `{zones_path}`")

    probe = None
    zone_set = None
    validation_ok = False
    if video_path and zones_path:
        try:
            probe = loaders.probe_source(video_path)
        except Exception as e:
            st.error(f"could not read source video: {e}")
        else:
            zone_set, zone_err = loaders.validate_zones(zones_path, probe["width"], probe["height"])
            if zone_err:
                st.error(f"zones config invalid: {zone_err}")
            else:
                validation_ok = True
                st.success(
                    f"{len(zone_set.zones)} zone(s) validated against "
                    f"{probe['width']}x{probe['height']}: "
                    + ", ".join(z.name for z in zone_set.zones)
                )

    st.markdown("**Run options**")
    o1, o2 = st.columns(2)
    with o1:
        tracker = st.selectbox("Tracker", ["botsort_reid", "bytetrack"])
    with o2:
        seconds_cap = st.number_input("Limit to first N seconds (0 = full source)",
                                       min_value=0, value=30, step=5)

    if st.button("Start / Run processing", type="primary", disabled=not validation_ok):
        argv = [
            "--video", video_path, "--zones", zones_path,
            "--output", str(loaders.RESULTS_DIR),
            "--tracker", tracker,
        ]
        if config_path:
            argv += ["--config", config_path]
        if seconds_cap > 0 and probe:
            argv += ["--max-frames", str(max(1, round(seconds_cap * probe["fps"])))]

        with st.spinner("Running pipeline... this can take a few minutes on CPU."):
            result = loaders.run_pipeline(argv)

        if result.returncode != 0:
            st.error(f"run.py failed (exit code {result.returncode})")
            with st.expander("stderr / stdout"):
                st.code((result.stderr or "") + "\n" + (result.stdout or ""))
        else:
            latest_txt = loaders.RESULTS_DIR / "latest_run.txt"
            new_run_id = latest_txt.read_text(encoding="utf-8").strip() if latest_txt.exists() else None
            st.success("Run complete.")
            if new_run_id:
                state.request_run_select(new_run_id)
            st.rerun()

# =============================================================================
# Run / camera selection
# =============================================================================

runs = loaders.list_runs()
if not runs:
    st.info("No runs found yet -- start one above.")
    st.stop()

run_ids = [r["run_id"] for r in runs]
run_id = st.sidebar.selectbox("Run", run_ids, key=state.RUN_SELECT_KEY)
run_json = loaders.load_run(run_id)
camera_id = st.sidebar.selectbox("Camera", list(run_json.get("cameras", {}).keys()) or ["cam01"])
cam = run_json["cameras"].get(camera_id, {})
cam_dir = loaders.cam_dir_for(run_id, camera_id)

st.session_state["dash_run_id"] = run_id
st.session_state["dash_camera_id"] = camera_id

st.sidebar.markdown(f"**status:** {run_json.get('status')}")
st.sidebar.markdown(f"**source:** {cam.get('source_id')}")
st.sidebar.markdown(f"**frames:** {cam.get('frames_processed')}")
st.sidebar.markdown(f"**model/tracker:** {cam.get('model')} / {cam.get('tracker')}")
if run_json.get("git_dirty"):
    st.sidebar.caption("(uncommitted changes at run time)")

events_mtime = (cam_dir / "events.jsonl").stat().st_mtime if (cam_dir / "events.jsonl").exists() else 0.0
events_df = loaders.load_events(str(cam_dir), events_mtime)

det_mtime = (cam_dir / "detections.parquet").stat().st_mtime if (cam_dir / "detections.parquet").exists() else 0.0
dets_df = loaders.load_detections(str(cam_dir), det_mtime)

zones_mtime = (cam_dir / "zones.resolved.json").stat().st_mtime if (cam_dir / "zones.resolved.json").exists() else 0.0
zones_resolved = loaders.load_zones_resolved(str(cam_dir), zones_mtime)

sources = run_json.get("sources", [])
src_meta = next((s for s in sources if s["id"] == cam.get("source_id")), None)

# =============================================================================
# 2. Live / summary metrics
# =============================================================================

persons_detected = cam.get("n_global_ids", cam.get("n_local_ids", "-"))

if not dets_df.empty:
    last_frame = dets_df["frame_idx"].max()
    active_ids = int(dets_df.loc[dets_df["frame_idx"] == last_frame, "global_id"].nunique())
else:
    active_ids = 0

n_intrusion = int((events_df["event_type"] == "zone_intrusion").sum()) if not events_df.empty else 0
n_loitering = int((events_df["event_type"] == "loitering").sum()) if not events_df.empty else 0
current_fps = run_json.get("timings", {}).get("end_to_end_fps", "-")
duration_s = (cam.get("frames_processed", 0) / src_meta["fps"]) if src_meta and src_meta.get("fps") else None

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Persons detected", persons_detected)
m2.metric("Active tracked IDs", active_ids)
m3.metric("Intrusion events", n_intrusion)
m4.metric("Loitering events", n_loitering)
m5.metric("Current FPS", f"{current_fps:.1f}" if isinstance(current_fps, (int, float)) else current_fps)
m6.metric("Video duration", _fmt_duration(duration_s))

# =============================================================================
# 3. Main annotated-video view (biggest section)
# =============================================================================

st.header("Annotated video")
view_mode = st.radio("View", ["Frame inspector", "Full video playback"], horizontal=True)

if view_mode == "Frame inspector":
    if dets_df.empty or src_meta is None:
        st.info("No detections to inspect.")
    else:
        max_frame = int(dets_df["frame_idx"].max())
        frame_idx = st.slider("Frame", 0, max_frame, key=state.FRAME_KEY)
        fps_for_ts = src_meta.get("fps") or 25.0
        st.caption(f"frame {frame_idx} / {max_frame} -- t={frame_idx / fps_for_ts:.2f}s")

        if not events_df.empty:
            end_col = events_df["end_frame_idx"].fillna(events_df["frame_idx"])
            active_events = events_df[(events_df["frame_idx"] <= frame_idx) & (end_col >= frame_idx)]
            for _, ev in active_events.iterrows():
                label = "ZONE INTRUSION" if ev["event_type"] == "zone_intrusion" else "LOITERING"
                st.warning(f"{label}: {ev.get('zone_name')} -- ID {ev.get('global_id')} "
                           f"(confidence {ev.get('confidence', 0):.2f})")

        source = loaders.open_frame_source(src_meta["path"])
        bgr = loaders.read_frame_at(source, frame_idx)

        if bgr is None:
            st.error(f"could not read frame {frame_idx} from {src_meta['path']}")
        else:
            frame_dets = dets_df[dets_df["frame_idx"] == frame_idx]
            records = [
                TrackRecord(
                    frame_idx=int(r.frame_idx), timestamp_s=float(r.timestamp_s),
                    global_id=(None if pd.isna(r.global_id) else int(r.global_id)),
                    track_id=(None if pd.isna(r.track_id) else int(r.track_id)),
                    bbox_xyxy=tuple(r.bbox_xyxy), smooth_bbox=None, conf=float(r.conf), cls=0,
                    source_id=r.source_id,
                )
                for r in frame_dets.itertuples()
            ]

            import cv2
            overlay_state = {
                z["id"]: {
                    "polygon_px": np.array(z["polygon"], dtype=np.float32).reshape(-1, 1, 2),
                    "color": tuple(z["color"]), "name": z["name"], "breached": False,
                }
                for z in zones_resolved.get("zones", [])
            }
            annotated = draw_overlay(bgr, records, overlay_state, frame_idx=frame_idx, show_trails=False)

            gt_path = src_meta.get("gt_path")
            if gt_path and Path(gt_path).exists() and st.checkbox("Show GT overlay (green)", value=True):
                from cctv.io.mot import load_gt
                gt = load_gt(gt_path)
                gt_frame = gt[(gt["frame_idx"] == frame_idx) & (gt["conf_flag"] != 0) & (gt["class"] == 1)]
                for r in gt_frame.itertuples():
                    cv2.rectangle(annotated, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), (0, 255, 0), 1)

            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width="stretch")
            st.dataframe(
                frame_dets[[c for c in ["global_id", "track_id", "conf", "reid_event"] if c in frame_dets.columns]],
                hide_index=True,
            )
else:
    video_file = cam_dir / "annotated.mp4"
    if not video_file.exists():
        st.info("This run was produced with --no-video.")
    elif cam.get("video_codec") == "mp4v":
        st.warning(
            "This run's video was encoded with the mp4v fallback codec, which "
            "plays in VLC but renders BLANK in a browser <video> tag. Re-run "
            "without --no-video and check that imageio-ffmpeg is installed."
        )
    else:
        st.video(str(video_file))

# =============================================================================
# 4. Zone configuration
# =============================================================================

st.header("Zone configuration")
zones_list = zones_resolved.get("zones", [])
if not zones_list:
    st.info("No zones configured for this run.")
else:
    for z in zones_list:
        poly = [[round(x, 1), round(y, 1)] for x, y in z.get("polygon", [])]
        intrusion = z.get("rules", {}).get("intrusion", {})
        loitering = z.get("rules", {}).get("loitering", {})
        with st.expander(f"{z.get('name', z.get('id'))}  ({z.get('zone_class')})"):
            c1, c2 = st.columns(2)
            c1.markdown(f"**Intrusion:** enabled={intrusion.get('enabled', False)}")
            if intrusion.get("enabled"):
                c1.caption(f"enter_seconds={intrusion.get('enter_seconds')}, "
                           f"min_confidence={intrusion.get('min_confidence')}, "
                           f"cooldown_seconds={intrusion.get('cooldown_seconds')}")
            c2.markdown(f"**Loitering:** enabled={loitering.get('enabled', False)}")
            if loitering.get("enabled"):
                c2.caption(f"loiter_seconds={loitering.get('loiter_seconds')}, "
                           f"stationary_radius={loitering.get('stationary_radius')}")
            st.caption(f"polygon (pixel space): {poly}")
            with st.expander("Full resolved rule parameters"):
                st.json(z.get("rules", {}))

# =============================================================================
# 5. Event panel + filtering
# =============================================================================

st.header("Events")
if events_df.empty:
    st.info("No events in this run.")
else:
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        type_filter = st.multiselect("Event type", sorted(events_df["event_type"].unique()),
                                      default=list(events_df["event_type"].unique()))
    with f2:
        zone_filter = st.multiselect("Zone", sorted(events_df["zone_id"].unique()),
                                      default=list(events_df["zone_id"].unique()))
    with f3:
        track_filter = st.multiselect("Track ID", sorted(events_df["global_id"].unique()),
                                       default=list(events_df["global_id"].unique()))
    with f4:
        min_conf = st.slider("Min confidence", 0.0, 1.0, 0.0, 0.05)

    t_min, t_max = float(events_df["timestamp_s"].min()), float(events_df["timestamp_s"].max())
    if t_min < t_max:
        time_range = st.slider("Time range (s)", t_min, t_max, (t_min, t_max))
    else:
        time_range = (t_min, t_max)

    filtered = events_df[
        events_df["event_type"].isin(type_filter) & events_df["zone_id"].isin(zone_filter)
        & events_df["global_id"].isin(track_filter)
        & (events_df["confidence"] >= min_conf)
        & events_df["timestamp_s"].between(time_range[0], time_range[1])
    ].sort_values("timestamp_s")

    TRUNC = 5000
    if len(filtered) > TRUNC:
        st.warning(f"{len(filtered)} events match -- showing first {TRUNC}. Download the full CSV below.")
        filtered_shown = filtered.head(TRUNC)
    else:
        filtered_shown = filtered

    display_cols = ["event_id", "event_type", "zone_name", "global_id", "frame_idx", "timestamp_hms",
                     "duration_s", "confidence", "status"]
    display_cols = [c for c in display_cols if c in filtered_shown.columns]

    selection = st.dataframe(
        filtered_shown[display_cols], hide_index=True,
        on_select="rerun", selection_mode="single-row", key="events_table",
    )
    rows = tuple(selection.selection.rows) if hasattr(selection, "selection") else ()
    if rows and state.selection_changed("events", rows):
        picked = filtered_shown.iloc[rows[0]]
        state.request_jump(int(picked["frame_idx"]))
        st.rerun()

# =============================================================================
# 6. Output / download section
# =============================================================================

st.header("Output & downloads")
d1, d2, d3, d4 = st.columns(4)
video_file = cam_dir / "annotated.mp4"
events_csv = cam_dir / "events.csv"
events_json = cam_dir / "events.jsonl"

with d1:
    if video_file.exists():
        st.download_button("Annotated video (.mp4)", data=video_file.read_bytes(),
                            file_name="annotated.mp4", mime="video/mp4")
    else:
        st.caption("no video for this run")
with d2:
    if events_csv.exists():
        st.download_button("Events (.csv)", data=events_csv.read_bytes(),
                            file_name="events.csv", mime="text/csv")
    else:
        st.caption("no events.csv")
with d3:
    if events_json.exists():
        st.download_button("Events (.jsonl)", data=events_json.read_bytes(),
                            file_name="events.jsonl", mime="application/json")
    else:
        st.caption("no events.jsonl")
with d4:
    st.download_button("Full run (.zip)", data=loaders.zip_run_dir(run_id, camera_id),
                        file_name=f"{run_id}.zip", mime="application/zip")

