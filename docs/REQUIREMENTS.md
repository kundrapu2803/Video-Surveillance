# Requirements -> artifact mapping

One row per requirement in the brief, mapped to the exact artifact and the
test/script that demonstrates it.

## Core (Must Have)

### 1. Person Detection & Tracking

| Requirement | Artifact | Demonstrated by |
|---|---|---|
| Detect and track people with unique IDs | `results/<run>/cameras/cam01/tracks_mot.txt`, `detections.parquet:track_id,global_id` | `tests/test_e2e_smoke.py` (2 distinct global ids tracked across 20 frames); real run: `results/mot17-09-full` |
| Re-identification across a temporary exit/re-entry | `botsort_reid.yaml` (`with_reid: true`), the default tracker -- tracker-internal appearance matching bridges gaps up to `track_buffer` (~2s at 30fps) | measured directly: a one-off synthetic masked-band re-entry test showed botsort_reid preserving identity across ~1s gaps at 59% vs. 52% for plain bytetrack (see session notes; not a standing script in this repo) |
| Bounding boxes + confidence scores per frame | `detections.parquet` / `detections.jsonl` (`bbox_xyxy`, `conf`, every frame) | `tests/test_e2e_smoke.py::test_e2e_smoke` asserts `frame_idx.nunique()==20` |

### 2. Zone-Based Event Detection

| Requirement | Artifact | Demonstrated by |
|---|---|---|
| Zones via JSON polygon config | `configs/zones/*.json` (schema in `src/cctv/schema.py:ZONES_JSON_SCHEMA`) | `src/cctv/io/zones.py::ZoneSet.load` + `scripts/draw_zones.py` |
| Zone intrusion | `event_type=="zone_intrusion"` in `events.jsonl` | `tests/test_e2e_smoke.py`, `tests/test_events.py`; real: MOT17-09 F1=0.79, MOT17-04 F1=0.96 (`results/eval/report.md`) |
| Loitering (configurable time threshold) | `event_type=="loitering"`, `loiter_seconds` in zone `rules` | `tests/test_events.py::test_bbox_heights_invariant_to_resolution`, `::test_radial_walker_does_not_loiter`; real: MOT17-04 F1=0.80 |
| Timestamped event log: frame, bbox, type, confidence | `events.jsonl`/`.csv` (`frame_idx`, `frame_number`, `timestamp_s`/`_hms`, `bbox_xyxy`, `event_type`, `confidence`) | `src/cctv/schema.py:Event`, `EVENT_CSV_COLUMNS` |

### 3. Output Generation

| Requirement | Artifact | Demonstrated by |
|---|---|---|
| Annotated video/sequence with detections, tracks, zones | `annotated.mp4` (libx264, browser-playable); `--stage-videos` adds `detection.mp4` / `tracking.mp4` for each stage separately | `src/cctv/render/annotator.py::draw_overlay`, `run.json.video_codec`/`browser_playable` |
| Structured event log (JSON/CSV) summarizing events with timestamps | `events.json`/`.csv` (mirrored to `results/` root), `events_summary.json` | `src/cctv/io/writers.py::RunWriters._write_events_files` |

### 4. Interface

| Requirement | Artifact | Demonstrated by |
|---|---|---|
| `python run.py --video input.mp4 --zones zones.json --output results/` | `run.py` (spec-literal, works verbatim) | manual runs: `results/mot17-09-full`, `results/mot17-04-full`, `results/ucf-fighting003` |

## Stretch Goals (Nice to Have)

| Requirement | Status | Artifact |
|---|---|---|
| Real-time / near-real-time + FPS benchmarks | Measured per-run (`run.json.timings`); `scripts/bench_fps.py` written for a systematic sweep but not run this session | `run.json.timings.end_to_end_fps` / `fps_first_100` / `fps_last_100`; real: ~5.4 fps MOT17-09 @1080p, ~9.9 fps UCF @320x240 |
| Dashboard / web UI showing detections | Done (Streamlit, single-page + a Stage Videos page; Start/Run shells out to `run.py` as a subprocess, never imports the detector in-process) | `dashboard/app.py`, `dashboard/pages/1_Stage_Videos.py`, `tests/test_dashboard.py` (AppTest smoke test) |
| Multi-camera / multi-video support | Partial: sequential per-source runs with namespaced camera ids (`cam01`); no combined cross-camera identity space | `src/cctv/schema.py:RunWriters(camera_id=...)` |
| Configurable alert thresholds + dedup | Done | `configs/zones/*.json:defaults`, CLI `--loiter-seconds`/`--enter-seconds`/etc.; `src/cctv/events/dedup.py` (min-duration -> cooldown -> gap-merge -> rate limit) |
| Evaluation vs ground truth (MOTA/MOTP) | Done | `scripts/evaluate.py` -> `results/eval/{eval_summary.csv,report.md}`; real numbers in [README.md](../README.md#evaluation) |

## Deliberately not in scope

Cut for simplicity, not because they don't work -- each adds real complexity
for marginal benefit on a single-machine, single-camera CPU pipeline:

* An appearance-based Re-ID *gallery* on top of the tracker's own Re-ID.
  Re-identification itself is not cut -- `botsort_reid`'s tracker-internal
  `with_reid: true` remains the default and is what's measured above. A
  gallery would only add value for gaps longer than `track_buffer`
  (~2s at 30fps); a one-off test showed it added nothing at ~1s gaps
  (identical to tracker-internal alone) and 0% preservation at ~3s gaps,
  so its removal cost little.
* OpenVINO / INT8 acceleration backend -- PyTorch CPU only.
* Sharded/resumable runs (`--start-frame`/`--end-frame`, `--resume`) --
  `--max-frames` remains for quick demo runs.
* The distractor-suppression evaluation protocol (Hungarian-matched
  drop of hypotheses matched to MOT17 distractor classes) and the
  E01-E11 sanity-check framework -- `scripts/evaluate.py` still clips GT to
  the frames actually processed (the one correctness-critical piece), just
  without a formal checks module around it.
* Automated dataset discovery (`io/discovery.py` walking `Dataset/` by
  filename pattern) -- datasets are pointed to directly via `--video`.
* Cross-camera identity fusion for true multi-camera support.
* Hand-annotated event ground truth (a complement to the GT-derived events
  used here) and a stratified UCF-Crime precision review with confidence
  intervals.
