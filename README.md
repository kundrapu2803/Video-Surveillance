# CCTV — Video Surveillance

A computer vision pipeline for processing CCTV footage, detecting and tracking people, and identifying events such as zone intrusion and loitering.

## What the project does

The pipeline takes a video or image sequence and:
1. Detects people in each frame.
2. Tracks people across frames and gives each person a unique ID.
3. Uses Re-ID to help keep the same ID when a person temporarily disappears.
4. Checks whether a person enters a configured zone.
5. Detects loitering when a person stays in a zone and remains mostly stationary for a configured amount of time.
6. Saves the results as annotated videos and structured event logs.
7. Provides a Streamlit dashboard for viewing runs and results.
8. Includes evaluation scripts for detection, tracking, and event metrics.


## Model choices

### Person detection - YOLO11n

The project uses YOLO11n through Ultralytics.

I chose the nano model mainly because this is a prototype and I wanted a reasonable balance between detection quality and speed. It is also practical for running the project on a CPU, although the current CPU performance is not real-time for high-resolution videos.

The project also contains:
1. models/yolo11n.pt
2. models/yolo11s.pt

The recorded runs use yolo11n.pt.

A larger detector could improve detection quality, but it would also increase processing time and hardware requirements.

### Tracking - BoT-SORT

The main tracker is BoT-SORT with Re-ID enabled.

The tracker configuration is: configs/trackers/botsort_reid.yaml

Re-ID is useful here because the assignment specifically asks the system to handle cases where a person temporarily leaves and then comes back into the scene.

The project also includes a ByteTrack configuration: configs/trackers/bytetrack.yaml

ByteTrack is a simpler alternative when speed is more important. I used BoT-SORT with Re-ID as the main option because maintaining identity is important for the event logic.


## Zone intrusion

Zones are defined using polygons in JSON files under: configs/zones/

For example:
1. configs/zones/mot17-04.json
2. configs/zones/mot17-09.json
3. configs/zones/ucf_generic.json

For every tracked person, the pipeline checks whether the reference point of the bounding box is inside a configured polygon.

When a person enters a restricted zone, an intrusion event can be generated.

The event log contains information such as: event type, zone, track ID, frame number, timestamp, bounding box, confidence

### Loitering

Loitering is based on how long a tracked person remains in a zone and how much they move during that period.

The configuration includes parameters such as:
1. loiter_seconds
2. stationary_window_seconds
3. stationary_radius
4. cooldown_seconds

This prevents a person from being marked as loitering just because they happen to remain in a zone for a few frames.

The exact thresholds can be changed through the zone configuration and supported command-line options.

## Configuration

There are separate configuration files for different parts of the system:

configs/
├── sources/
├── trackers/
└── zones/

For example:
configs/sources/mot17.yaml
configs/sources/ucf.yaml

configs/trackers/botsort_reid.yaml
configs/trackers/bytetrack.yaml

Zone files contain the polygon coordinates and event settings.

This makes it possible to change the restricted area or loitering threshold without changing the code.


## Installation

The project uses Python 3.11.

Create a virtual environment and install the dependencies:

'''pip install -r requirements.txt'''

You can check the environment with:

''' python scripts/verify_env.py'''

The exact dependencies are listed in requirements.txt.

## Running the project

The basic command is:

'''python run.py --video input.mp4 --zones zones.json --output results/'''

### MOT17

For MOT17 image sequences, pass the image directory as the video/source argument.

For example:

'''python run.py ^
  --video Dataset\mot17\MOT17-09-FRCNN\img1 ^
  --zones configs\zones\mot17-09.json ^
  --config configs\sources\mot17.yaml ^
  --output results\'''

### UCF-Crime

For a video file:

'''python run.py ^
  --video Dataset\ucf-crime\Fighting003_x264.mp4 ^
  --zones configs\zones\ucf_generic.json ^
  --config configs\sources\ucf.yaml ^
  --output results\'''

You can also limit the number of frames for a quick test:

'''python run.py ^
  --video Dataset\mot17\MOT17-09-FRCNN\img1 ^
  --zones configs\zones\mot17-09.json ^
  --config configs\sources\mot17.yaml ^
  --output results\ ^
  --max-frames 900'''

## Output

Each run creates a result directory containing the generated files.

Typical outputs include:

results/<run-id>/
├── run.json
├── effective_config.yaml
├── tracker_resolved.yaml
└── cameras/
    └── cam01/
        ├── annotated.mp4
        ├── detections.jsonl
        ├── detections.parquet
        ├── events.jsonl
        ├── events.csv
        ├── events_summary.json
        ├── alerts.jsonl
        ├── tracks_mot.txt
        └── zones.resolved.json

Depending on the options used, separate detection, tracking, intrusion and loitering videos can also be generated.

The annotated output shows the detected people, tracking IDs and configured zones.

## Streamlit dashboard

The project also has a Streamlit dashboard.

Start it with:

'''streamlit run dashboard/app.py'''

The dashboard is mainly used to inspect the results produced by the pipeline.

It can be used to select a previous run, view run information and metrics, inspect detections and tracking IDs, view zones and events, inspect event records, view annotated output, run the pipeline from the interface where supported, and access evaluation results.

## Evaluation

The project contains evaluation support for detection, tracking and events.

Run:

'''python scripts/evaluate.py --run <run-id>'''

For event evaluation:

'''python scripts/evaluate.py ^
  --run <run-id> ^
  --with-events ^
  --zones configs\zones\mot17-04.json'''

The evaluation includes metrics such as: Precision, Recall, F1, AP50, MOTA, MOTP, IDF1,, ID switches, and Event precision/recall/F1

The project uses MOT17 ground-truth annotations for tracking evaluation where available.

## Difficult Cases

1. **Occlusion:** The tracker can keep tracks when detections are temporarily missing.
2. **ID switches:** Crowded scenes can still cause tracking IDs to change.
3. **Camera movement:** Fixed zones are intended for fixed-camera views.
4. **Repeated alerts:** Cooldown settings help avoid duplicate alerts.

## Limitations

1. CPU processing is slow for high-resolution videos.
2. Detection and tracking can degrade with heavy occlusion or crowded scenes.
3. Loitering is rule-based and may produce false alerts.
4. Zones need to be configured for each camera.
5. Long disappearances may still result in an ID change.
6. Rule-based detection can sometimes flag unusual but legitimate behavior.


