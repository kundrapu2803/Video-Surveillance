"""Synthesises everything the offline test suite needs: no model, no data,
no network. A 20-frame 320x240 sequence at 10 fps where Person A walks
left-to-right into a restricted zone (tests zone_intrusion) and Person B
stands still inside a monitored zone (tests loitering) -- both from a
deterministic --detector stub script, so the whole pipeline is exercised
without ever importing torch.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

WIDTH, HEIGHT = 320, 240
N_FRAMES = 20
FPS = 10.0

# Person A: walks left -> right, crosses into the right-half zone around frame 10.
PERSON_A_W, PERSON_A_H = 30, 70
PERSON_A_Y1 = 100


def person_a_bbox(i: int) -> tuple[float, float, float, float]:
    x1 = 10 + i * 14
    return (float(x1), float(PERSON_A_Y1), float(x1 + PERSON_A_W), float(PERSON_A_Y1 + PERSON_A_H))


# Person B: stationary the whole clip, inside the bottom-left zone.
PERSON_B_BBOX = (20.0, 180.0, 60.0, 230.0)


@pytest.fixture
def stub_script() -> dict[int, list[dict]]:
    script = {}
    for i in range(N_FRAMES):
        script[i] = [
            {"bbox_xyxy": person_a_bbox(i), "conf": 0.92, "track_id": 1},
            {"bbox_xyxy": PERSON_B_BBOX, "conf": 0.88, "track_id": 2},
        ]
    return script


@pytest.fixture
def synth_source_dir(tmp_path: Path) -> Path:
    import cv2

    seq_dir = tmp_path / "SYNTH-01"
    img1 = seq_dir / "img1"
    img1.mkdir(parents=True)

    for i in range(N_FRAMES):
        frame = np.full((HEIGHT, WIDTH, 3), 40, dtype=np.uint8)
        cv2.imwrite(str(img1 / f"{i + 1:06d}.jpg"), frame)

    (seq_dir / "seqinfo.ini").write_text(
        "[Sequence]\n"
        f"name=SYNTH-01\nimDir=img1\nframeRate={int(FPS)}\nseqLength={N_FRAMES}\n"
        f"imWidth={WIDTH}\nimHeight={HEIGHT}\nimExt=.jpg\n",
        encoding="utf-8",
    )

    gt_dir = seq_dir / "gt"
    gt_dir.mkdir()
    lines = []
    for i in range(N_FRAMES):
        frame_1based = i + 1
        x1, y1, x2, y2 = person_a_bbox(i)
        lines.append(f"{frame_1based},1,{x1+1:.2f},{y1+1:.2f},{PERSON_A_W},{PERSON_A_H},1,1,1.0")
        bx1, by1, bx2, by2 = PERSON_B_BBOX
        lines.append(f"{frame_1based},2,{bx1+1:.2f},{by1+1:.2f},{bx2-bx1:.2f},{by2-by1:.2f},1,1,1.0")
    (gt_dir / "gt.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return img1


@pytest.fixture
def synth_zones_path(tmp_path: Path) -> Path:
    zones = {
        "schema_version": "1.0",
        "source": {"name": "SYNTH-01", "kind": "imgseq", "width": WIDTH, "height": HEIGHT,
                    "fps": FPS, "camera_motion": "static"},
        "defaults": {
            "normalized": True, "min_confidence": 0.30,
            "enter_seconds": 0.10, "exit_seconds": 0.20,
            "dwell_grace_seconds": 0.5, "track_timeout_seconds": 1.0,
            "loiter_seconds": 0.30, "stationary_window_seconds": 1.0,
            "stationary_radius": 0.5, "stationary_scale_ratio_max": 1.25,
            "cooldown_seconds": 5.0, "merge_gap_seconds": 1.0,
            "min_event_duration_s": 0.0, "max_events_per_minute": 60,
            "max_events_per_minute_per_zone": 60, "min_area_frac": 0.0001,
        },
        "zones": [
            {
                "id": "z_restricted", "name": "Restricted half", "zone_class": "restricted",
                "enabled": True, "priority": 1, "color": [0, 0, 255],
                "polygon": [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]],
                "rules": {"intrusion": {"enabled": True}, "loitering": {"enabled": False}},
            },
            {
                # y >= 0.85 (204px) so Person A's foot point (fixed y=170) never
                # enters this zone while crossing the frame -- only stationary
                # Person B (foot y=230) is inside it.
                "id": "z_monitored", "name": "Monitored corner", "zone_class": "monitored",
                "enabled": True, "priority": 2, "color": [0, 180, 255],
                "polygon": [[0.0, 0.85], [0.3, 0.85], [0.3, 1.0], [0.0, 1.0]],
                "rules": {"intrusion": {"enabled": False}, "loitering": {"enabled": True}},
            },
        ],
    }
    path = tmp_path / "zones.json"
    path.write_text(json.dumps(zones, indent=2), encoding="utf-8")
    return path
