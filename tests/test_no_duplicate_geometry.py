"""Fails the build if geometry/IoU-distance logic is duplicated outside
geometry.py. This is the enforcement mechanism for the plan's single most
important architectural rule: ONE conversion point, ONE iou_matrix call site.
"""
import re
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "cctv"
GEOMETRY_FILE = SRC_ROOT / "geometry.py"

DEF_XYXY_TO_TLWH = re.compile(r"^\s*def\s+xyxy_to_tlwh")
BARE_IOU_MATRIX_CALL = re.compile(r"mm\.distances\.iou_matrix\(")
XYWH_ATTR_ACCESS = re.compile(r"\.xywh\b")


def _all_source_files():
    return [p for p in SRC_ROOT.rglob("*.py")]


def test_no_second_xyxy_to_tlwh_definition():
    offenders = []
    for f in _all_source_files():
        if f == GEOMETRY_FILE:
            continue
        text = f.read_text(encoding="utf-8")
        if DEF_XYXY_TO_TLWH.search(text):
            offenders.append(str(f))
    assert not offenders, f"duplicate xyxy_to_tlwh definition(s) found: {offenders}"


def test_no_bare_iou_matrix_call_outside_geometry():
    offenders = []
    for f in _all_source_files():
        if f == GEOMETRY_FILE:
            continue
        text = f.read_text(encoding="utf-8")
        if BARE_IOU_MATRIX_CALL.search(text):
            offenders.append(str(f))
    assert not offenders, (
        f"bare mm.distances.iou_matrix(...) call outside geometry.py: {offenders} "
        "-- use cctv.geometry.iou_distance instead, which pins max_iou=0.5 "
        "explicitly (the library's own default is the permissive 1.0)."
    )


def test_no_xywh_attribute_access():
    offenders = []
    for f in _all_source_files():
        if f == GEOMETRY_FILE:
            continue  # its own docstring documents the trap by name
        text = f.read_text(encoding="utf-8")
        if XYWH_ATTR_ACCESS.search(text):
            offenders.append(str(f))
    assert not offenders, (
        f".xywh attribute access found in: {offenders} -- Ultralytics' "
        "Boxes.xywh is center-based; only .xyxy may be consumed."
    )
