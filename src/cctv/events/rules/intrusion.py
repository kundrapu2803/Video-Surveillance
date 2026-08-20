"""Zone-intrusion trigger condition: pure function over episode state, no I/O."""
from __future__ import annotations


def should_confirm_entry(inside_run_s: float, params: dict) -> bool:
    return inside_run_s >= params.get("enter_seconds", 0.20)
