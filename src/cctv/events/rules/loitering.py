"""Loitering trigger condition: dwell AND stationarity, not dwell alone.

Pure dwell fires on every passer-by -- a pedestrian at ~1.2 m/s crosses a 20m
monitored region in ~17s, so any useful dwell threshold alone fires on
ordinary foot traffic. Requiring the stationarity gate (see rules/base.py) is
what turns this into an actual loitering signal.
"""
from __future__ import annotations

from cctv.events.rules.base import Stationarity


def should_fire(dwell_s: float, stat: Stationarity, params: dict) -> bool:
    return dwell_s >= params.get("loiter_seconds", 5.0) and stat == Stationarity.STATIONARY