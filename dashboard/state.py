"""Session-state ordering rules for this app (Streamlit raises
StreamlitAPIException if a widget's key is assigned AFTER that widget was
instantiated this run):

1. Read/consume any pending cross-widget mutation (e.g. "jump scrubber to
   frame N because an event row was clicked") at the TOP of the script,
   before the target widget is instantiated.
2. Only mutate a widget's session-state key from inside an on_click/on_select
   callback, never from the middle of the render body.
3. Guard event-selection -> scrubber wiring with a (view_id, tuple(rows))
   signature so re-running the script without a new selection doesn't
   re-trigger the jump and fight the user's own slider drags.
"""
from __future__ import annotations

import streamlit as st

FRAME_KEY = "scrubber_frame_idx"
PENDING_JUMP_KEY = "_pending_frame_jump"
LAST_SELECTION_SIG_KEY = "_last_event_selection_sig"

RUN_SELECT_KEY = "run_select"
PENDING_RUN_KEY = "_pending_run_select"


def request_run_select(run_id: str) -> None:
    """Called right after a dashboard-launched run finishes, so the sidebar
    Run selectbox opens on the new run instead of whatever was picked before."""
    st.session_state[PENDING_RUN_KEY] = run_id


def apply_pending_run_select() -> None:
    """Call once, at the very top of the script, before the Run selectbox
    widget is instantiated -- same ordering rule as apply_pending_jump()."""
    pending = st.session_state.pop(PENDING_RUN_KEY, None)
    if pending is not None:
        st.session_state[RUN_SELECT_KEY] = pending


def request_jump(frame_idx: int) -> None:
    st.session_state[PENDING_JUMP_KEY] = frame_idx


def apply_pending_jump() -> None:
    """Call once, at the very top of the script, before the scrubber widget
    is instantiated."""
    pending = st.session_state.pop(PENDING_JUMP_KEY, None)
    if pending is not None:
        st.session_state[FRAME_KEY] = pending


def selection_changed(view_id: str, rows: tuple) -> bool:
    sig = (view_id, rows)
    if st.session_state.get(LAST_SELECTION_SIG_KEY) == sig:
        return False
    st.session_state[LAST_SELECTION_SIG_KEY] = sig
    return True
