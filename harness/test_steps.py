"""``lifts`` is pure, so it is tested without a GPU, a venv or a model.

    uv run pytest harness/test_steps.py

The detection half of ``steps`` takes a trace and returns events. Keeping it
free of mujoco is what makes these assertions possible under cdx-rl's own
interpreter, which deliberately has no mujoco pin — and every one of them
encodes a bug that was actually made.

**Every check must be able to fail** (DESIGN's rule 6, hazard 18). Each test
below asserts both directions where there are two.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _steps  # noqa: E402

DOWN = {"down": {"f": True}, "xy": {"f": [0.0, 0.0]}, "z": {"f": 0.0}}


def trace(*segments):
    """``('down', 3), ('up', 4, 40.0)`` → a frame list."""

    frames = []
    x = 0.0
    for segment in segments:
        kind, count = segment[0], segment[1]
        if kind == "up":
            x = segment[2] if len(segment) > 2 else x
            frames += [{"down": {"f": False}, "xy": {"f": [x, 0.0]},
                        "z": {"f": 8.0}}] * count
        else:
            x = segment[2] if len(segment) > 2 else x
            frames += [{"down": {"f": True}, "xy": {"f": [x, 0.0]},
                        "z": {"f": 0.0}}] * count
    return frames


def test_threshold_is_a_duration_not_a_step_count():
    """The bug experiment 003's control-rate change would have caused.

    30 ms is 3 control steps at 100 Hz and 2 at 50 Hz. The original constant
    was the literal ``3``, so halving the control rate would silently have
    doubled the criterion to 60 ms and made every step count incomparable
    with the baseline it was measured against.
    """

    assert _steps.min_airborne_steps(0.010, 0.030) == 3   # 100 Hz
    assert _steps.min_airborne_steps(0.020, 0.030) == 2   # 50 Hz
    assert _steps.min_airborne_steps(0.002, 0.030) == 15  # solver rate
    # Never zero, however coarse the rate.
    assert _steps.min_airborne_steps(1.0, 0.030) == 1


def test_a_step_is_a_lift_that_went_somewhere():
    events = _steps.lifts(trace(("down", 3), ("up", 4), ("down", 3, 40.0)),
                          0.02, step_mm=10.0, min_airborne_s=0.030)
    assert len(events) == 1
    assert events[0]["step"] is True
    assert round(events[0]["travel_mm"], 1) == 40.0

    # …and the same lift that lands where it left is NOT a step.
    events = _steps.lifts(trace(("down", 3), ("up", 4), ("down", 3, 2.0)),
                          0.02, step_mm=10.0, min_airborne_s=0.030)
    assert len(events) == 1
    assert events[0]["step"] is False


def test_the_reset_drop_is_not_a_step():
    """The calibration failure that found this: a 30 mm "step" at 0.00 s in
    four episodes of four, which was the reset variation's lift-and-drop.

    A foot cannot step off a floor it has not reached, so an airborne
    interval before the foot has ever been planted is not an event at all.
    """

    airborne_first = trace(("up", 5, 30.0), ("down", 5, 0.0), ("up", 4),
                           ("down", 3, 40.0))
    events = _steps.lifts(airborne_first, 0.02, step_mm=10.0,
                          min_airborne_s=0.030)
    assert len(events) == 1, "the opening airborne stretch must not count"
    assert events[0]["takeoff_s"] > 0.1


def test_too_brief_a_lift_is_contact_chatter():
    one_frame = trace(("down", 3), ("up", 1), ("down", 3, 40.0))
    assert _steps.lifts(one_frame, 0.02, step_mm=10.0,
                        min_airborne_s=0.030) == []
    # Two frames is 40 ms at 50 Hz and does clear a 30 ms threshold.
    two_frames = trace(("down", 3), ("up", 2), ("down", 3, 40.0))
    assert len(_steps.lifts(two_frames, 0.02, step_mm=10.0,
                            min_airborne_s=0.030)) == 1


def test_a_foot_still_airborne_at_the_end_never_landed():
    events = _steps.lifts(trace(("down", 3), ("up", 6, 90.0)), 0.02,
                          step_mm=10.0, min_airborne_s=0.030)
    assert len(events) == 1
    assert events[0]["landed"] is False
    # However far it travelled: it is a fall, not a step.
    assert events[0]["step"] is False


def test_no_frames_is_no_events_rather_than_an_exception():
    assert _steps.lifts([], 0.02, step_mm=10.0, min_airborne_s=0.030) == []
