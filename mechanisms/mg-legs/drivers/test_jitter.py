"""``jitter``'s statistics are pure, so they are tested without mujoco or a GPU.

    uv run pytest mechanisms/mg-legs/drivers/test_jitter.py

The driver keeps its arithmetic above a divider and defers ``import
CadexDynamics`` into a function, which is what makes this file possible under
cdx-rl's own interpreter — `harness/_steps.py` is laid out the same way and
`hazard15.py` is not, which is why `hazard15.py` has no test.

**Every check must be able to fail** (DESIGN's rule 6, hazard 18). Each test
below asserts both directions where there are two, and most of them encode a
mistake this repository has actually made.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jitter  # noqa: E402


# ---------------------------------------------------------------------------
# The settle window
# ---------------------------------------------------------------------------


def test_settle_window_is_a_duration_not_a_step_count():
    """003 halved the control rate, and a step count would have changed meaning.

    The same failure `harness/_steps.min_airborne_steps` was rewritten to
    prevent: 1.0 s is 100 control steps at 100 Hz and 50 at 50 Hz, and a
    literal ``50`` would silently have become a 0.5 s window on the older
    bundles and made every settled figure incomparable.
    """

    assert jitter.settle_frames(0.010, 1.0) == 100   # 100 Hz
    assert jitter.settle_frames(0.020, 1.0) == 50    # 50 Hz — this bundle
    assert jitter.settle_frames(0.002, 1.0) == 500   # solver rate
    # And it is not a constant: a different duration gives a different window.
    assert jitter.settle_frames(0.020, 2.0) == 100


# ---------------------------------------------------------------------------
# Command jitter
# ---------------------------------------------------------------------------


def test_reset_frame_is_dropped_not_zero_filled():
    """``action`` is ``None`` at the reset pose, and zero-filling it lies.

    ``evaluate_episode`` passes ``None`` before any action has been taken. A
    driver that turned that into a row of zeros would manufacture one enormous
    Δ at step 0 — the entire nominal pose appearing as a single-step command
    jump — which is the reset-drop instrument error in a new place.
    """

    actions = [None, [10.0, -5.0], [10.5, -5.0], [11.0, -5.0]]
    deltas = jitter.command_deltas(actions, 0.02)
    # Three commands issued -> two deltas. Four frames would mean the None
    # became a row.
    assert deltas.shape == (2, 2)
    assert np.allclose(deltas, [[0.5, 0.0], [0.5, 0.0]])
    # The direction that can fail: had the None been zero-filled, the first
    # delta would be the whole pose.
    assert deltas.max() < 1.0


def test_command_deltas_are_zero_for_a_constant_command():
    actions = [None] + [[3.0, 4.0]] * 10
    deltas = jitter.command_deltas(actions, 0.02)
    assert deltas.shape == (9, 2)
    assert np.allclose(deltas, 0.0)
    # …and non-zero for one that moves, or the test above proves nothing.
    moving = [None] + [[float(i), 4.0] for i in range(10)]
    assert jitter.command_deltas(moving, 0.02).max() == pytest.approx(1.0)


def test_command_deltas_of_a_ramp_are_its_slope():
    actions = [None] + [[2.0 * i] for i in range(6)]
    deltas = jitter.command_deltas(actions, 0.02)
    assert np.allclose(deltas, 2.0)


def test_too_few_commands_is_an_empty_array_not_a_crash():
    assert jitter.command_deltas([None], 0.02).shape[0] == 0
    assert jitter.command_deltas([None, [1.0]], 0.02).shape[0] == 0


def test_reversal_rate_separates_chatter_from_tracking():
    """The whole reason the reversal rate is reported beside the magnitude.

    A monotone sweep and a strict alternation can have the *same* mean |Δ|.
    Only the reversal rate tells them apart, and reporting magnitude alone is
    how "the legs jitter" would be confused with "the legs move".
    """

    ramp = np.asarray([[1.0]] * 10)                    # always +1
    alternating = np.asarray([[1.0], [-1.0]] * 5)      # +1, -1, +1, …

    assert jitter.sign_reversals_per_s(ramp, 0.02)[0] == 0.0
    # 10 samples at 50 Hz is 0.2 s; 9 reversals -> 45 /s, approaching the true
    # ceiling of 1/dt = 50 /s. **Not 25** — a full oscillation carries two
    # direction changes, and printing 25 as the ceiling is what made a
    # measured 38.6 look impossible when it was merely large.
    assert jitter.sign_reversals_per_s(alternating, 0.02)[0] == pytest.approx(45.0)
    assert jitter.sign_reversals_per_s(alternating, 0.02)[0] < 1.0 / 0.02
    # Both directions asserted: the ramp is not merely small, it is zero.
    assert jitter.sign_reversals_per_s(ramp, 0.02)[0] != \
        jitter.sign_reversals_per_s(alternating, 0.02)[0]


def test_a_zero_delta_is_not_a_direction():
    """A joint that pauses has not reversed, and the pause must not break the run."""

    # +1, 0, +1 is one continuous direction with a pause in it.
    paused = np.asarray([[1.0], [0.0], [1.0]])
    assert jitter.sign_reversals_per_s(paused, 0.02)[0] == 0.0
    # …but +1, 0, -1 IS a reversal, pause or no pause.
    turned = np.asarray([[1.0], [0.0], [-1.0]])
    assert jitter.sign_reversals_per_s(turned, 0.02)[0] > 0.0


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def test_rms_and_mean_differ_on_a_spiky_sample():
    """Both are reported because they answer different questions.

    A joint still most of the time with occasional spikes has a low mean and a
    high RMS. Calling either one "the jitter" alone hides the other.
    """

    still_with_spike = np.asarray([0.0] * 99 + [100.0])
    s = jitter.summarise(still_with_spike)
    assert s["mean"] == pytest.approx(1.0)
    assert s["rms"] == pytest.approx(10.0)
    assert s["rms"] > 5.0 * s["mean"]
    # A flat sample has them equal — the direction that can fail.
    flat = jitter.summarise(np.asarray([3.0] * 50))
    assert flat["mean"] == pytest.approx(flat["rms"])


def test_summarise_agrees_with_numpy_and_handles_empty():
    v = np.asarray([1.0, -2.0, 3.0, -4.0, 5.0])
    s = jitter.summarise(v)
    assert s["mean"] == pytest.approx(float(np.mean(np.abs(v))))
    assert s["p95"] == pytest.approx(float(np.percentile(np.abs(v), 95)))
    assert s["max"] == pytest.approx(5.0)
    assert s["n"] == 5
    empty = jitter.summarise(np.zeros(0))
    assert empty["n"] == 0 and empty["rms"] == 0.0


# ---------------------------------------------------------------------------
# Slip
# ---------------------------------------------------------------------------


def test_motion_along_the_normal_is_not_slip():
    """A foot pressing straight down is not sliding, however fast it presses."""

    n = np.asarray([0.0, 0.0, 1.0])
    assert jitter.tangential_speed(np.asarray([0.0, 0.0, -500.0]), n) == \
        pytest.approx(0.0)
    # In-plane motion is slip in full.
    assert jitter.tangential_speed(np.asarray([3.0, 4.0, 0.0]), n) == \
        pytest.approx(5.0)
    # And a mixture keeps only the in-plane part.
    assert jitter.tangential_speed(np.asarray([3.0, 4.0, -99.0]), n) == \
        pytest.approx(5.0)


def test_slip_is_measured_in_the_contact_frame_not_the_world():
    """A tilted contact — a foot on its edge — has a tilted normal.

    Projecting against world +Z instead of the contact normal would report
    sliding where there is none, and this is the case that catches it.
    """

    n = np.asarray([1.0, 0.0, 1.0]) / np.sqrt(2.0)   # 45° normal
    along = np.asarray([1.0, 0.0, 1.0])              # straight along it
    assert jitter.tangential_speed(along, n) == pytest.approx(0.0)
    # Perpendicular to it, in the same plane, is full slip.
    across = np.asarray([1.0, 0.0, -1.0])
    assert jitter.tangential_speed(across, n) == pytest.approx(np.sqrt(2.0))


def test_a_degenerate_normal_falls_back_to_the_full_speed():
    """Rather than dividing by zero and reporting NaN slip."""

    v = np.asarray([3.0, 4.0, 0.0])
    assert jitter.tangential_speed(v, np.zeros(3)) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Centre of pressure
# ---------------------------------------------------------------------------


def test_cop_of_one_contact_is_that_contact():
    p = [np.asarray([1.0, 2.0, 0.0])]
    assert np.allclose(jitter.centre_of_pressure(p, [7.0]), [1.0, 2.0, 0.0])


def test_cop_is_force_weighted_not_a_centroid():
    """The bug this prevents: an unweighted centroid puts the CoP between a
    fully-loaded heel and a barely-touching toe, which is nowhere."""

    p = [np.asarray([0.0, 0.0, 0.0]), np.asarray([0.0, 40.0, 0.0])]
    assert np.allclose(jitter.centre_of_pressure(p, [1.0, 1.0]),
                       [0.0, 20.0, 0.0])
    # 3:1 -> a quarter of the way, not halfway.
    assert np.allclose(jitter.centre_of_pressure(p, [3.0, 1.0]),
                       [0.0, 10.0, 0.0])


def test_an_unloaded_foot_has_no_centre_of_pressure():
    """``None``, not a divide — and not the centroid, which would draw a
    confident point through an airborne foot."""

    p = [np.asarray([0.0, 0.0, 0.0]), np.asarray([0.0, 40.0, 0.0])]
    assert jitter.centre_of_pressure(p, [0.0, 0.0]) is None
    assert jitter.centre_of_pressure([], []) is None
    # The other branch: a real load returns a point.
    assert jitter.centre_of_pressure(p, [0.0, 1.0]) is not None


def test_cop_margins_go_negative_outside_the_box():
    """Clamping to zero would make 'on the edge' and 'off the edge' read the
    same, and the second is a machine that is rolling over."""

    box = {"x": [-20.0, 20.0], "y": [-36.75, 33.25]}
    middle = jitter.cop_margins(np.asarray([0.0, 0.0]), box)
    assert middle["forward_mm"] == pytest.approx(33.25)
    assert middle["backward_mm"] == pytest.approx(36.75)
    assert middle["lateral_mm"] == pytest.approx(20.0)

    over_the_toe = jitter.cop_margins(np.asarray([0.0, 40.0]), box)
    assert over_the_toe["forward_mm"] < 0.0
    assert over_the_toe["backward_mm"] > 0.0


def test_cop_lateral_margin_is_the_nearer_of_the_two_edges():
    box = {"x": [-20.0, 20.0], "y": [-36.75, 33.25]}
    assert jitter.cop_margins(np.asarray([15.0, 0.0]), box)["lateral_mm"] == \
        pytest.approx(5.0)
    assert jitter.cop_margins(np.asarray([-15.0, 0.0]), box)["lateral_mm"] == \
        pytest.approx(5.0)


# ---------------------------------------------------------------------------
# The σ rule
# ---------------------------------------------------------------------------


def test_sigma_is_the_median_of_exactly_what_it_is_given():
    """`swirl_scale.py`'s rule: half above, half below, so the term is a
    gradient rather than a constant.

    **The list matters more than the statistic**, and this pins that the
    function does not silently widen itself. σ for `quiet` must come from the
    settled frames alone; concatenating the reset drop — whose Σ|q̇| is an
    order of magnitude larger — would produce a σ so wide the kernel could not
    tell jitter from stillness at all.
    """

    settled = [10.0, 12.0, 14.0, 16.0]
    drop = [400.0, 500.0, 600.0, 700.0]
    assert jitter.sigma_from(settled) == pytest.approx(13.0)
    # The direction that can fail: had the driver concatenated, σ would be
    # (16 + 400) / 2 = 208 — sixteen times wider, and useless as a gradient
    # over the regime the kernel actually operates in.
    assert jitter.sigma_from(settled + drop) == pytest.approx(208.0)


def test_sigma_of_nothing_is_zero_not_a_crash():
    assert jitter.sigma_from([]) == 0.0


# ---------------------------------------------------------------------------
# The series trend
# ---------------------------------------------------------------------------


def test_trend_keeps_its_key_set_on_a_one_checkpoint_run():
    """`hazard15.trend` printed a KeyError because its early return had a
    different shape. Same keys either way, so no caller has to check."""

    one = jitter.trend([100], [5.0])
    many = jitter.trend([100, 200, 300], [5.0, 6.0, 7.0])
    assert set(one) == set(many)
    assert one["degenerate"] is True and many["degenerate"] is False


def test_trend_reports_the_sign_of_a_real_slope():
    rising = jitter.trend([0, 1000, 2000], [10.0, 20.0, 30.0])
    falling = jitter.trend([0, 1000, 2000], [30.0, 20.0, 10.0])
    assert rising["slope_per_1000"] == pytest.approx(10.0)
    assert falling["slope_per_1000"] == pytest.approx(-10.0)
