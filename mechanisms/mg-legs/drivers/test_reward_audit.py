"""The pure half of reward_audit.py — no mujoco, runnable under `uv run`.

    uv run pytest mechanisms/mg-legs/drivers/test_reward_audit.py -q

Every check asserts BOTH directions where there are two, following
`test_jitter.py`: a threshold that only ever fires one way is not tested.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reward_audit import (  # noqa: E402
    SD_LIVE, SHARE_ACHIEVED, SHARE_DEAD,
    classify, mean_sd, settle_index, shaping_shares, term_stats,
)


# --- mean_sd ---------------------------------------------------------------

def test_mean_sd_of_a_constant_is_zero_spread():
    mean, sd = mean_sd([0.4] * 10)
    assert mean == pytest.approx(0.4)
    assert sd == 0.0


def test_mean_sd_is_the_population_deviation_not_the_sample_one():
    # [0, 2]: population sd is 1.0; the sample (n-1) one would be 1.414.
    mean, sd = mean_sd([0.0, 2.0])
    assert mean == pytest.approx(1.0)
    assert sd == pytest.approx(1.0)


def test_mean_sd_of_empty_is_zero_and_does_not_raise():
    assert mean_sd([]) == (0.0, 0.0)


# --- term_stats ------------------------------------------------------------

def test_term_stats_reports_in_absolute_reward_units():
    # A kernel varying over its whole range, at weight 0.2.
    s = term_stats([0.0, 1.0], 0.2)
    assert s["mean_paid"] == pytest.approx(0.1)
    assert s["sd_paid"] == pytest.approx(0.1)
    assert s["min_paid"] == pytest.approx(0.0)
    assert s["max_paid"] == pytest.approx(0.2)


def test_term_stats_share_is_scale_free():
    # Same kernel readings, different weights -> same share, different paid.
    low = term_stats([0.5, 0.5], 0.2)
    high = term_stats([0.5, 0.5], 1.5)
    assert low["share"] == pytest.approx(high["share"]) == pytest.approx(0.5)
    assert low["mean_paid"] != pytest.approx(high["mean_paid"])


def test_term_stats_headroom_is_what_is_still_on_the_table():
    s = term_stats([0.25], 0.4)
    assert s["headroom"] == pytest.approx(0.3)
    # A term already at its maximum has no headroom.
    assert term_stats([1.0], 0.4)["headroom"] == pytest.approx(0.0)


def test_term_stats_counts_states():
    assert term_stats([0.1, 0.2, 0.3], 1.0)["states"] == 3


# --- classify: the 2x2 ------------------------------------------------------

def test_classify_calls_a_pinned_low_term_dead():
    # effort at experiment 006 seed 1: kernel ~0.037 at weight 0.2, flat.
    s = term_stats([0.037, 0.038, 0.036], 0.2)
    assert s["share"] < SHARE_DEAD
    assert s["sd_paid"] < SD_LIVE
    assert classify(s) == "dead"


def test_classify_calls_a_pinned_HIGH_term_achieved_not_dead():
    # `upright` on a policy that learned to stand: near 1.0 and flat. Same
    # spread as the dead case above and the opposite finding — this is the
    # distinction the share column exists to make.
    s = term_stats([0.98, 0.99, 0.985], 1.0)
    assert s["sd_paid"] < SD_LIVE
    assert s["share"] > SHARE_ACHIEVED
    assert classify(s) == "achieved"


def test_classify_calls_a_varying_term_live_regardless_of_share():
    low = term_stats([0.0, 0.4], 1.0)      # low share, big spread
    high = term_stats([0.6, 1.0], 1.0)     # high share, big spread
    assert classify(low) == "live"
    assert classify(high) == "live"


def test_classify_weights_the_spread_so_a_small_term_is_not_called_live():
    # The SAME kernel spread at two weights: 0.01 of the budget is the floor,
    # so a tiny-weight term swinging fully still cannot steer.
    tiny = term_stats([0.0, 0.08], 0.1)    # sd_paid = 0.004
    big = term_stats([0.0, 0.08], 1.5)     # sd_paid = 0.06
    assert tiny["sd_paid"] < SD_LIVE < big["sd_paid"]
    assert classify(tiny) != "live"
    assert classify(big) == "live"


def test_classify_middling_is_reachable():
    # Low spread, share between the two thresholds — deliberately not called
    # either way rather than being forced into one.
    s = term_stats([0.3, 0.3], 1.0)
    assert SHARE_DEAD < s["share"] < SHARE_ACHIEVED
    assert classify(s) == "middling"


# --- shaping_shares --------------------------------------------------------

def test_shaping_shares_normalise_to_one():
    stats = {"a": term_stats([0.0, 1.0], 1.0),
             "b": term_stats([0.4, 0.6], 1.0)}
    shares = shaping_shares(stats)
    assert math.fsum(shares.values()) == pytest.approx(1.0)
    assert shares["a"] > shares["b"]


def test_shaping_shares_rank_by_spread_not_by_weight():
    # `heavy` has 5x the weight and pays far more, but never moves; `light`
    # is what PPO can actually act on. This is the whole point of the column.
    stats = {"heavy": term_stats([1.0, 1.0], 1.5),
             "light": term_stats([0.0, 1.0], 0.3)}
    shares = shaping_shares(stats)
    assert stats["heavy"]["mean_paid"] > stats["light"]["mean_paid"]
    assert shares["heavy"] == pytest.approx(0.0)
    assert shares["light"] == pytest.approx(1.0)


def test_shaping_shares_of_an_all_constant_reward_do_not_divide_by_zero():
    stats = {"a": term_stats([0.5, 0.5], 1.0), "b": term_stats([1.0], 0.2)}
    assert shaping_shares(stats) == {"a": 0.0, "b": 0.0}


# --- settle_index ----------------------------------------------------------

def test_settle_index_rounds_up_to_the_first_frame_at_or_after():
    assert settle_index(0.02, 1.0) == 50      # 50 Hz control, 1.0 s
    assert settle_index(0.01, 1.0) == 100     # 100 Hz
    assert settle_index(0.03, 1.0) == 34      # 33.33 frames -> 34


def test_settle_index_refuses_a_non_positive_interval():
    with pytest.raises(ValueError):
        settle_index(0.0, 1.0)
    with pytest.raises(ValueError):
        settle_index(-0.02, 1.0)
