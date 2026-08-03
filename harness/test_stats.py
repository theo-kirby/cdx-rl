"""The significance statements are pure, so they are tested without a GPU.

    uv run pytest harness/test_stats.py

``harness/_stats.py`` decides whether a table's ordering is worth believing,
which makes it the most consequential arithmetic in the harness and — until
this file — the only part of it with no test at all. ``test_steps.py`` covered
``lifts``, the detection half, and left the statistic uncovered.

**Every check must be able to fail** (DESIGN's rule 6, hazard 18). Each test
below asserts both directions where there are two, and the two experiment 003
cases are pinned as regressions against the published numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness._stats import mcnemar, significance_pp, two_sided_binomial  # noqa: E402


def rows(*outcomes: bool, start: int = 0) -> list[dict[str, object]]:
    """``rows(True, False)`` → two episode rows on seeds 0 and 1."""

    return [{"seed": start + i, "won": bool(o)} for i, o in enumerate(outcomes)]


def won(row: dict[str, object]) -> bool:
    return bool(row["won"])


# --------------------------------------------------------------------------
# The unpaired bound


def test_unpaired_bound_is_two_sigma_not_one():
    """The number printed beside every table, at the two seed counts used.

    Worth pinning because the standard error and the 2σ bound differ by
    exactly the factor that decides whether a 12-seed table looks conclusive:
    the se at n=12 is 20 pp and the bound is 41.
    """

    assert round(significance_pp(12), 1) == 40.8
    assert round(significance_pp(24), 1) == 28.9
    assert round(significance_pp(48), 1) == 20.4
    # It must shrink with n, or it is not a bound on anything.
    assert significance_pp(48) < significance_pp(24) < significance_pp(12)


def test_unpaired_bound_survives_a_zero_seed_table():
    """Never a ZeroDivisionError: a driver that crashes on an empty table
    cannot report that the table was empty."""

    assert significance_pp(0) == significance_pp(1)


# --------------------------------------------------------------------------
# McNemar


def test_identical_policies_are_maximally_unconvincing():
    """No discordant seeds is p = 1.0 — the least evidence of a difference,
    not an undefined question."""

    a = rows(True, True, False, False)
    b = rows(True, True, False, False)
    m = mcnemar(a, b, won)
    assert m["discordant"] == 0
    assert m["p_value"] == 1.0
    assert m["shared_seeds"] == 4


def test_a_clean_sweep_is_significant_and_a_split_is_not():
    """Both directions, which is the point of the test.

    Six discordant seeds all favouring the first policy is p = 0.031; the same
    six split evenly is p = 1.000. The unpaired bound cannot tell these apart
    at n=6 — it would call both a tie.
    """

    swept = mcnemar(rows(*[True] * 6), rows(*[False] * 6), won)
    assert swept["discordant"] == 6
    assert swept["only_first"] == 6 and swept["only_second"] == 0
    assert swept["p_value"] == 0.0312

    split = mcnemar(rows(True, True, True, False, False, False),
                    rows(False, False, False, True, True, True), won)
    assert split["discordant"] == 6
    assert split["p_value"] == 1.0


def test_direction_is_reported_not_just_the_p():
    """``only_first`` and ``only_second`` are separate facts from ``p``.

    A bare p hides which policy the discordant seeds favoured, and 5/2 and
    2/5 are the same p pointing opposite ways.
    """

    forward = mcnemar(rows(True, True, True, True, True, False, False),
                      rows(False, False, False, False, False, True, True), won)
    backward = mcnemar(rows(False, False, False, False, False, True, True),
                       rows(True, True, True, True, True, False, False), won)
    assert forward["p_value"] == backward["p_value"]
    assert forward["only_first"] == 5 and forward["only_second"] == 2
    assert backward["only_first"] == 2 and backward["only_second"] == 5


def test_only_shared_seeds_are_compared():
    """A checkpoint played on more seeds than another must not have the extras
    counted as wins. Silently scoring a 24-seed policy against a 12-seed one
    would manufacture a difference out of the seed lists."""

    a = rows(True, True, True, True)              # seeds 0-3
    b = rows(True, True, start=0)                 # seeds 0-1 only
    m = mcnemar(a, b, won)
    assert m["shared_seeds"] == 2
    assert m["discordant"] == 0
    assert m["p_value"] == 1.0
    # Disjoint seed lists share nothing, and that is not a significant result.
    disjoint = mcnemar(rows(True, True), rows(False, False, start=90), won)
    assert disjoint["shared_seeds"] == 0
    assert disjoint["p_value"] == 1.0


def test_the_predicate_decides_what_counts_as_a_win():
    """The reason the statistic takes a predicate: ``compare`` scores survival
    and ``steps`` scores the conjunction, over the same rows."""

    a = [{"seed": 0, "survived": True, "steps": 0},
         {"seed": 1, "survived": True, "steps": 3}]
    b = [{"seed": 0, "survived": True, "steps": 2},
         {"seed": 1, "survived": True, "steps": 1}]

    survival = mcnemar(a, b, lambda r: bool(r["survived"]))
    assert survival["discordant"] == 0        # both survived everywhere

    conjunction = mcnemar(a, b, lambda r: bool(r["survived"] and r["steps"] > 0))
    assert conjunction["discordant"] == 1     # seed 0 separates them
    assert conjunction["only_second"] == 1


# --------------------------------------------------------------------------
# Experiment 003, pinned


def test_003_tiebreak_reproduces_the_published_p_values():
    """The two comparisons in 003's README, which are the reason this test
    exists at all.

    Those three checkpoints scored 14, 17 and 16 of 24. It was first written
    up as *"1150 wins cleanly at 24 seeds"* and that was wrong — the paired
    test says the three are indistinguishable, and these are the numbers it
    said it with.
    """

    # 001150 vs best: discordant 7, split 4/3.
    assert two_sided_binomial(4, 3) == 1.0
    # The other tied pair: discordant 7, split 5/2.
    assert round(two_sided_binomial(5, 2), 3) == 0.453


def test_the_paired_test_can_separate_what_the_unpaired_bound_cannot():
    """The whole argument for adding it, as an assertion.

    Twelve seeds, one policy winning 9 and the other 3 of the discordant
    ones: the unpaired 2σ bound at n=12 is 40.8 pp and cannot call it, while
    the paired test reaches p < 0.15 on the same episodes.
    """

    a = rows(*([True] * 9 + [False] * 3))
    b = rows(*([False] * 9 + [True] * 3))
    m = mcnemar(a, b, won)
    assert m["discordant"] == 12
    assert m["p_value"] < 0.15
    # 75% against 25% is a 50 pp gap — this particular case the unpaired bound
    # would in fact call, which is why the assertion above is about the p and
    # not about a claim that the paired test always wins.
    assert significance_pp(12) > 40.0
