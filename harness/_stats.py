"""The two significance statements the drivers make, in one place.

``compare`` and ``steps`` both rank policies by playing them against a fixed
list of evaluation seeds, and both have to say how much of the resulting
ordering the sample size will support. They were saying it differently: for a
while ``steps`` printed the paired test and ``compare`` printed only the
unpaired bound, which is the weaker of the two on exactly the comparison both
drivers exist to make.

Kept free of mujoco, of the trainer venv and of every other harness module, so
that it is testable under cdx-rl's own interpreter — ``harness/test_stats.py``
runs with no GPU, no model and no bundle.

## The two tests, and when each is right

**The unpaired 2σ bound** (:func:`significance_pp`) asks whether two
*independent* samples differ. It is the right test for "is this run's 17/24
better than a *different* run's 14/24", and it is deliberately worst-case:
it assumes p = 0.5 in both, which maximises the variance.

**McNemar** (:func:`mcnemar`) asks whether two policies played against *the
same* seeds differ. That is what both drivers actually do, and the pairing is
strong — a seed fixes the reset draw and the entire disturbance schedule, so
two checkpoints will agree on most episodes for reasons that have nothing to
do with either policy. The unpaired bound throws that structure away, and it
throws it away *conservatively*: experiment 003's three tied checkpoints
scored 14, 17 and 16 of 24, a spread the 29 pp unpaired bound cannot separate
at any sample size the budget allows.

Neither is used to auto-select anything. They say how much evidence there is;
they do not say which checkpoint to install (ADR-099).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable


def significance_pp(n: int) -> float:
    """2σ on a *difference* of two binomial rates, in percentage points.

    The worst case is p = 0.5 in both, so ``se = sqrt(2 * 0.25 / n)``. Printed
    beside every table, because a driver that ranks without it invites the
    reader to believe an ordering the sample size cannot support.
    """

    return 2.0 * math.sqrt(2.0 * 0.25 / max(1, n)) * 100.0


def two_sided_binomial(only_a: int, only_b: int) -> float:
    """Exact two-sided binomial p for a split of discordant pairs.

    Under the null the discordant seeds are a fair coin, so this is exact —
    no normal approximation and no minimum sample size, which matters because
    a 12-seed table can easily produce three or four discordant pairs and the
    χ² form of McNemar is not trustworthy there.

    No discordant pairs at all is ``p = 1.0``: the two policies did exactly
    the same thing on every shared seed, which is the least possible evidence
    of a difference rather than an undefined question.
    """

    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def mcnemar(
    a: Iterable[dict[str, Any]],
    b: Iterable[dict[str, Any]],
    outcome: Callable[[dict[str, Any]], bool],
    seed_key: str = "seed",
) -> dict[str, Any]:
    """McNemar over the evaluation seeds two policies share.

    ``outcome`` is what counts as a success for the metric being compared, and
    it is a parameter because the two drivers score different things:
    ``compare`` ranks on survival alone, ``steps`` on the conjunction
    *stepped ≥ 10 mm AND survived*. Passing the predicate in keeps one
    implementation of the statistic rather than two that can drift.

    Only the **discordant** seeds — the ones where exactly one of the two
    succeeded — carry information. Count them, then ask whether the split is
    worth believing.

    ``only_first`` and ``only_second`` are reported alongside the p, because
    the direction and the weight of the evidence are different facts and a
    bare p hides the first. Seven discordant seeds split 4/3 and seventy split
    40/30 both fail to reach significance, but they are not the same finding.
    """

    by_seed_a = {row[seed_key]: bool(outcome(row)) for row in a}
    by_seed_b = {row[seed_key]: bool(outcome(row)) for row in b}
    shared = sorted(set(by_seed_a) & set(by_seed_b))
    only_a = sum(1 for s in shared if by_seed_a[s] and not by_seed_b[s])
    only_b = sum(1 for s in shared if by_seed_b[s] and not by_seed_a[s])
    return {
        "shared_seeds": len(shared),
        "only_first": only_a,
        "only_second": only_b,
        "discordant": only_a + only_b,
        "p_value": round(two_sided_binomial(only_a, only_b), 4),
    }
