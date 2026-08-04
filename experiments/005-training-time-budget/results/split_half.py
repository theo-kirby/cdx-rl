#!/usr/bin/env python
"""§8.2's table: where each arm's bracing saturates.

    /home/theo/cadex-train-venv/bin/python \\
        experiments/005-training-time-budget/results/split_half.py

Reads the three `series-*.json` beside it, produced by

    hazard15.py --series <run dir> --stride 50 --task <bundle> --seeds 6 --json

**This statistic is post-hoc and the README says so.** The gate's decision
rule was a whole-series linear fit, written down before anything was run
(§6b). That fit predicts a duty cycle of 140.6 % for the unclamped control,
which is impossible, so the veto it fired cannot rest on its magnitude — only
on its direction. The split below is how far the direction can honestly be
pushed, and it is reported as a diagnostic rather than as the rule.

Split at iteration 1000 because that is where the *control* transitions, not
because it flatters any arm: below it the unclamped arm sits at mean 53.9 %
of rating with 20.4 % duty, above it at 78.0 % with 54.9 %.

Needs numpy only. `/home/theo/cadex-train-venv` has it; cdx-rl's own venv
does not, by design.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPLIT = 1000
ARMS = [("unclamped (b8)", "unclamped"),
        ("clamp25", "clamp25"),
        ("clamp15", "clamp15")]


def main() -> int:
    print("Duty (>90 % of rating, settled) and mean % of rating, "
          f"split at iteration {SPLIT}")
    print(f"{'arm':<16}{'half':<8}{'n':>3}{'duty mean':>11}{'duty slope':>12}"
          f"{'mean% mean':>12}{'mean% slope':>13}")
    print("-" * 76)
    for label, stem in ARMS:
        path = HERE / f"series-{stem}-s2.json"
        rows = json.loads(path.read_text())["rows"]
        it = np.array([r["iteration"] for r in rows], dtype=float)
        duty = np.array([r["settled_duty_worst"] for r in rows],
                        dtype=float) * 100
        mean = np.array([r["settled_mean_frac_of_rating"] for r in rows],
                        dtype=float) * 100
        for name, mask in [(f"<{SPLIT}", it < SPLIT), (f">={SPLIT}", it >= SPLIT)]:
            # Slopes are per 1000 iterations, to match the driver's own
            # trend line and the README's tables.
            ds = np.polyfit(it[mask], duty[mask], 1)[0] * 1000
            ms = np.polyfit(it[mask], mean[mask], 1)[0] * 1000
            print(f"{label:<16}{name:<8}{int(mask.sum()):>3}"
                  f"{duty[mask].mean():>10.1f}%{ds:>+11.2f}"
                  f"{mean[mask].mean():>11.1f}%{ms:>+12.2f}")
        print()
    print("Duty is a THRESHOLD statistic on a rising distribution: flat while "
          "the mean is\nlow, then sharp once the mean nears the threshold. "
          "clamp25's duty is flat over\n1000-1750 at 15.0 %, but its mean is "
          "still climbing at +10 pp/1000 and sits at\n54.6 % — the unclamped "
          "arm's PRE-transition operating point. That is why the\nflat duty "
          "is not evidence of a buildable equilibrium.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
