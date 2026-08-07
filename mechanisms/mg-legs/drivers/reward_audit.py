#!/usr/bin/env python
"""Which reward terms actually SHAPE behaviour at the states a policy reaches?

    /home/theo/cadex-train-venv/bin/python \\
        mechanisms/mg-legs/drivers/reward_audit.py \\
        --policy <p.cxpolicy> --task tasks/stand-b8-clamp25/stand-task.json \\
        [--seeds 12] [--json]

**Run under the trainer interpreter, never `uv run`** — same rule as
`hazard15.py` and `jitter.py`: this imports the engine, which imports mujoco,
which cdx-rl's own venv deliberately does not pin.

## Why this file exists

Experiment 006 shipped a `quiet` kernel whose σ was sized, by a pre-registered
rule, to read **0.368 at the control's resting state** — deliberately live in
the regime the machine occupies. It then split across two seeds: one bought
its quiet by standing still, the other by bracing against the servo, and the
term that was supposed to price bracing — `effort`, on real `actuator_force`
— was never checked the same way.

It reads **0.071** at the control and **0.037** where seed 1 rests. σ = 191.32
sits *below* every resting state the machine actually visits, so `effort` lives
on the far tail of its own Gaussian: cutting Σ|τ| by a third of the machine's
resting load buys 0.030 reward out of a 5.3 budget.

**Amended after the audit ran (2026-08-07): "far tail" overstates it, and this
file's own classifier says so.** Over the settled window `effort` returns
`sd_paid` of 0.020–0.032 against the 0.01 floor, so it classifies **`live`**,
not `dead` — it has gradient. What the audit actually found is that the term
is *priced trivially*: it collects 0.10–0.14 of its own weight where every
other shaping term collects 0.47–0.99, it steers 2.5–3.0 % of the total
spread against `capture`'s ~39 %, and it separates 006's two seeds — 0.98 %
and 45.19 % resting duty — by **0.0368 of a 5.30 budget**. The sign is right;
the magnitude is not. `experiments/007-price-the-bracing/README.md` §8.

**The discipline that caught this in 006 was applied only to the NEW term.**
`reward_standing.py` and `check_reward_pays.py` both check hazard 9 — that
every term pays its weight *at the nominal pose*. A kernel can pass that and
be flat where the trained machine lives; `effort` does exactly that. This
driver asks the other question, over the states a policy actually reaches.

## The statistic, and why it is a SPREAD and not a mean

`reward_decompose.py` (laptop-era, B7) prints per-term mean and worst. Those
answer "is the per-step reward negative", which since B8 has a structural
answer. They do not answer "can PPO exploit this term", because **a term that
is constant over the visited states has no gradient regardless of its mean**.
`alive` is literally the expression `1`: weight 0.2, mean 0.2, and it shapes
nothing. Its standard deviation is exactly zero, and this driver asserts that
— it is the built-in self-check that the statistic is measuring what it says.

So the reported quantity is the **standard deviation of w·kernel over visited
states**, in absolute reward units, beside the mean.

## The 2x2 this exists to make visible, and the trap inside it

A low spread has TWO causes and they are opposite:

|                | low spread                | high spread |
|----------------|---------------------------|-------------|
| **high share** | `achieved` — the policy won this term; little gradient left, and that is success | `live` |
| **low share**  | **`dead`** — pinned near zero, no gradient, unreachable from here | `live` |

**`achieved` and `dead` are the same standard deviation and opposite
findings.** Reading spread alone would call `upright` (share ~1.0 because the
machine learned to stand) the same as `effort` (share 0.04 because bracing is
unpriced). The share column is what separates them and neither may be read
without the other.

**This is correlational and the caveat is load-bearing.** A term flat at the
states *this* policy visits may be flat because the policy already solved it,
because it is unreachable, or because the policy never goes where it varies.
The driver reports; it does not conclude which.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MODULE_DIR = os.environ.get("CADEX_ENGINE_DEV_TREE", "/home/theo/cadex")

# Pre-registered before the audit was run, so the verdicts are not chosen by
# looking at the answer (ADR-097). A term collecting under 15 % of its own
# weight is pinned low; over 50 % is pinned high. `sd` is in absolute reward
# units and 0.01 of a 5.3 budget is the floor for "this can be exploited".
SHARE_DEAD = 0.15
SHARE_ACHIEVED = 0.50
SD_LIVE = 0.01

# ---------------------------------------------------------------------------
# The pure half — no mujoco, no engine. Tested by test_reward_audit.py under
# cdx-rl's own interpreter.
# ---------------------------------------------------------------------------


def mean_sd(values: list) -> tuple:
    """Mean and POPULATION standard deviation. Empty is (0.0, 0.0)."""
    if not values:
        return 0.0, 0.0
    n = len(values)
    mean = math.fsum(values) / n
    var = math.fsum((v - mean) ** 2 for v in values) / n
    return mean, math.sqrt(max(0.0, var))


def term_stats(kernel_values: list, weight: float) -> dict:
    """Per-term statistics in ABSOLUTE REWARD UNITS.

    ``kernel_values`` are the bare kernel readings in [0, 1]; every figure
    reported is multiplied by ``weight``, because a 0.3-weight term varying by
    its whole range moves the objective less than a 1.5-weight term varying by
    a fifth of its own.
    """
    mean, sd = mean_sd(kernel_values)
    lo = min(kernel_values) if kernel_values else 0.0
    hi = max(kernel_values) if kernel_values else 0.0
    return {
        "weight": weight,
        "mean_paid": weight * mean,
        "sd_paid": weight * sd,
        "min_paid": weight * lo,
        "max_paid": weight * hi,
        # The fraction of its own weight the term collects. Scale-free, so it
        # is comparable across terms of different weights.
        "share": mean,
        # What is still on the table: reward available if the term went to 1.
        "headroom": weight * (1.0 - mean),
        "states": len(kernel_values),
    }


def classify(stats: dict) -> str:
    """`dead` / `achieved` / `live` — see the 2x2 in the module docstring."""
    if stats["sd_paid"] >= SD_LIVE:
        return "live"
    if stats["share"] <= SHARE_DEAD:
        return "dead"
    if stats["share"] >= SHARE_ACHIEVED:
        return "achieved"
    return "middling"


ZERO_ACTION_LABEL = "zero-action servo"


def policy_label(policy_path) -> str:
    """`<run dir>/<file>` — because a BASENAME does not identify a seed.

    Two seeds of one label write the same filename into different run
    directories: `stand15-s2-.../stand15.001750.cxpolicy` and
    `stand15-s1-.../stand15.001750.cxpolicy`. Recording only `path.name`
    leaves the seed readable from **row order and nothing else**, which is the
    same defect `harness steps` has (it keys `results` by basename and
    silently sums two seeds into `survived 36/24`). Here the rows stay
    separate, so nothing is corrupted — but the record cannot be read without
    the command that produced it, which is worse than it looks a week later.

    ``None`` is the zero-action servo, which has no file.
    """
    if policy_path is None:
        return ZERO_ACTION_LABEL
    path = Path(policy_path)
    parent = path.parent.name
    return f"{parent}/{path.name}" if parent else path.name


def check_labels_unique(policy_paths: list) -> list:
    """Resolved labels, or `SystemExit` if two rows would carry the same one.

    A run-qualified label collides only when the *same file* is passed twice,
    which is always a mistake and never a comparison.
    """
    labels = [policy_label(p) for p in policy_paths]
    seen, duplicated = set(), []
    for label in labels:
        if label in seen and label not in duplicated:
            duplicated.append(label)
        seen.add(label)
    if duplicated:
        raise SystemExit(
            "two rows would carry the same label, so the table could not be "
            f"read: {', '.join(duplicated)}. Pass each policy once.")
    return labels


def shaping_shares(stats_by_label: dict) -> dict:
    """Each term's share of the TOTAL spread — "who is steering".

    Normalised over `sd_paid`, so it answers a different question from the
    declared weights: the weight is what the objective says a term is worth,
    this is what it is worth at the states the policy actually visits.
    """
    total = math.fsum(s["sd_paid"] for s in stats_by_label.values())
    if total <= 0.0:
        return {label: 0.0 for label in stats_by_label}
    return {label: s["sd_paid"] / total for label, s in stats_by_label.items()}


def settle_index(interval_s: float, settle_s: float) -> int:
    """First frame index at or after ``settle_s``. Shared shape with jitter.py."""
    if interval_s <= 0.0:
        raise ValueError(f"control_interval_s must be positive, got {interval_s}")
    return int(math.ceil(settle_s / interval_s))


# ---------------------------------------------------------------------------
# The engine half.
# ---------------------------------------------------------------------------


def _engine():
    """Import the engine lazily, so the pure half above stays importable."""
    sys.path.insert(0, str(Path(MODULE_DIR) / "src" / "Mod" / "cadex"))
    sys.path.insert(0, str(REPO / "harness"))
    sys.path.insert(0, str(REPO))
    import CadexDynamics as cd  # noqa: E402
    from _episodes import apply_variant  # noqa: E402
    return cd, apply_variant


def measure(policy_path, task: dict, model_xml: bytes, seeds: list,
            *, settle_s: float = 1.0, disturbance: str = "declared") -> dict:
    """Per-term statistics over the states ``policy_path`` reaches.

    ``policy_path`` of ``None`` audits the **zero-action servo** — the reward
    the machine collects with no network in it. That is the floor: any term
    already at its maximum there is not something training has to earn.
    """
    cd, apply_variant = _engine()

    if policy_path is None:
        header = weights = None
    else:
        container = cd.decode_policy(policy_path.read_bytes(),
                                     context=policy_path.name)
        header, weights = container["header"], container["weights"]
    row_label = policy_label(policy_path)

    played = (apply_variant(task, {"disturbance": False})
              if disturbance == "none" else task)

    interval_s = float(task["episode"]["control_interval_s"])
    settle = settle_index(interval_s, settle_s)

    rows = list(task["reward"])
    whole: dict = {str(r["label"]): [] for r in rows}
    settled: dict = {str(r["label"]): [] for r in rows}
    lengths, endings = [], {}
    compiled = None

    for seed in seeds:
        model = cd.load_model(model_xml)   # per episode (ADR-103 §9)
        observations: list = []

        def sample(step, data, _final, _action, _obs=observations):
            # `observation_values` is the ENGINE's own decoder of the sensor
            # vector into the task's declared channel names — the same route
            # the trainer's reward takes. Re-deriving it here would be a
            # second evaluator, which is the one thing this file must not be.
            _obs.append(cd.observation_values(played, data.sensordata))
            return None

        if header is None:
            zeros = [0.0] * len(task["actions"])

            def action_fn(_step, _o, _z=zeros):
                return _z
        else:
            def action_fn(_step, obs):
                return cd.policy_forward(header, weights, obs)

        episode = cd.evaluate_episode(model, played, actions=action_fn,
                                      sample=sample, seed=int(seed))
        lengths.append(len(observations))
        ending = ("survived" if episode.get("truncated")
                  else str(episode.get("termination")))
        endings[ending] = endings.get(ending, 0) + 1

        if compiled is None and observations:
            names = list(observations[0])
            compiled = {str(r["label"]):
                        cd.compile_reward(str(r["expression"]), names=names,
                                          context=str(r["label"]))
                        for r in rows}

        for index, observation in enumerate(observations):
            for r in rows:
                label = str(r["label"])
                value = float(cd.evaluate_reward(compiled[label], observation,
                                                 context=label))
                whole[label].append(value)
                if index >= settle:
                    settled[label].append(value)

    stats_whole, stats_settled = {}, {}
    for r in rows:
        label, weight = str(r["label"]), float(r["weight"])
        stats_whole[label] = term_stats(whole[label], weight)
        stats_settled[label] = term_stats(settled[label], weight)

    # **Self-check, not decoration.** `alive` is the expression `1`, so its
    # spread must be exactly zero. If this ever fires, the statistic is not
    # measuring what the table says it is and no other row can be believed.
    for label, s in stats_whole.items():
        expression = next(str(r["expression"]).strip() for r in rows
                          if str(r["label"]) == label)
        if expression == "1" and s["sd_paid"] != 0.0:
            raise SystemExit(
                f"self-check failed: constant term {label!r} has spread "
                f"{s['sd_paid']!r}, which is impossible. The audit is wrong.")

    return {
        "policy": row_label,
        # The whole path as given, so a row can be re-run from the record
        # alone. `policy` is what a table prints; this is what identifies it.
        "policy_path": (None if policy_path is None
                        else str(Path(policy_path).resolve())),
        "run": (None if policy_path is None else Path(policy_path).parent.name),
        "seeds": len(seeds),
        "settle_seconds": settle_s,
        "settle_frames": settle,
        "disturbance": disturbance,
        "budget": math.fsum(float(r["weight"]) for r in rows),
        "mean_episode_steps": (math.fsum(lengths) / len(lengths)
                               if lengths else 0.0),
        "endings": endings,
        "states_whole": sum(len(v) for v in whole.values()) // max(1, len(rows)),
        "states_settled": (sum(len(v) for v in settled.values())
                           // max(1, len(rows))),
        "whole": stats_whole,
        "settled": stats_settled,
        "shaping_share_settled": shaping_shares(stats_settled),
    }


def _print(row: dict) -> None:
    print(f"\n==> {row['policy']}  —  {row['seeds']} seeds, "
          f"{row['mean_episode_steps']:.1f} mean steps, {row['endings']}")
    print(f"    settled window from frame {row['settle_frames']} "
          f"({row['settle_seconds']} s); {row['states_settled']} settled states "
          f"of {row['states_whole']}")
    print(f"    budget {row['budget']:.2f}\n")
    print("    term           weight   paid     sd    share   steer   verdict")
    print("    " + "-" * 62)
    share = row["shaping_share_settled"]
    for label, s in row["settled"].items():
        print(f"    {label:<14} {s['weight']:5.2f}  {s['mean_paid']:6.4f} "
              f"{s['sd_paid']:6.4f}  {s['share']*100:5.1f}% "
              f"{share.get(label, 0.0)*100:5.1f}%   {classify(s)}")
    paid = math.fsum(s["mean_paid"] for s in row["settled"].values())
    print("    " + "-" * 62)
    print(f"    {'TOTAL':<14} {row['budget']:5.2f}  {paid:6.4f}"
          f"          {paid/row['budget']*100:5.1f}%")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="which reward terms shape behaviour at the visited states")
    ap.add_argument("--policy", action="append", type=Path, default=[])
    ap.add_argument("--zero-action", action="store_true",
                    help="audit the zero-action servo as the floor")
    ap.add_argument("--task", required=True, type=Path)
    ap.add_argument("--model", type=Path)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--settle-seconds", type=float, default=1.0)
    ap.add_argument("--disturbance", choices=("declared", "none"),
                    default="declared")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if not args.policy and not args.zero_action:
        ap.error("one of --policy or --zero-action is required")

    task = json.loads(args.task.read_text())
    model_path = args.model or (args.task.parent / "model-model.xml")
    model_xml = model_path.read_bytes()
    seeds = list(range(args.seeds))

    selected = ([None] if args.zero_action else []) + list(args.policy)
    check_labels_unique(selected)
    rows = [measure(p, task, model_xml, seeds,
                    settle_s=args.settle_seconds,
                    disturbance=args.disturbance) for p in selected]

    if args.json:
        # The task is recorded because a policy may legitimately be scored
        # under a bundle it was not trained on — 006 §4c — and the term list
        # in this table is the SCORING bundle's, not the training one's.
        print(json.dumps({"task": str(args.task.resolve()),
                          "model": str(model_path.resolve()),
                          "rows": rows}, indent=2))
    else:
        for row in rows:
            _print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
