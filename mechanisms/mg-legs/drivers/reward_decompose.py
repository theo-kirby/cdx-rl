#!/usr/bin/env python
"""Which term is the reward, at the states a policy actually reaches?

    CADEX_REPO=~/cadex pixi run python ~/cdx-mjc/reward_decompose.py \\
        runs/b7/stand9-probe.000100.cxpolicy

A training curve reports one number per iteration and the number is a sum.
When it goes the wrong way the sum says nothing about WHICH of fourteen
terms did it, and this project has now twice spent a run finding out the
slow way. This prints the per-term mean over the states the policy visits,
which is the decomposition the curve is hiding.

**The question it exists to answer is whether the per-step reward is
negative.** Through B7 `alive` paid +1.0 and every other term was a cost, so
if the costs outweighed it in the states the reset produces, then ending the
episode was worth more than continuing it, the optimal policy was to fall
over immediately, and PPO would find that -- which reads on the curve as "it
failed to learn" and is really "it learned exactly what was asked". It
happened twice: B7 as first written paid -0.2060 per step on the states B6's
own policy visits, where B6's own objective paid +0.0103.

**Since B8 that question has a structural answer and this prints the
evidence for it.** Every term is `w * exp(-(e/s)^2)` with `w > 0`, bounded in
[0, w], so the per-step total is bounded below by the constant term's weight
and cannot be negative in ANY state. So the columns that matter here are no
longer the totals -- they are the MINIMUM over the visited states, and the
per-term means, which say which terms are actually doing work rather than
sitting at their maximum.

**IT NO LONGER RECONSTRUCTS AN EARLIER OBJECTIVE.** Through B7 that was
exact and worth doing: B7 changed four of B6's eleven terms and left the
other ten byte-identical, so B6's reward could be rebuilt out of B7's and
scored on the same trajectories, making the comparison about the objective
rather than about two runs that visited different states. B8 shares no term
with B6 -- different shape, different sign, different scales -- so a
reconstruction would be a second reward invented here, which is the one
thing this file must not do. What replaces it is the per-state minimum.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("CADEX_REPO", Path.home() / "cadex"))
sys.path.insert(0, str(REPO / "src" / "Mod" / "cadex"))
sys.path.insert(0, str(HERE))

import CadexDynamics as dyn  # noqa: E402
import compare  # noqa: E402

PROJECT = HERE / "mg-legs.cadex"
SEEDS = tuple(range(12))


def totals(rows, observations):
    """Per-term mean and worst, and the per-STATE total's mean and worst.

    The per-state total is accumulated rather than derived from the means,
    because the number B8 turns on is the WORST single state -- and a mean
    of means cannot see it.
    """

    sums = {str(row["label"]): 0.0 for row in rows}
    worst = {str(row["label"]): float("inf") for row in rows}
    state_totals = []
    for observation in observations:
        state = 0.0
        for row in rows:
            label = str(row["label"])
            value = dyn.evaluate_reward(row["expression"], observation,
                                        context="decompose")
            paid = value * float(row["weight"])
            sums[label] += paid
            worst[label] = min(worst[label], paid)
            state += paid
        state_totals.append(state)
    count = max(1, len(observations))
    per_term = {label: total / count for label, total in sums.items()}
    return per_term, worst, state_totals


def main(argv: list[str]) -> int:
    if not argv:
        raise SystemExit("usage: reward_decompose.py <policy.cxpolicy>")
    policy_path = Path(argv[0])
    if not policy_path.is_absolute():
        policy_path = HERE / policy_path

    task, model_factory = compare.load_task(compare.newest_task_bundle(PROJECT))
    container = dyn.decode_policy(policy_path.read_bytes())
    header = container["header"]
    weights = container["weights"]
    names = [str(name) for name in header["observations"]]
    print(f"==> policy {policy_path.name}: iteration "
          f"{compare.iteration_of(container)}, {len(names)} channels")

    def actions(_step, observation):
        return dyn.policy_forward(header, weights,
                                  [observation[name] for name in names])

    visited, lengths, endings = [], [], {}
    for seed in SEEDS:
        episode = dyn.evaluate_episode(model_factory(), task,
                                       actions=actions, seed=seed)
        steps = episode["steps"]
        lengths.append(len(steps))
        ending = "survived" if episode["truncated"] else str(episode["termination"])
        endings[ending] = endings.get(ending, 0) + 1
        visited.extend(step["observation"] for step in steps)

    print(f"==> {len(SEEDS)} seeds: mean episode {sum(lengths) / len(lengths):.1f} "
          f"steps of {task['episode']['max_steps']}, {dict(endings)}")
    print(f"    {len(visited)} states visited")

    rows = list(task["reward"])
    per_term, worst_term, state_totals = totals(rows, visited)
    budget = sum(float(row["weight"]) for row in rows)

    print("==> per-step reward at those states")
    print("    term             weight      mean     worst    of weight")
    for row in rows:
        label = str(row["label"])
        weight = float(row["weight"])
        mean = per_term.get(label, 0.0)
        share = mean / weight if weight else float("nan")
        print(f"    {label:14s} {weight:+8.3f}  {mean:+8.4f}  "
              f"{worst_term.get(label, 0.0):+8.4f}  {share * 100:7.1f}%")
    total = sum(per_term.values())
    print(f"    {'TOTAL':14s} {budget:+8.3f}  {total:+8.4f}  "
          f"{min(state_totals):+8.4f}  {total / budget * 100:7.1f}%")

    print("==> the question")
    verdict = ("NEGATIVE -- ending the episode beats continuing it"
               if total < 0 else "positive -- surviving pays")
    print(f"    mean per-step reward {total:+.4f}: {verdict}")
    floor = min(state_totals)
    print(f"    WORST SINGLE STATE of {len(visited)}: {floor:+.4f}")
    if floor < 0:
        print("    FAIL: some state pays a negative per-step reward, so "
              "terminating there beats\n    continuing. Under an "
              "all-positive reward this is impossible, so either a\n"
              "    weight is negative or a term is not the kernel it "
              "looks like.")
        return 1
    constant = sum(float(row["weight"]) for row in rows
                   if str(row["expression"]).strip() == "1")
    print(f"    ...and it cannot go below {constant:+.4f}, which is the "
          f"constant term's weight.")
    print("    Every shaping term is a positive bounded kernel, so no state "
          "the machine can\n    reach -- however badly it is doing -- makes "
          "ending the episode worth more than\n    continuing it. That is "
          "the failure B7 measured, removed by construction rather\n    "
          "than avoided by arithmetic.")
    print("==> what to read in the table above")
    print("    A term sitting near 100% of its weight is one the machine is "
          "already\n    satisfying and which is therefore paying no "
          "gradient; a term far below it is\n    where the work is. "
          "`capture` is the term this run is about, so it is the one\n"
          "    that should have room to grow.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
