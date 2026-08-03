"""``steps`` — did it actually STEP, and did it survive doing so?

The question ``compare`` and ``capability`` cannot ask. Both reduce an episode
to *did it survive*, and survival is not what a push-recovery task is for: a
machine that rejects every shove on its ankles and a machine that catches
itself with a step both read as ``12/12``, and only the second is the
behaviour the reward was written to buy.

**The metric is the conjunction: stepped >10 mm AND survived the full
episode.** Not either alone, and the reason is measured rather than argued.
Across experiment 003's twenty-eight checkpoints the two marginals move
*independently* — checkpoint 500 stepped in 7 of 12 episodes and survived 3,
while the probe stepped in 4 and survived 4 — so a run scored on stepping
alone rewards a machine that throws a foot out and falls over, and a run
scored on survival alone is the number six runs of this project already
maximised without ever landing a step.

    uv run python -m harness steps --policy FILE… --task BUNDLE
                                   [--dir RUN] [--profile NAME|PATH]
                                   [--seeds N | --seed-list 0,1,2]
                                   [--json] [--jobs N]

Exit codes are the package's: ``0`` fine, ``1`` infrastructure, ``2`` usage.

## Reading the table

``both`` is the metric. ``surv`` and ``step`` are printed beside it because
the *relationship* between them is the diagnosis: when they are equal, every
episode the policy survives it survives by stepping, which is the shape a
recovery policy is supposed to have and which experiment 003 is the first run
of this project to reach.

## Two bounds, and the unpaired one is the wrong test here

``compare``'s bound is printed first because it is the one this repository
already quotes: survival is binomial, so the worst-case standard error of a
difference of two *independent* samples is ``sqrt(2 * 0.25 / n)`` — 41 pp at
n=12, 29 pp at n=24.

**But checkpoints are not independent samples of each other.** Every one is
played against the *same seeds*, and a seed fixes the reset draw and the
whole disturbance schedule, so most episodes agree for reasons that have
nothing to do with the policy. The unpaired bound throws that away and is
conservative by a wide margin: experiment 003's three tied checkpoints score
14, 17 and 16 of 24, a 12.5 pp spread that the 29 pp unpaired bound cannot
separate — **and neither can 24 seeds, on that test.** So the driver also
prints McNemar over the discordant seeds, which is the test the design
actually supports.

Both are reported and neither auto-selects. Install by the conjunction and
say what the evidence was.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from harness import EXIT_INFRASTRUCTURE, EXIT_OK, EXIT_USAGE
from harness._stats import mcnemar, significance_pp
from harness.episodes import BundleError, load_bundle, resolve_model
from harness.provenance import envelope, sha256_file
from harness.trainer_venv import (
    TrainerVenvError,
    EpisodeJobFailed,
    run_bridge_job,
    unpinned_drift,
    STEPS_SCRIPT,
)

PROFILE_DIR = Path(__file__).resolve().parent / "profiles"


def load_profile(name: str) -> dict[str, Any]:
    """A mechanism profile, by name from ``harness/profiles`` or by path."""

    candidate = Path(name)
    if not candidate.is_file():
        candidate = PROFILE_DIR / f"{name}.json"
    if not candidate.is_file():
        available = sorted(p.stem for p in PROFILE_DIR.glob("*.json"))
        raise BundleError(
            f"No profile {name!r}. Available: {', '.join(available) or 'none'}. "
            "A profile names the mechanism's feet and ground bodies; there is "
            "no default, because a wrong one reports a permanent lift rather "
            "than an error."
        )
    profile = json.loads(candidate.read_text(encoding="utf-8"))
    for key in ("feet", "ground", "step_mm", "min_airborne_s"):
        if key not in profile:
            raise BundleError(f"{candidate} has no {key!r}")
    profile["feet"] = {k: v for k, v in profile["feet"].items()
                       if not k.startswith("$")}
    return profile


#: The conjunction, as a predicate over one episode row. This is what makes
#: ``steps``' paired test different from ``compare``'s: both play the same
#: seeds and both use :func:`harness._stats.mcnemar`, but this driver scores a
#: policy on *stepped AND survived* where ``compare`` scores it on survival
#: alone.
def stepped_and_survived(row: dict[str, Any]) -> bool:
    return bool(row["survived"] and row["steps"] > 0)


def paired(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, Any]:
    """McNemar on the conjunction, over the seeds the two policies share.

    The statistic and the argument for it live in :mod:`harness._stats`, which
    ``compare`` now shares. This is the conjunction-scored spelling of it.

    Reported, never used to auto-select. It says how much evidence there is;
    it does not say which checkpoint to install.
    """

    return mcnemar(a, b, stepped_and_survived)


def score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fold one policy's episode rows into the conjunction and its marginals."""

    n = len(rows)
    survived = sum(1 for r in rows if r["survived"])
    stepped = sum(1 for r in rows if r["steps"] > 0)
    both = sum(1 for r in rows if r["survived"] and r["steps"] > 0)
    deaths: dict[str, int] = {}
    for row in rows:
        if not row["survived"]:
            deaths[str(row["termination"])] = deaths.get(str(row["termination"]), 0) + 1
    return {
        "episodes": n,
        "survived": survived,
        "stepped": stepped,
        "both": both,
        "mean_control_steps": statistics.fmean(r["control_steps"] for r in rows) if rows else 0.0,
        "max_steps": rows[0]["max_steps"] if rows else 0,
        "lifts": sum(r["lifts"] for r in rows),
        "steps": sum(r["steps"] for r in rows),
        "longest_step_mm": max((r["longest_step_mm"] for r in rows), default=0.0),
        "terminations": dict(sorted(deaths.items())),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="harness steps", description=__doc__)
    parser.add_argument("--policy", nargs="*", default=[],
                        help="one or more .cxpolicy files")
    parser.add_argument("--dir", help="a run directory; every .cxpolicy in it")
    parser.add_argument("--task", required=True, help="the run's OWN bundle")
    parser.add_argument("--profile", default="mg-legs")
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--seed-list", help="explicit comma-separated seeds")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--events", action="store_true",
                        help="print every lift, not just the totals")
    args = parser.parse_args(argv)

    policies = [Path(p).expanduser().resolve() for p in args.policy]
    if args.dir:
        found = sorted(Path(args.dir).expanduser().resolve().glob("*.cxpolicy"))
        policies.extend(p for p in found if p not in policies)
    if not policies:
        print("steps: no policies. Pass --policy FILE… or --dir RUN.",
              file=sys.stderr)
        return EXIT_USAGE

    seeds = ([int(s) for s in args.seed_list.split(",")] if args.seed_list
             else list(range(args.seeds)))

    try:
        bundle, bundle_sha = load_bundle(args.task)
        # Search beside the bundle first and beside each policy after, the
        # order `compare` uses: a run directory carries its own model copy,
        # and the match is on the bundle's own `model.sha256` rather than on
        # a filename, so the order only decides which of two identical files
        # is named in the envelope.
        seen: list[Path] = []
        for directory in [Path(args.task).expanduser().resolve().parent,
                          *(p.parent for p in policies)]:
            for xml in sorted(directory.glob("*model*.xml")):
                if xml not in seen:
                    seen.append(xml)
        model = resolve_model(bundle, seen)
        profile = load_profile(args.profile)
    except BundleError as exc:
        print(f"steps: {exc}", file=sys.stderr)
        return EXIT_USAGE

    job = {
        "model_xml": str(model),
        "task_path": str(Path(args.task).expanduser().resolve()),
        "profile": profile,
        "episodes": [
            {"id": f"{p.name}:{seed}", "policy": str(p), "seed": seed}
            for p in policies for seed in seeds
        ],
    }

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    def on_row(row: dict[str, Any]) -> None:
        if row.get("row") == "episode":
            rows.append(row)
        elif row.get("row") == "error":
            errors.append(row)
            print(f"  ! {row.get('id')}: {row.get('error')}", file=sys.stderr)

    try:
        run_bridge_job(STEPS_SCRIPT, job, schema="cdxrl-steps-job-v1",
                       on_row=on_row)
    except (TrainerVenvError, EpisodeJobFailed) as exc:
        print(f"steps: {exc}", file=sys.stderr)
        return EXIT_INFRASTRUCTURE

    by_policy: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_policy.setdefault(Path(row["policy"]).name, []).append(row)

    results = {name: score(rs) for name, rs in sorted(by_policy.items())}
    bound = significance_pp(len(seeds))

    payload = envelope("steps", not errors, {
        "task": str(args.task),
        "task_sha256": bundle_sha,
        "model": str(model),
        "profile": args.profile,
        "seeds": seeds,
        "metric": "stepped >= step_mm AND survived max_steps",
        "significance_pp_2sigma": round(bound, 1),
        "results": results,
        # THE PER-EPISODE ROWS, kept deliberately. Without them the envelope
        # supports only unpaired comparisons after the fact, and the paired
        # one is both more powerful and the one that matches how these
        # numbers are actually collected. Events are dropped — they are the
        # bulky part and `--events` prints them.
        "episodes": [{k: v for k, v in row.items() if k != "events"}
                     for row in rows],
        "errors": errors,
        # Empty unless CDXRL_ALLOW_UNPINNED_EVAL was set. A number measured
        # against versions other than the pinned ones carries the difference
        # rather than looking identical to one that was not.
        "unpinned_drift": dict(unpinned_drift),
    })

    if args.json:
        print(json.dumps(payload, indent=2))
        return EXIT_OK if not errors else EXIT_INFRASTRUCTURE

    print(f"task     {args.task}")
    print(f"profile  {args.profile}  "
          f"step >= {profile['step_mm']:g} mm, airborne >= "
          f"{profile['min_airborne_s'] * 1000:g} ms")
    print(f"seeds    {len(seeds)}  "
          f"(2σ on a difference: {bound:.0f} pp — enough to reject a "
          f"checkpoint, not to crown one)")
    print()
    print(f"{'checkpoint':34s} {'surv':>7} {'step':>7} {'BOTH':>7}  "
          f"{'mean':>6}  longest  terminations")
    for name, s in results.items():
        n = s["episodes"]
        print(f"{name:34s} {s['survived']:3d}/{n:<3d} {s['stepped']:3d}/{n:<3d} "
              f"{s['both']:3d}/{n:<3d}  {s['mean_control_steps']:6.1f}  "
              f"{s['longest_step_mm']:6.1f}mm  "
              f"{', '.join(f'{k} {v}' for k, v in s['terminations'].items())}")

    if args.events:
        print()
        for row in rows:
            for e in row["events"]:
                print(f"  {Path(row['policy']).name:28s} seed {row['seed']:2d} "
                      f"{e['foot']:7s} off {e['takeoff_s']:5.2f}s "
                      f"air {e['airborne_s']:4.2f}s "
                      f"travel {e['travel_mm']:6.1f}mm "
                      f"peak {e['peak_mm']:5.1f}mm"
                      f"{'  STEP' if e['step'] else ''}"
                      f"{'  (never landed)' if not e['landed'] else ''}")

    if len(results) > 1:
        ranked = sorted(results.items(), key=lambda kv: -kv[1]["both"])
        leader, top = ranked[0][0], ranked[0][1]["both"]
        print()
        print(f"leader on the conjunction: {leader} at {top}/{len(seeds)}")
        print(f"  Against the {bound:.0f} pp unpaired bound, everything "
              f"within {top - round(bound * len(seeds) / 100):d}"
              f"…{top}/{len(seeds)} is tied with it.")
        print("  But the seeds are PAIRED — same reset draw, same shove "
              "schedule — so the\n  unpaired bound is the wrong test and it "
              "is wrong conservatively. McNemar\n  over the discordant seeds "
              "only:")
        print(f"  {'against':34s} {'only ' + leader[:8]:>14s} {'only other':>12s} "
              f"{'disc':>5} {'p':>7}")
        for name, _ in ranked[1:]:
            m = paired(by_policy[leader], by_policy[name])
            print(f"  {name:34s} {m['only_first']:14d} {m['only_second']:12d} "
                  f"{m['discordant']:5d} {m['p_value']:7.3f}")
        print("  p is two-sided exact binomial. It says how much evidence "
              "there is; it does\n  not choose — install by the conjunction "
              "and say what the evidence was.")

    return EXIT_OK if not errors else EXIT_INFRASTRUCTURE
