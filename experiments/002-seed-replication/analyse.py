#!/usr/bin/env python
"""The morning. CPU-seconds; the GPU is already finished with.

```
uv run python experiments/002-seed-replication/analyse.py \
    --sweep jobs/stand9-sweep-20260803-010902 [--seeds 48] [--skip-capability]
```

Four run directories in, one `crossseed.json` and a pile of per-seed
envelopes out. Nothing here trains anything, and nothing here is allowed to
decide anything the README's §7 did not already name.

**Why a script rather than four sets of commands.** §2 fixed the cross-seed
statistics before dispatch — per seed, the reward-peak iteration, the
best-by-survival iteration, and the sign of (survival at final − survival at
reward peak). Typing those out four times invites reading the curves first
and choosing the summary second. This computes exactly the three that were
named, for every seed, whatever they say.

**48 seeds, not 12.** `compare --seeds 12` cannot crown a winner: survival is
binomial, so the worst-case 2σ bound on a *difference* is `√(2·0.25/n)`·2 —
20 pp at n=12, ~10 pp at n=48. 12 is enough to reject a checkpoint and not
enough to rank two. Every claim this script makes about "best" is printed
with that bound beside it, and anything inside it is reported **tied**.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

#: 200109's final policy — the recorded curve seed 0 is checked against.
BASELINE = Path("/home/theo/cadex-jobs/stand-task-20260802-200109/stand8.cxpolicy")


def run(command: list[str], *, capture: Path | None = None) -> dict[str, Any]:
    """Run a harness driver and return its parsed envelope."""

    print(f"  $ {' '.join(command)}", flush=True)
    finished = subprocess.run(
        command, cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    if capture is not None:
        capture.write_text(finished.stdout + finished.stderr, encoding="utf-8")
    if finished.returncode != 0:
        print(f"    exit {finished.returncode}")
        print("    " + "\n    ".join(finished.stderr.strip().splitlines()[-8:]))
    try:
        return json.loads(finished.stdout)
    except json.JSONDecodeError:
        return {"error": "no JSON envelope", "returncode": finished.returncode}


def two_sigma(n: int) -> float:
    """Worst-case 2σ bound on a *difference* of two binomial rates at n each."""

    return 2.0 * math.sqrt(2.0 * 0.25 / n)


def analyse_seed(
    run_dir: Path, seeds: int, skip_capability: bool, jobs: int | None = None,
) -> dict[str, Any]:
    label = run_dir.name
    print(f"\n=== {label} " + "=" * (54 - len(label)))
    facts: dict[str, Any] = {"run_dir": str(run_dir), "label": label}

    hyper = run_dir / "hyperparameters.json"
    if hyper.is_file():
        facts["hyperparameters"] = json.loads(hyper.read_text(encoding="utf-8"))
        facts["seed"] = facts["hyperparameters"].get("seed")

    bundle = next(iter(sorted(run_dir.glob("*-task.json"))), None)
    if bundle is None:
        facts["error"] = "no task bundle in the run directory"
        return facts
    facts["bundle"] = str(bundle)

    # 1 — the post-mortem. Witness margins, the curve, liveness, checkpoints.
    report = run(
        ["uv", "run", "python", "-m", "harness", "supervise",
         "--run", str(run_dir), "--report-only", "--json"],
        capture=RESULTS / f"{label}.supervise.txt",
    )
    data = report.get("data") or report
    facts["supervise"] = {
        "device": data.get("device"),
        "state": data.get("state"),
        "error": data.get("error"),
        "iteration": data.get("iteration"),
        "total": data.get("total"),
        "wall_time_s": data.get("wall_time_s"),
        "best_reward": data.get("best_reward"),
        "best_episode": data.get("best_episode"),
        "final": data.get("final"),
        "trend": (data.get("trend") or {}).get("verdict"),
        "witness": data.get("witness"),
        "witness_below_floor": data.get("witness_below_floor"),
        "series_points": data.get("series_points"),
        "checkpoints_missing": data.get("checkpoints_missing"),
        "checkpoints_mismatched": data.get("checkpoints_mismatched"),
    }
    (RESULTS / f"{label}.supervise.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # 2 — play every checkpoint. This is the measurement.
    envelope = run(
        ["uv", "run", "python", "-m", "harness", "compare",
         "--dir", str(run_dir), "--task", str(bundle),
         "--seeds", str(seeds), "--json",
         *(["--jobs", str(jobs)] if jobs else []),
         "--table", str(RESULTS / f"{label}.compare.table.json"),
         "--csv", str(RESULTS / f"{label}.compare.csv")],
    )
    payload = envelope.get("data") or envelope
    (RESULTS / f"{label}.compare.json").write_text(
        json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    facts["determinism"] = payload.get("determinism")
    facts["selection"] = payload.get("selection")
    facts["correlation"] = payload.get("correlation")
    table = payload.get("table") or []
    facts["checkpoints_played"] = len([row for row in table if row.get("episodes")])

    playable = [row for row in table if row.get("episodes")]
    with_trainer = [
        row for row in playable if row.get("trainer_reward_per_step") is not None
    ]

    # The three statistics §2 named, before the curves were visible.
    if with_trainer:
        peak = max(with_trainer,
                   key=lambda row: float(row["trainer_reward_per_step"]))
        facts["reward_peak"] = {
            "iteration": peak["iteration"], "name": peak["name"],
            "trainer_reward_per_step": peak["trainer_reward_per_step"],
            "survival": peak["survival"],
        }
    if playable:
        best = max(playable, key=lambda row: (row["survival"], row["steps_mean"] or 0.0))
        facts["best_by_survival"] = {
            "iteration": best["iteration"], "name": best["name"],
            "survival": best["survival"], "steps_mean": best["steps_mean"],
        }
        final = max(playable, key=lambda row: row["iteration"])
        facts["final_checkpoint"] = {
            "iteration": final["iteration"], "name": final["name"],
            "survival": final["survival"],
        }

    bound = two_sigma(seeds)
    facts["separation_2sigma"] = round(bound, 4)
    if facts.get("reward_peak") and facts.get("best_by_survival"):
        gap = facts["best_by_survival"]["survival"] - facts["reward_peak"]["survival"]
        facts["peak_vs_best"] = {
            "gap": round(gap, 4),
            "separated": bool(abs(gap) > bound),
            # The headline. "Not the best checkpoint" only counts when the
            # difference clears the bound; inside it, the honest answer is tied.
            "reward_peak_is_not_best": bool(gap > bound),
            "verdict": (
                "reward peak is NOT the best checkpoint" if gap > bound
                else "tied — inside the 2σ bound" if abs(gap) <= bound
                else "reward peak IS the best checkpoint"
            ),
        }
    if facts.get("reward_peak") and facts.get("final_checkpoint"):
        delta = facts["final_checkpoint"]["survival"] - facts["reward_peak"]["survival"]
        facts["final_minus_peak"] = {
            "delta": round(delta, 4),
            "sign": (0 if abs(delta) <= bound else (1 if delta > 0 else -1)),
            "separated": bool(abs(delta) > bound),
        }

    # 3 — what the task was asking of the checkpoint that actually survived best.
    if not skip_capability and facts.get("best_by_survival"):
        policy = run_dir / facts["best_by_survival"]["name"]
        if policy.is_file():
            cap = run(
                ["uv", "run", "python", "-m", "harness", "capability",
                 "--policy", str(policy), "--task", str(bundle),
                 "--seeds", str(seeds), "--json",
                 *(["--jobs", str(jobs)] if jobs else []),
                 "--table", str(RESULTS / f"{label}.capability.table.json")],
            )
            (RESULTS / f"{label}.capability.json").write_text(
                json.dumps(cap, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            facts["capability"] = (cap.get("data") or cap).get("sweep")

    return facts


def main() -> int:
    parser = argparse.ArgumentParser(prog="analyse.py")
    parser.add_argument("--sweep", required=True, help="The sweep directory.")
    parser.add_argument("--seeds", type=int, default=48)
    parser.add_argument("--skip-capability", action="store_true")
    parser.add_argument(
        "--jobs", type=int, default=None,
        help="Worker processes per driver. Left alone, compare uses cores-2; "
             "cap it when the box is busy with something else.",
    )
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    sweep_dir = Path(args.sweep).expanduser().resolve()
    manifest = json.loads((sweep_dir / "sweep.json").read_text(encoding="utf-8"))

    run_dirs = [Path(item["run_dir"]) for item in manifest["runs"]
                if item.get("run_dir")]
    print(f"sweep {sweep_dir}")
    print(f"  {len(run_dirs)} run(s), seeds {manifest['seeds']}, "
          f"{args.seeds} evaluation seeds per checkpoint")
    print(f"  2σ bound on a survival difference at n={args.seeds}: "
          f"{two_sigma(args.seeds) * 100:.1f} pp")

    seeds_facts = [analyse_seed(path, args.seeds, args.skip_capability, args.jobs)
                   for path in run_dirs if path.is_dir()]

    # The cross-seed answer, computed the way §2 said it would be.
    decided = [item for item in seeds_facts if item.get("peak_vs_best")]
    not_best = [item for item in decided
                if item["peak_vs_best"]["reward_peak_is_not_best"]]
    tied = [item for item in decided
            if not item["peak_vs_best"]["separated"]]

    summary = {
        "sweep_dir": str(sweep_dir),
        "evaluation_seeds": args.seeds,
        "separation_2sigma": round(two_sigma(args.seeds), 4),
        "seeds": seeds_facts,
        "headline": {
            "seeds_decided": len(decided),
            "reward_peak_not_best": len(not_best),
            "tied_inside_bound": len(tied),
            "statement": (
                f"In {len(not_best)} of {len(decided)} seeds the trainer's "
                f"reward peak is not the best checkpoint by measured "
                f"survival, at a 2σ separation bound of "
                f"{two_sigma(args.seeds) * 100:.1f} pp."
            ),
        },
    }
    (RESULTS / "crossseed.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\n" + "=" * 64)
    print("cross-seed")
    print(f"  {'seed':>4} {'reward peak':>12} {'best survival':>14} "
          f"{'gap':>8}  verdict")
    for item in seeds_facts:
        peak = item.get("reward_peak") or {}
        best = item.get("best_by_survival") or {}
        gap = (item.get("peak_vs_best") or {}).get("gap")
        print(f"  {str(item.get('seed')):>4} "
              f"{str(peak.get('iteration', '—')):>6}"
              f"@{peak.get('survival', float('nan')):.2f} "
              f"{str(best.get('iteration', '—')):>8}"
              f"@{best.get('survival', float('nan')):.2f} "
              f"{gap if gap is not None else float('nan'):>+8.3f}  "
              f"{(item.get('peak_vs_best') or {}).get('verdict', '—')}")
    print()
    print(f"  {summary['headline']['statement']}")
    print(f"\n  wrote {RESULTS / 'crossseed.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
