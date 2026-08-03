"""``compare`` — choose a checkpoint by what it did, not by what it scored.

```
compare --dir RUNDIR --task BUNDLE [--seeds N] [--jobs N] [--json]
```

Plays **every** ``.cxpolicy`` in a directory against several seeds — stock
MuJoCo through the engine's own reference runner, no GPU, seconds — and
prints one table.

### Why this driver is the one that makes any checkpoint claim meaningful

ADR-099 measured the trainer's reward and real survival as *anti-correlated
across a whole run*: 12/12 survival where the trainer reported its worst
numbers, 0/12 exactly where it reported its best. The checkpoint the trainer
labelled ``best`` fell in 43 steps of 600, from every seed and every
direction, before the first shove window even opened. Nothing about that is
visible in ``progress.json``, and the only way to find it is to play the
file.

### Three behaviours that are not options

**`--task` is mandatory and means the run's own bundle.** A rebuild is keyed
by script digest and replaces ``script_artifacts/``, so a finished run's task
can vanish from the project store while its checkpoints sit beside you.
There is no fallback to "the project's current task", because that fallback
silently answers a different question.

**The model is matched by digest**, against the bundle's own
``model.sha256``. A policy played against the wrong MJCF still verifies and
still produces numbers.

**The same-file-twice test runs first and fails the whole invocation.**
ADR-103 §9: ``evaluate_episode`` multiplies domain randomisation into the
model in place and keeps no baseline, so an evaluator that reuses one loaded
model compounds every draw it has made and its rows drift. The test plays one
checkpoint twice **through a single worker** — ``--jobs 1`` — because a pool
that happened to give each replica a fresh process would pass a test the bug
is designed to fail.

### The torque columns are not optional and not a flag

Hazard 15 is a policy that plays as a clean stand while holding three of
eight motors above 95 % of a servo's *stall* rating on 100 % of frames.
Nothing in the trajectory shows it, nothing in the reward shows it, and the
mechanism it describes cannot be built. Peak, mean and per-cent-of-frames
above 90 % of the limit are printed for **every motor of every checkpoint**.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from harness import EXIT_INFRASTRUCTURE, EXIT_OK, EXIT_USAGE
from harness.episodes import (
    BundleError,
    aggregate,
    discover_policies,
    flags,
    load_bundle,
    resolve_model,
)
from harness.provenance import envelope, load_env_file, sha256_file
from harness.trainer_venv import EpisodeJobFailed, TrainerVenvError, run_episode_job

#: What the determinism test compares. Not the whole row: the torque columns
#: are floats summed over hundreds of steps and comparing them exactly would
#: make the test about floating-point association rather than about state
#: leaking between episodes. These three are integers and an exact sum, and
#: any of the leak's symptoms moves at least one of them.
DETERMINISM_KEYS = ("step_count", "termination", "truncated")


def determinism_check(
    job: dict[str, Any], policy: dict[str, Any], seeds: list[int]
) -> dict[str, Any]:
    """Play one file twice, in one worker, and demand identical rows.

    ``--jobs 1`` on purpose: see the module docstring. The whole point is
    that the second pass runs in a process that has already played the first.
    """

    episodes = [
        {"id": f"{pass_}-{seed}", "policy": policy["path"],
         "variant": "declared", "seed": seed}
        for pass_ in ("a", "b") for seed in seeds
    ]
    rows = run_episode_job({**job, "jobs": 1, "episodes": episodes, "verify": []})
    by_id = {row["id"]: row for row in rows if row.get("row") == "episode"}

    disagreements = []
    for seed in seeds:
        first, second = by_id.get(f"a-{seed}"), by_id.get(f"b-{seed}")
        if first is None or second is None:
            disagreements.append({"seed": seed, "error": "an episode is missing"})
            continue
        left = {key: first[key] for key in DETERMINISM_KEYS}
        right = {key: second[key] for key in DETERMINISM_KEYS}
        left["total_reward"] = round(first["total_reward"], 9)
        right["total_reward"] = round(second["total_reward"], 9)
        if left != right:
            disagreements.append({"seed": seed, "first": left, "second": right})

    return {
        "policy": policy["name"],
        "seeds": seeds,
        "episodes": len(episodes),
        "ok": not disagreements,
        "disagreements": disagreements,
    }


def run(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    run_dir = Path(args.dir).expanduser().resolve()
    if not run_dir.is_dir():
        return EXIT_USAGE, {"error": f"no such run directory: {run_dir}"}

    try:
        bundle, task_sha256 = load_bundle(args.task)
        task_path = Path(args.task).expanduser().resolve()
        model = resolve_model(
            bundle,
            [*sorted(run_dir.glob("*-model.xml")),
             *sorted(task_path.parent.glob("*-model.xml"))],
        )
    except BundleError as exc:
        return EXIT_USAGE, {"error": str(exc)}

    policies = discover_policies(run_dir)
    if args.only:
        wanted = set(args.only)
        policies = [item for item in policies if item["name"] in wanted]
    if not policies:
        return EXIT_USAGE, {"error": f"no .cxpolicy files in {run_dir}"}

    seeds = list(range(args.seeds))
    base_job = {
        "model_xml": str(model),
        "task_path": str(task_path),
        "task_sha256": task_sha256,
        "variants": {"declared": {}},
    }

    payload: dict[str, Any] = {
        "run_dir": str(run_dir),
        "task": str(task_path),
        "task_sha256": task_sha256,
        "model": str(model),
        "model_sha256": sha256_file(model),
        "seeds": seeds,
        "max_steps": int(bundle["episode"]["max_steps"]),
        "actuators": [action["actuator"] for action in bundle["actions"]],
        "limits_nmm": [
            max(abs(float(a["low"])), abs(float(a["high"]))) for a in bundle["actions"]
        ],
        "policies": policies,
    }

    # First, before a single number is trusted.
    reference = next(
        (item for item in policies if item["role"] == "best"), policies[-1]
    )
    started = time.monotonic()
    try:
        payload["determinism"] = determinism_check(base_job, reference, seeds)
    except (TrainerVenvError, EpisodeJobFailed) as exc:
        return EXIT_INFRASTRUCTURE, {**payload, "error": str(exc)}
    if not payload["determinism"]["ok"]:
        payload["error"] = (
            "The same file played twice produced different rows. Every number "
            "this driver would print is untrustworthy until that is fixed "
            "(ADR-103 §9)."
        )
        return EXIT_INFRASTRUCTURE, payload

    episodes = [
        {"id": f"{item['name']}#{seed}", "policy": item["path"],
         "variant": "declared", "seed": seed}
        for item in policies for seed in seeds
    ]
    progress = {"done": 0}

    def tick(row: dict[str, Any]) -> None:
        if row.get("row") == "episode":
            progress["done"] += 1
            if progress["done"] % 100 == 0 or progress["done"] == len(episodes):
                print(f"  … {progress['done']}/{len(episodes)} episodes",
                      file=sys.stderr)

    try:
        rows = run_episode_job(
            {**base_job, "jobs": args.jobs,
             "verify": [item["path"] for item in policies],
             "episodes": episodes},
            on_row=tick,
        )
    except (TrainerVenvError, EpisodeJobFailed) as exc:
        return EXIT_INFRASTRUCTURE, {**payload, "error": str(exc)}
    payload["wall_seconds"] = round(time.monotonic() - started, 2)

    verified = {
        row["policy"]: row for row in rows if row.get("row") == "verify"
    }
    refused = [
        {"policy": Path(path).name, "error": row.get("error")}
        for path, row in verified.items() if not row.get("ok")
    ]
    payload["verify_refused"] = refused

    by_policy: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("row") == "episode":
            by_policy.setdefault(row["policy"], []).append(row)

    table: list[dict[str, Any]] = []
    for item in policies:
        group = by_policy.get(item["path"], [])
        if not group:
            table.append({
                **item, "episodes": 0,
                "skipped": "verify_policy refused this container",
            })
            continue
        summary = aggregate(group, bundle)
        summary.update(item)
        summary["flags"] = flags(summary)
        summary["witness"] = {
            key: (verified.get(item["path"], {}).get("detail") or {}).get(key)
            for key in ("witness_error", "witness_tolerance", "parameters")
        }
        table.append(summary)
    payload["table"] = table

    playable = [row for row in table if row.get("episodes")]
    if playable:
        best_survival = max(playable, key=lambda row: (
            row["survival"], row["steps_mean"] or 0.0))
        best_steps = max(playable, key=lambda row: row["steps_mean"] or 0.0)
        by_reward = max(playable, key=lambda row: row["reward_per_step_mean"] or -1e9)
        payload["selection"] = {
            "best_by_survival": {
                key: best_survival[key] for key in
                ("name", "iteration", "survival", "steps_mean", "role")
            },
            "best_by_episode_length": {
                key: best_steps[key] for key in
                ("name", "iteration", "survival", "steps_mean", "role")
            },
            "best_by_measured_reward": {
                key: by_reward[key] for key in
                ("name", "iteration", "survival", "steps_mean", "role")
            },
            "trainer_best_file": next(
                (row["name"] for row in playable if row["role"] == "best"), ""
            ),
            **separation(playable, len(seeds)),
        }
        # The correlation the whole experiment is about, computed rather than
        # eyeballed off the table.
        usable = [row for row in playable if row["iteration"] >= 0]
        if len(usable) > 2:
            survivals = [row["survival"] for row in usable]
            payload["correlation"] = {
                "survival_vs_iteration": _pearson(
                    [float(row["iteration"]) for row in usable], survivals),
                "survival_vs_measured_reward": _pearson(
                    [row["reward_per_step_mean"] or 0.0 for row in usable], survivals),
                "n": len(usable),
            }
            # ADR-099's claim, restated against this run rather than assumed.
            with_trainer = [
                row for row in usable if row.get("trainer_reward_per_step") is not None
            ]
            if len(with_trainer) > 2:
                payload["correlation"]["survival_vs_trainer_reward"] = _pearson(
                    [float(row["trainer_reward_per_step"]) for row in with_trainer],
                    [row["survival"] for row in with_trainer],
                )
                payload["correlation"]["n_with_trainer_reward"] = len(with_trainer)

                # And again from the trainer's own peak onward, which is the
                # only span a stopping rule gets to act on. A whole-run r
                # averages the early stretch — where reward and survival both
                # climb because the network is going from nothing to
                # something — together with the span the rule is about, and
                # the two can cancel to a number that means neither.
                peak = max(with_trainer,
                           key=lambda row: float(row["trainer_reward_per_step"]))
                after = [
                    row for row in with_trainer
                    if row["iteration"] >= peak["iteration"]
                ]
                if len(after) > 2:
                    payload["correlation"]["post_peak"] = {
                        "from_iteration": peak["iteration"],
                        "n": len(after),
                        "survival_vs_trainer_reward": _pearson(
                            [float(row["trainer_reward_per_step"]) for row in after],
                            [row["survival"] for row in after],
                        ),
                        "survival_vs_iteration": _pearson(
                            [float(row["iteration"]) for row in after],
                            [row["survival"] for row in after],
                        ),
                    }

    ok = not refused and bool(playable)
    return (EXIT_OK if ok else EXIT_INFRASTRUCTURE), payload


def separation(rows: list[dict[str, Any]], seeds: int) -> dict[str, Any]:
    """Is the winner actually separated from the runner-up, at this many seeds?

    **Added because the sweep found the answer was no.** ``compare`` at 12
    seeds ranked iteration 1699 above the final network, 7/12 against 6/12 —
    a one-episode margin. At 48 seeds the order reverses (23/48 against
    27/48), and neither gap is significant. Both are decisively above the
    trainer's reward peak, and that is the whole of what the table supports.

    Survival is a binomial proportion, so the standard error of a difference
    of two independent ones is at most ``sqrt(2 · 0.25 / n)``. At n = 12 that
    is 20 percentage points, which is 2.4 episodes: **a 12-seed table cannot
    tell two checkpoints apart unless they differ by about half their range.**
    Reporting a winner without saying so invites a 3.9-GPU-hour decision to
    be made on a coin toss.
    """

    ranked = sorted(rows, key=lambda row: row["survival"], reverse=True)
    if len(ranked) < 2 or seeds < 1:
        return {}
    best, runner = ranked[0], ranked[1]
    gap = best["survival"] - runner["survival"]
    # Worst-case standard error of the difference, at p = 0.5.
    error = math.sqrt(2.0 * 0.25 / seeds)
    separated = gap > 2.0 * error
    tied = [row["name"] for row in ranked if best["survival"] - row["survival"] <= 2.0 * error]
    return {
        "separation": {
            "seeds": seeds,
            "gap": round(gap, 4),
            "two_sigma": round(2.0 * error, 4),
            "separated": separated,
            "indistinguishable": tied,
        }
    }


def _pearson(left: list[float], right: list[float]) -> float | None:
    """Plain Pearson r. Returns ``None`` where it is undefined rather than 0.0,
    because a zero correlation and an unanswerable question are different
    findings and the table must not conflate them."""

    if len(left) < 3:
        return None
    try:
        return round(statistics.correlation(left, right), 4)
    except statistics.StatisticsError:
        return None


# ---------------------------------------------------------------------------
# Prose
# ---------------------------------------------------------------------------


def _mix(mix: dict[str, int]) -> str:
    return " ".join(f"{label}:{count}" for label, count in mix.items())


def report(payload: dict[str, Any]) -> None:
    if payload.get("error") and not payload.get("table"):
        print(f"ERROR  {payload['error']}")
        return

    print(f"run dir   {payload['run_dir']}")
    print(f"task      {payload['task']}")
    print(f"          sha256 {payload['task_sha256']}")
    print(f"model     {payload['model']}")
    print(f"          sha256 {payload['model_sha256']}  (matches the bundle's own)")
    print(f"seeds     {len(payload['seeds'])}  ({payload['seeds'][0]}…"
          f"{payload['seeds'][-1]})   budget {payload['max_steps']} steps")
    limits = payload["limits_nmm"]
    if limits and len(set(limits)) == 1:
        print(f"motors    {len(payload['actuators'])} × ±{limits[0]:.0f} N·mm")
    else:
        print(f"motors    {len(payload['actuators'])}, limits {limits}")
    if payload.get("wall_seconds"):
        print(f"wall      {payload['wall_seconds']:.1f} s")

    check = payload.get("determinism") or {}
    print()
    print(f"same-file-twice (ADR-103 §9)  {check.get('policy')}  "
          f"{check.get('episodes')} episodes, one worker")
    if check.get("ok"):
        print("  PASS — the evaluator reloads the model per episode and its "
              "rows do not drift.")
    else:
        print("  FAIL — the same file played twice produced different rows:")
        for item in check.get("disagreements", [])[:5]:
            print(f"    {json.dumps(item, sort_keys=True)}")
        if payload.get("error"):
            print(f"  {payload['error']}")
            return

    if payload.get("verify_refused"):
        print()
        print("verify_policy REFUSED these containers — reported, not dropped:")
        for item in payload["verify_refused"]:
            print(f"  {item['policy']:<26} {item['error']}")

    table = payload.get("table") or []
    print()
    print("survival, and what it cost   (trainer r/s is what the trainer "
          "recorded; every other column is measured here)")
    print(f"  {'checkpoint':<24} {'iter':>6} {'surv':>7} {'steps µ':>8} "
          f"{'med':>5} {'tilt':>7} {'drift':>8} {'reward':>9} {'train r/s':>10}"
          f"  termination mix")
    for row in table:
        if not row.get("episodes"):
            print(f"  {row['name']:<24} {row['iteration']:>6}   SKIPPED  "
                  f"{row.get('skipped', '')}")
            continue
        trainer = row.get("trainer_reward_per_step")
        print(
            f"  {row['name']:<24} {row['iteration']:>6} "
            f"{row['survived']:>3}/{row['episodes']:<3} "
            f"{row['steps_mean']:>8.1f} {row['steps_median']:>5.0f} "
            f"{(row['final_tipped_mean'] if row['final_tipped_mean'] is not None else float('nan')):>7.3f} "
            f"{(row['drift_mean'] if row['drift_mean'] is not None else float('nan')):>8.1f} "
            f"{row['reward_mean']:>9.1f} "
            f"{(f'{trainer:+.4f}' if trainer is not None else '—'):>10}"
            f"  {_mix(row['termination_mix'])}"
        )
    print("  tilt is the task's own `tipped` expression on the last "
          "observation; it terminates above 0.15.")
    print("  drift is the task's own `drift` reward expression, in mm, "
          "evaluated through compile_reward.")

    motors = payload["actuators"]
    for title, key, scale, note in (
        ("peak torque per motor, N·mm", "peak_torque_nmm", 1.0,
         "the worst instant any of the seeds saw"),
        ("mean |torque| per motor, N·mm", "mean_torque_nmm", 1.0,
         "averaged over frames, then over episodes"),
        ("% of frames above 90 % of limit, per motor", "frac_above_90pct", 100.0,
         "hazard 15's column: a clean-looking stand can live here"),
    ):
        print()
        print(f"{title}   ({note})")
        print(f"  {'checkpoint':<24} " + " ".join(
            f"{name.split('/')[0][:6]:>6}" for name in motors))
        if scale == 1.0:
            print(f"  {'limit N·mm':<24} " + " ".join(
                f"{limit:>6.0f}" for limit in limits))
        for row in table:
            if not row.get("episodes"):
                continue
            print(f"  {row['name']:<24} " + " ".join(
                f"{value * scale:>6.1f}" for value in row[key]))

    flagged = [row for row in table if row.get("flags")]
    print()
    print("flags")
    if not flagged:
        print("  none. No checkpoint is commanding near-zero torque, and no "
              "motor is pinned at its limit.")
    for row in flagged:
        for note in row["flags"]:
            print(f"  {row['name']:<24} {note}")

    selection = payload.get("selection")
    if selection:
        print()
        print("selection, by measured survival (ADR-099 — not by the trainer's "
              "scalar)")
        for label, key in (
            ("best by survival", "best_by_survival"),
            ("longest episodes", "best_by_episode_length"),
            ("best measured reward", "best_by_measured_reward"),
        ):
            item = selection[key]
            print(f"  {label:<22} {item['name']:<24} iteration "
                  f"{item['iteration']:>6}  survival "
                  f"{item['survival'] * 100:.0f}%  steps {item['steps_mean']:.1f}")
        print(f"  the trainer's own 'best' file was {selection['trainer_best_file']}")
        gap = selection.get("separation")
        if gap:
            print()
            if gap["separated"]:
                print(f"  the winner IS separated from the runner-up: gap "
                      f"{gap['gap'] * 100:.0f} pp against a 2-sigma bound of "
                      f"{gap['two_sigma'] * 100:.0f} pp at {gap['seeds']} seeds.")
            else:
                print(f"  ** THE WINNER IS NOT SEPARATED FROM THE RUNNER-UP. **")
                print(f"  Gap {gap['gap'] * 100:.0f} pp against a 2-sigma bound of "
                      f"{gap['two_sigma'] * 100:.0f} pp at {gap['seeds']} seeds "
                      f"(survival is binomial; at n={gap['seeds']} the standard "
                      f"error of a difference is worst-case "
                      f"{gap['two_sigma'] / 2 * 100:.0f} pp).")
                print(f"  Indistinguishable at this seed count: "
                      f"{', '.join(gap['indistinguishable'])}")
                print("  Raise --seeds before treating the top row as the answer.")

    correlation = payload.get("correlation")
    if correlation:
        print()
        print("correlation over the checkpoints played")
        print(f"  survival vs iteration            r = {correlation['survival_vs_iteration']}")
        print(f"  survival vs measured reward/step r = {correlation['survival_vs_measured_reward']}")
        if "survival_vs_trainer_reward" in correlation:
            print(f"  survival vs TRAINER reward/step  r = "
                  f"{correlation['survival_vs_trainer_reward']}"
                  f"   ← ADR-099's claim, on this run "
                  f"(n = {correlation['n_with_trainer_reward']})")
        print(f"  (n = {correlation['n']} checkpoints)")
        after = correlation.get("post_peak")
        if after:
            print(f"  from the trainer's peak (iteration {after['from_iteration']}) "
                  f"onward, n = {after['n']} — the only span a stopping rule acts on:")
            print(f"    survival vs TRAINER reward/step  r = "
                  f"{after['survival_vs_trainer_reward']}")
            print(f"    survival vs iteration            r = "
                  f"{after['survival_vs_iteration']}")


def tabulate(payload: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
    """The table as columns and rows, once, for both serialisations.

    The prose report is for reading and the envelope is for a program; this
    is for neither, and it exists because a *table* artifact should be one.
    One row per checkpoint, one column per motor per torque measure, so the
    per-motor detail survives into a spreadsheet rather than only into JSON.

    **Flywheel validates uploaded bytes against the declared
    ``artifact_type``, and ``table`` means JSON.** Measured on 2026-08-02:
    an array of row objects and a ``{columns, rows}`` document are both
    accepted; CSV and TSV are both refused with *"uploaded artifact bytes
    are not valid for requested artifact type"*. Hence two writers over one
    builder — :func:`write_table` for the graph, :func:`write_csv` for a
    spreadsheet.
    """

    motors = [name.split("/")[0] for name in payload["actuators"]]
    header = [
        "checkpoint", "iteration", "role", "episodes", "survived", "survival",
        "steps_mean", "steps_median", "steps_min", "steps_max", "max_steps",
        "final_tipped_mean", "drift_mean", "reward_mean", "reward_per_step_mean",
        "trainer_reward_per_step", "term_survived", "term_tipped", "term_collapsed",
        "peak_frac_of_limit", "mean_frac_of_limit", "saturation_worst", "flags",
    ]
    header += [f"peak_nmm[{name}]" for name in motors]
    header += [f"mean_nmm[{name}]" for name in motors]
    header += [f"above90pct[{name}]" for name in motors]

    rows: list[list[Any]] = []
    for row in payload.get("table") or []:
        if not row.get("episodes"):
            rows.append([row["name"], row["iteration"], row.get("role", ""), 0]
                        + [None] * 18 + [row.get("skipped", "skipped")]
                        + [None] * (3 * len(motors)))
            continue
        mix = row["termination_mix"]
        rows.append([
            row["name"], row["iteration"], row.get("role", ""),
            row["episodes"], row["survived"], round(row["survival"], 4),
            round(row["steps_mean"], 2), row["steps_median"],
            row["steps_min"], row["steps_max"], row["max_steps"],
            None if row["final_tipped_mean"] is None else round(row["final_tipped_mean"], 5),
            None if row["drift_mean"] is None else round(row["drift_mean"], 3),
            round(row["reward_mean"], 3), round(row["reward_per_step_mean"], 5),
            row.get("trainer_reward_per_step"),
            mix.get("survived", 0), mix.get("tipped", 0), mix.get("collapsed", 0),
            round(row["peak_frac_of_limit"], 4),
            round(row["mean_frac_of_limit"], 4),
            round(row["saturation_worst"], 5),
            "; ".join(row.get("flags") or []),
        ] + list(row["peak_torque_nmm"])
          + list(row["mean_torque_nmm"])
          + list(row["frac_above_90pct"]))
    return header, rows


def write_table(payload: dict[str, Any], path: Path) -> None:
    """``{columns, rows}`` JSON — the shape a Flywheel ``table`` artifact takes."""

    columns, rows = tabulate(payload)
    document = {
        "schema": "cdxrl-compare-table-v1",
        "run_dir": payload["run_dir"],
        "task_sha256": payload["task_sha256"],
        "seeds": len(payload["seeds"]),
        "columns": columns,
        "rows": rows,
    }
    path.write_text(json.dumps(document, indent=1, sort_keys=False) + "\n",
                    encoding="utf-8")


def write_csv(payload: dict[str, Any], path: Path) -> None:
    """The same table, for a spreadsheet rather than for the graph."""

    import csv

    columns, rows = tabulate(payload)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harness compare",
        description="Play every checkpoint in a run directory and compare them.",
    )
    parser.add_argument("--dir", required=True, help="A run directory of .cxpolicy files.")
    parser.add_argument(
        "--task", required=True,
        help="The run's OWN task bundle. Mandatory: a rebuild replaces "
             "script_artifacts/ and there is no safe fallback.",
    )
    parser.add_argument("--seeds", type=int, default=12, help="Seeds per checkpoint.")
    parser.add_argument("--jobs", type=int, default=0, help="Worker processes (0 = cores − 2).")
    parser.add_argument(
        "--only", action="append", default=[], metavar="NAME",
        help="Play only these checkpoint filenames. Repeatable.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the envelope.")
    parser.add_argument(
        "--csv", metavar="PATH",
        help="Also write the table as CSV, one row per checkpoint and one "
             "column per motor per torque measure.",
    )
    parser.add_argument(
        "--table", metavar="PATH",
        help="Also write the table as a {columns, rows} JSON document — the "
             "shape a Flywheel `table` artifact must be in (CSV is refused).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_env_file()
    args = build_parser().parse_args(argv)
    code, payload = run(args)
    if args.csv and payload.get("table"):
        write_csv(payload, Path(args.csv).expanduser())
        print(f"csv written to {args.csv}", file=sys.stderr)
    if args.table and payload.get("table"):
        write_table(payload, Path(args.table).expanduser())
        print(f"table written to {args.table}", file=sys.stderr)
    if args.json:
        json.dump(envelope("compare", code == EXIT_OK, payload),
                  sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        report(payload)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
