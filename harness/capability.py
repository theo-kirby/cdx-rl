"""``capability`` — what was the task even asking?

```
capability --policy FILE [--policy FILE …] --task BUNDLE
           [--scales …] [--seeds N] [--jobs N] [--json]
```

The driver ADR-106 is named after, and **the one that must run before anyone
writes "the policy failed to learn X"**. A policy that survives 0/12 of its
declared task has told you nothing about itself until you know whether 12/12
was reachable.

It scales the declared shove band by mutating a **copy** of the task's
``disturbance[].newtons_low`` / ``newtons_high`` and sweeps. Default scales
`[0.0, 0.15, 0.30, 0.50, 0.75, 1.00]`, plus two rows that are not scales at
all:

``reset-variation-only``
    the disturbance list removed entirely, the reset variation kept. "Can it
    recover from where it starts, with nothing pushing it."
``no-variation``
    reset variation, disturbance and domain randomisation all removed. "Can
    it stand at all." If this row is not near-perfect, nothing further down
    the sweep means what it appears to mean.

Note that ``no-disturbance`` and ``scale 0.0`` are **different rows on
purpose**. At scale 0.0 the schedule still runs — it draws a start time, an
azimuth and a zero-magnitude force — so the episode is otherwise identical.
Removing the list removes the schedule. Two rows that should agree, and a
cheap check on the evaluator if they ever do not.

### What it prints, and why each column had to be fought for

* **survival, split by azimuth.** The drawn ``azimuth_rad`` comes back on
  every episode, so the split costs nothing. A machine with ankle roll can be
  fine sagittally and hopeless laterally, and a single survival fraction
  averages exactly that away.
* **the termination mix**, always, never behind a flag (ADR-106 collected it
  from M9 onward and never showed it; that cost three runs).
* **how far into its own disturbance schedule each death got.** The drawn
  ``start_s`` against the death time is what revealed, in ADR-106, that the
  second shove window had never once been exercised — so half the episode's
  design had never run and no amount of training could have been measured
  against it.

### It must say when it measured nothing

A curve that is flat across the whole sweep — 0/12 everywhere, or 12/12
everywhere — **is not a finding**. A 50-iteration sanity run reads 0/12 at
every scale and looks exactly like a mechanism that cannot be helped. This
driver says so out loud, on its own output, rather than leaving it to be
noticed.
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
    flags,
    load_bundle,
    resolve_model,
)
from harness.provenance import envelope, load_env_file, sha256_file
from harness.trainer_venv import EpisodeJobFailed, TrainerVenvError, run_episode_job

DEFAULT_SCALES = (0.0, 0.15, 0.30, 0.50, 0.75, 1.00)

#: Quadrants rather than octants. With a dozen seeds a row an octant split
#: puts one or two episodes in a bin, and a "0/1 survival" cell reads as a
#: result when it is a coin. Every bin prints its own n so the sparsity is
#: visible rather than implied.
AZIMUTH_BINS = (
    ("+x  (0°–90°)", 0.0, 0.5 * math.pi),
    ("+y  (90°–180°)", 0.5 * math.pi, math.pi),
    ("-x  (180°–270°)", math.pi, 1.5 * math.pi),
    ("-y  (270°–360°)", 1.5 * math.pi, 2.0 * math.pi),
)


def build_variants(scales: list[float]) -> dict[str, dict[str, Any]]:
    """The sweep's rows, in the order they should be read.

    The two establishing rows come first deliberately: a reader who sees
    ``no-variation`` at 12/12 and ``scale 1.00`` at 0/12 has learned
    something, and a reader who sees both at 0/12 has learned that the sweep
    measured nothing.
    """

    variants: dict[str, dict[str, Any]] = {
        "no-variation": {
            "disturbance": False, "reset_variation": False, "randomisation": False,
        },
        "reset-variation-only": {"disturbance": False},
    }
    for scale in scales:
        variants[f"scale-{scale:.2f}"] = {"disturbance_scale": scale}
    return variants


def azimuth_split(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Survival by the drawn azimuth of the **first non-sustained** shove.

    The first one, not the mean of all of them: a sustained wind has an
    azimuth too and averaging it in would smear the direction that actually
    knocked the machine over.
    """

    bins: list[dict[str, Any]] = [
        {"label": label, "low": low, "high": high, "n": 0, "survived": 0}
        for label, low, high in AZIMUTH_BINS
    ]
    unbinned = 0
    for row in rows:
        azimuth = None
        for drawn in row.get("disturbances") or []:
            if drawn.get("azimuth_rad") is not None and drawn.get("start_s") is not None:
                azimuth = float(drawn["azimuth_rad"]) % (2.0 * math.pi)
                break
        if azimuth is None:
            unbinned += 1
            continue
        for record in bins:
            if record["low"] <= azimuth < record["high"]:
                record["n"] += 1
                record["survived"] += 1 if row["truncated"] else 0
                break
    for record in bins:
        record["survival"] = (
            record["survived"] / record["n"] if record["n"] else None
        )
    return bins


def schedule_reach(rows: list[dict[str, Any]], bundle: dict[str, Any]
                   ) -> list[dict[str, Any]]:
    """Per declared disturbance: did the episode live to see it?

    ADR-106's discovery, made a column. ``reached`` counts episodes whose end
    time was at or past the drawn ``start_s``; ``died_before`` is the rest.
    A window that is never reached is a window that measured nothing, however
    carefully it was declared — and the fix for that is a shorter schedule or
    a longer-surviving policy, not more training against a window that never
    opens.
    """

    labels = [str(item.get("label") or "") for item in bundle.get("disturbance") or []]
    summary: list[dict[str, Any]] = []
    for label in labels:
        reached = 0
        total = 0
        starts: list[float] = []
        for row in rows:
            for drawn in row.get("disturbances") or []:
                if str(drawn.get("label") or "") != label:
                    continue
                total += 1
                if drawn.get("start_s") is not None:
                    starts.append(float(drawn["start_s"]))
                if drawn.get("reached"):
                    reached += 1
        summary.append({
            "label": label,
            "episodes": total,
            "reached": reached,
            "fraction": reached / total if total else None,
            "mean_start_s": statistics.fmean(starts) if starts else None,
        })
    return summary


def run(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    try:
        bundle, task_sha256 = load_bundle(args.task)
        task_path = Path(args.task).expanduser().resolve()
    except BundleError as exc:
        return EXIT_USAGE, {"error": str(exc)}

    policies = [Path(item).expanduser().resolve() for item in args.policy]
    missing = [str(path) for path in policies if not path.is_file()]
    if missing:
        return EXIT_USAGE, {"error": f"no such policy file(s): {missing}"}

    search = [*sorted(policies[0].parent.glob("*-model.xml")),
              *sorted(task_path.parent.glob("*-model.xml"))]
    if args.model:
        search.insert(0, Path(args.model).expanduser().resolve())
    try:
        model = resolve_model(bundle, search)
    except BundleError as exc:
        return EXIT_USAGE, {"error": str(exc)}

    scales = list(args.scales) if args.scales else list(DEFAULT_SCALES)
    variants = build_variants(scales)
    seeds = list(range(args.seeds))

    episodes = [
        {"id": f"{path.name}|{name}|{seed}", "policy": str(path),
         "variant": name, "seed": seed}
        for path in policies for name in variants for seed in seeds
    ]

    payload: dict[str, Any] = {
        "task": str(task_path),
        "task_sha256": task_sha256,
        "model": str(model),
        "model_sha256": sha256_file(model),
        "policies": [str(path) for path in policies],
        "scales": scales,
        "variants": list(variants),
        "seeds": len(seeds),
        "max_steps": int(bundle["episode"]["max_steps"]),
        "declared_newtons": [
            {"label": item.get("label"),
             "low": item.get("newtons_low"), "high": item.get("newtons_high"),
             "sustained": item.get("sustained"),
             # Printed beside the azimuth split because it is what makes an
             # empty bin readable. `shove` is declared 210°–330° here, so two
             # of the four quadrants can never be drawn — those cells are the
             # task's design, not missing measurement, and a reader without
             # this line cannot tell the difference.
             "azimuth_low_deg": round(math.degrees(float(item.get("azimuth_low_rad") or 0.0)), 1),
             "azimuth_high_deg": round(math.degrees(float(item.get("azimuth_high_rad") or 0.0)), 1)}
            for item in bundle.get("disturbance") or []
        ],
        "episodes_planned": len(episodes),
    }

    progress = {"done": 0}

    def tick(row: dict[str, Any]) -> None:
        if row.get("row") == "episode":
            progress["done"] += 1
            if progress["done"] % 100 == 0 or progress["done"] == len(episodes):
                print(f"  … {progress['done']}/{len(episodes)} episodes",
                      file=sys.stderr)

    started = time.monotonic()
    try:
        rows = run_episode_job(
            {
                "model_xml": str(model), "task_path": str(task_path),
                "task_sha256": task_sha256, "jobs": args.jobs,
                "variants": variants,
                "verify": [str(path) for path in policies],
                "episodes": episodes,
            },
            on_row=tick,
        )
    except (TrainerVenvError, EpisodeJobFailed) as exc:
        return EXIT_INFRASTRUCTURE, {**payload, "error": str(exc)}
    payload["wall_seconds"] = round(time.monotonic() - started, 2)

    verified = {row["policy"]: row for row in rows if row.get("row") == "verify"}
    payload["verify_refused"] = [
        {"policy": Path(path).name, "error": row.get("error")}
        for path, row in verified.items() if not row.get("ok")
    ]

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("row") == "episode":
            grouped.setdefault((row["policy"], row["variant"]), []).append(row)

    sweeps: list[dict[str, Any]] = []
    for path in policies:
        for name in variants:
            group = grouped.get((str(path), name), [])
            if not group:
                continue
            summary = aggregate(group, bundle)
            summary.update({
                "policy": path.name,
                "policy_path": str(path),
                "variant": name,
                "scale": variants[name].get("disturbance_scale"),
                "azimuth": azimuth_split(group),
                "schedule": schedule_reach(group, bundle),
                "flags": flags(summary),
                "death_time_mean_s": statistics.fmean(
                    [row["end_time_s"] for row in group if not row["truncated"]]
                ) if any(not row["truncated"] for row in group) else None,
            })
            sweeps.append(summary)
    payload["sweep"] = sweeps

    # The check this driver exists to make about itself.
    payload["flatness"] = flatness(sweeps, policies)

    ok = not payload["verify_refused"] and bool(sweeps)
    return (EXIT_OK if ok else EXIT_INFRASTRUCTURE), payload


def flatness(sweeps: list[dict[str, Any]], policies: list[Path]) -> dict[str, Any]:
    """Did this sweep measure anything? Per policy, stated plainly.

    Flat at the bottom and flat at the top are different diagnoses and both
    are non-results: the first says the band starts above what the policy can
    do, the second says it ends below. Either way the *next* action is to
    re-size the sweep, not to read the numbers as a capability curve.
    """

    verdicts = []
    for path in policies:
        rows = [item for item in sweeps if item["policy"] == path.name
                and item["scale"] is not None]
        if len(rows) < 2:
            continue
        values = [row["survival"] for row in rows]
        low, high = min(values), max(values)
        if high == 0.0:
            verdict, detail = "flat-zero", (
                "0 survivals at every scale, including 0.0. The sweep "
                "measured nothing about the band: this policy cannot hold "
                "the reset variation, so no shove magnitude was ever the "
                "thing that killed it."
            )
        elif low == 1.0:
            verdict, detail = "flat-full", (
                "survives everything at every scale, including 1.00. The "
                "sweep measured nothing: the declared band is entirely "
                "inside this policy's capability and the ceiling is above "
                "the sweep."
            )
        elif high - low < 0.15:
            verdict, detail = "nearly-flat", (
                f"survival moves only {high - low:.0%} across the whole "
                "sweep. Weak evidence about the band; widen the scales "
                "before reading a curve into it."
            )
        else:
            verdict, detail = "informative", (
                f"survival runs from {low:.0%} to {high:.0%} across the "
                "sweep, so the band is doing work."
            )
        verdicts.append({
            "policy": path.name, "verdict": verdict, "detail": detail,
            "min_survival": low, "max_survival": high,
        })
    return {"per_policy": verdicts,
            "any_informative": any(v["verdict"] == "informative" for v in verdicts)}


# ---------------------------------------------------------------------------


def report(payload: dict[str, Any]) -> None:
    if payload.get("error"):
        print(f"ERROR  {payload['error']}")
        return

    print(f"task      {payload['task']}")
    print(f"          sha256 {payload['task_sha256']}")
    print(f"model     {payload['model']}  (digest matches the bundle's own)")
    print(f"seeds     {payload['seeds']} per row   budget "
          f"{payload['max_steps']} steps   wall "
          f"{payload.get('wall_seconds', 0):.1f} s")
    print("declared disturbance:")
    for item in payload["declared_newtons"]:
        print(f"  {str(item['label']):<10} {item['low']}–{item['high']} N   "
              f"azimuth {item.get('azimuth_low_deg')}°–"
              f"{item.get('azimuth_high_deg')}°"
              + ("  (sustained)" if item.get("sustained") else ""))

    if payload.get("verify_refused"):
        print()
        print("verify_policy REFUSED — reported, not dropped:")
        for item in payload["verify_refused"]:
            print(f"  {item['policy']:<26} {item['error']}")

    for policy in payload["policies"]:
        name = Path(policy).name
        rows = [row for row in payload["sweep"] if row["policy"] == name]
        if not rows:
            continue
        print()
        print("=" * 78)
        print(f"{name}")
        print()
        print(f"  {'row':<22} {'survival':>10} {'steps µ':>8} {'med':>5} "
              f"{'death µ s':>10}  termination mix")
        for row in rows:
            death = row["death_time_mean_s"]
            death_text = "—" if death is None else f"{death:.2f}"
            mix = " ".join(f"{k}:{v}" for k, v in row["termination_mix"].items())
            print(
                f"  {row['variant']:<22} "
                f"{row['survived']:>4}/{row['episodes']:<5} "
                f"{row['steps_mean']:>8.1f} {row['steps_median']:>5.0f} "
                f"{death_text:>10}  {mix}"
            )

        print()
        first = next(
            (item for item in payload["declared_newtons"] if not item.get("sustained")),
            None,
        )
        print("  survival by the azimuth of the first shove drawn (n per bin "
              "printed, because the bins are sparse)")
        if first is not None:
            print(f"  the task draws `{first['label']}` over "
                  f"{first['azimuth_low_deg']}°–{first['azimuth_high_deg']}° only, "
                  "so a bin outside that range is empty BY DESIGN, not unmeasured")
        header = "  " + f"{'row':<22}" + "".join(
            f"{label:>18}" for label, _, _ in AZIMUTH_BINS)
        print(header)
        for row in rows:
            cells = []
            for record in row["azimuth"]:
                if record["n"] == 0:
                    cells.append(f"{'—':>18}")
                else:
                    cells.append(
                        f"{record['survived']:>7}/{record['n']:<3}"
                        f"{record['survival'] * 100:>6.0f}%"
                    )
            print(f"  {row['variant']:<22}" + "".join(cells))

        print()
        print("  how far into its own disturbance schedule each row got")
        print(f"  {'row':<22} " + " ".join(
            f"{str(item['label']):>18}" for item in payload["declared_newtons"]))
        for row in rows:
            cells = []
            for record in row["schedule"]:
                if not record["episodes"]:
                    cells.append(f"{'—':>18}")
                else:
                    cells.append(
                        f"{record['reached']:>6}/{record['episodes']:<4}"
                        f"{record['fraction'] * 100:>6.0f}%"
                    )
            print(f"  {row['variant']:<22} " + " ".join(cells))
        print("  (a window that is never reached measured nothing, however "
              "carefully it was declared)")

        flagged = [row for row in rows if row.get("flags")]
        if flagged:
            print()
            print("  torque flags")
            for row in flagged:
                for note in row["flags"]:
                    print(f"    {row['variant']:<22} {note}")

    print()
    print("=" * 78)
    print("DID THIS SWEEP MEASURE ANYTHING?")
    for verdict in payload["flatness"]["per_policy"]:
        mark = "ok  " if verdict["verdict"] == "informative" else "NO  "
        print(f"  {mark}{verdict['policy']:<26} {verdict['verdict']}")
        print(f"      {verdict['detail']}")
    if not payload["flatness"]["any_informative"]:
        print()
        print("  No policy produced a non-flat curve. A flat sweep is not a")
        print("  finding about the mechanism — it is a sweep that was pointed")
        print("  at the wrong range. Re-size the scales before concluding")
        print("  anything about capability.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harness capability",
        description="Sweep the declared disturbance band and say what the "
                    "task was asking.",
    )
    parser.add_argument(
        "--policy", action="append", required=True, metavar="FILE",
        help="A .cxpolicy to sweep. Repeatable.",
    )
    parser.add_argument("--task", required=True, help="The policy's OWN task bundle.")
    parser.add_argument("--model", help="An explicit MJCF; otherwise found by digest.")
    parser.add_argument(
        "--scales", type=float, nargs="+", default=None,
        help=f"Scale factors on the declared newtons. Default "
             f"{list(DEFAULT_SCALES)}.",
    )
    parser.add_argument("--seeds", type=int, default=12, help="Seeds per row.")
    parser.add_argument("--jobs", type=int, default=0, help="Worker processes.")
    parser.add_argument("--json", action="store_true", help="Emit the envelope.")
    parser.add_argument("--table", metavar="PATH", help="Write the sweep as a JSON table.")
    return parser


def write_table(payload: dict[str, Any], path: Path) -> None:
    columns = [
        "policy", "row", "scale", "episodes", "survived", "survival",
        "steps_mean", "steps_median", "death_time_mean_s",
        "term_survived", "term_tipped", "term_collapsed",
        "peak_frac_of_limit", "mean_frac_of_limit", "saturation_worst",
    ]
    for label, _, _ in AZIMUTH_BINS:
        columns += [f"n[{label}]", f"survival[{label}]"]
    rows: list[list[Any]] = []
    for row in payload.get("sweep") or []:
        mix = row["termination_mix"]
        record: list[Any] = [
            row["policy"], row["variant"], row["scale"], row["episodes"],
            row["survived"], round(row["survival"], 4),
            round(row["steps_mean"], 2), row["steps_median"],
            None if row["death_time_mean_s"] is None else round(row["death_time_mean_s"], 4),
            mix.get("survived", 0), mix.get("tipped", 0), mix.get("collapsed", 0),
            round(row["peak_frac_of_limit"], 4),
            round(row["mean_frac_of_limit"], 4),
            round(row["saturation_worst"], 5),
        ]
        for bin_row in row["azimuth"]:
            record += [
                bin_row["n"],
                None if bin_row["survival"] is None else round(bin_row["survival"], 4),
            ]
        rows.append(record)
    path.write_text(json.dumps({
        "schema": "cdxrl-capability-table-v1",
        "task_sha256": payload["task_sha256"],
        "seeds": payload["seeds"],
        "columns": columns, "rows": rows,
    }, indent=1) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    load_env_file()
    args = build_parser().parse_args(argv)
    code, payload = run(args)
    if args.table and payload.get("sweep"):
        write_table(payload, Path(args.table).expanduser())
        print(f"table written to {args.table}", file=sys.stderr)
    if args.json:
        json.dump(envelope("capability", code == EXIT_OK, payload),
                  sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        report(payload)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
