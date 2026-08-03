#!/usr/bin/env python
"""Read a ``.cxpolicy`` header — and diff two runs' reward curves.

```
uv run python tools/cxpolicy.py --policy PATH [--curve] [--json]
uv run python tools/cxpolicy.py --compare A.cxpolicy B.cxpolicy [--limit N]
```

**Why this exists.** ``cadex-wishlist.md`` #6 used to claim the reward curve
lived only as prose in ``train.log`` and asked Cadex for a
``progress.jsonl``. That was wrong, and this file is the correction made
executable: every ``.cxpolicy`` already carries the **complete**
per-iteration series in its header, as structured JSON.

The container, verified on this box:

```
b"CXPOLICY1\\n"        10 bytes of magic
<uint64 little-endian>  8 bytes: the header's length
<header>                that many bytes of UTF-8 JSON
<weights>               the rest
```

For ``stand-task-20260802-200109``'s final policy that is a 417 756-byte
header on 33 576 bytes of weights: **the header is what grows, not the
network.** It carries ``training.reward_curve`` (2 500 entries of
``iteration``, ``reward_per_step``, ``episode_steps``, ``loss``,
``action_std``), plus ``hyperparameters``, ``seed``, ``device``,
``trainer_sha256``, ``wall_time_s``, ``versions`` and ``randomisation``.

Nothing here needs the trainer venv — it is a length prefix and
``json.loads``, so it runs in cdx-rl's own small environment. Playing a
policy is a different matter and still goes through ``CadexDynamics``.

``--compare`` is experiment 002's criterion 6: seed 0 is a replication
control, and this is what checks it against 200109's recorded curve.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any

MAGIC = b"CXPOLICY1\n"

#: The fields of one ``reward_curve`` entry that a comparison reads.
SERIES_FIELDS = ("reward_per_step", "episode_steps", "loss", "action_std")


def read_header(path: Path) -> dict[str, Any]:
    """The JSON header of a ``.cxpolicy``. Raises ``ValueError`` if it is not one."""

    raw = path.read_bytes()
    if not raw.startswith(MAGIC):
        raise ValueError(
            f"{path} does not start with {MAGIC!r} — this is not a "
            f"cadex-policy-v1 container (first bytes: {raw[:12]!r})"
        )
    (length,) = struct.unpack("<Q", raw[len(MAGIC):len(MAGIC) + 8])
    start = len(MAGIC) + 8
    header = json.loads(raw[start:start + length].decode("utf-8"))
    header["_bytes"] = {
        "file": len(raw),
        "header": length,
        "weights": len(raw) - start - length,
    }
    return header


def curve(header: dict[str, Any]) -> list[dict[str, Any]]:
    return list((header.get("training") or {}).get("reward_curve") or [])


def describe(path: Path, header: dict[str, Any], *, show_curve: bool) -> None:
    training = header.get("training") or {}
    sizes = header["_bytes"]
    series = curve(header)

    print(f"file        {path}")
    print(f"bytes       {sizes['file']} = {sizes['header']} header "
          f"+ {sizes['weights']} weights + 18 prefix")
    print(f"label       {header.get('label', '—')}")
    print(f"seed        {training.get('seed', '—')}   "
          f"device {training.get('device', '—')}   "
          f"iterations {training.get('iterations', '—')}")
    wall = training.get("wall_time_s")
    if isinstance(wall, (int, float)):
        print(f"wall        {wall:.0f} s ({wall / 3600:.2f} h)")
    print(f"trainer     {str(training.get('trainer_sha256') or '—')[:16]}…")
    print(f"model       {str((header.get('model') or {}).get('sha256') or '—')[:16]}…")
    print(f"task        {str((header.get('task') or {}).get('sha256') or '—')[:16]}…")

    print()
    print("hyperparameters")
    for key, value in sorted((training.get("hyperparameters") or {}).items()):
        print(f"  {key:<18} {value}")

    print()
    print(f"reward_curve  {len(series)} entries"
          + (f", iterations {series[0]['iteration']}…{series[-1]['iteration']}"
             if series else ""))
    if series and show_curve:
        print(f"  {'iteration':>9} {'reward/step':>12} {'episode':>9} "
              f"{'sigma':>8} {'loss':>10}")
        step = max(1, len(series) // 20)
        for point in series[::step] + ([series[-1]] if len(series) > 1 else []):
            print(f"  {point['iteration']:>9} {point['reward_per_step']:>12.6f} "
                  f"{point.get('episode_steps', float('nan')):>9.1f} "
                  f"{point.get('action_std', float('nan')):>8.4f} "
                  f"{point.get('loss', float('nan')):>10.4g}")


def compare(left: Path, right: Path, limit: int | None) -> dict[str, Any]:
    """Two curves, iteration by iteration, over the span they share.

    **What a difference here means.** Same seed, same hyperparameters and the
    same ``trainer_sha256`` do *not* guarantee the same trajectory: MJX and
    XLA reductions on a GPU are not bitwise reproducible run to run, and PPO
    is a feedback loop, so a difference in the last decimal at iteration 0
    is amplified by every iteration after it. This prints where the two
    curves separate rather than asserting they should not — a divergence is a
    **finding about run-to-run comparability on this card**, not a failure.
    """

    a, b = read_header(left), read_header(right)
    series_a, series_b = curve(a), curve(b)
    by_iteration = {point["iteration"]: point for point in series_b}
    shared = [point for point in series_a if point["iteration"] in by_iteration]
    if limit is not None:
        shared = [point for point in shared if point["iteration"] < limit]

    rows: list[dict[str, Any]] = []
    for point in shared:
        other = by_iteration[point["iteration"]]
        row: dict[str, Any] = {"iteration": point["iteration"]}
        for field in SERIES_FIELDS:
            first, second = point.get(field), other.get(field)
            if isinstance(first, (int, float)) and isinstance(second, (int, float)):
                row[field] = (float(first), float(second), float(second) - float(first))
        rows.append(row)

    print(f"left    {left}")
    print(f"        seed {(a.get('training') or {}).get('seed')}  "
          f"{len(series_a)} entries")
    print(f"right   {right}")
    print(f"        seed {(b.get('training') or {}).get('seed')}  "
          f"{len(series_b)} entries")
    print(f"shared  {len(rows)} iterations"
          + (f" (capped at {limit})" if limit is not None else ""))

    left_hp = (a.get("training") or {}).get("hyperparameters") or {}
    right_hp = (b.get("training") or {}).get("hyperparameters") or {}
    ignored = {"progress", "iterations", "checkpoint_every"}
    differing = {
        key for key in set(left_hp) | set(right_hp)
        if key not in ignored and left_hp.get(key) != right_hp.get(key)
    }
    print()
    if differing:
        print("HYPERPARAMETERS DIFFER — these curves are not a replication:")
        for key in sorted(differing):
            print(f"  {key:<18} {left_hp.get(key)!r}  vs  {right_hp.get(key)!r}")
    else:
        print("hyperparameters identical (ignoring iterations, checkpoint_every,")
        print("progress path — none of which enter the update; the learning rate")
        print("is a constant float and no schedule reads the total).")
    trainer_a = (a.get("training") or {}).get("trainer_sha256")
    trainer_b = (b.get("training") or {}).get("trainer_sha256")
    print(f"trainer     {'identical' if trainer_a == trainer_b else 'DIFFERENT'}"
          f"  {str(trainer_a)[:12]}… / {str(trainer_b)[:12]}…")

    if not rows:
        print("\nNo shared iterations — nothing to compare.")
        return {"rows": 0}

    print()
    print(f"  {'iteration':>9} {'reward L':>11} {'reward R':>11} {'Δ':>11}"
          f" {'episode L':>10} {'episode R':>10}")
    step = max(1, len(rows) // 15)
    for row in rows[::step] + [rows[-1]]:
        reward = row.get("reward_per_step")
        episode = row.get("episode_steps")
        if reward is None:
            continue
        print(f"  {row['iteration']:>9} {reward[0]:>11.6f} {reward[1]:>11.6f} "
              f"{reward[2]:>+11.6f}"
              + (f" {episode[0]:>10.1f} {episode[1]:>10.1f}" if episode else ""))

    # Where they separate, and by how much, on the metric the trainer reports.
    deltas = [abs(row["reward_per_step"][2]) for row in rows if "reward_per_step" in row]
    firsts = [row for row in rows if "reward_per_step" in row]
    identical = sum(1 for value in deltas if value == 0.0)
    print()
    print(f"reward_per_step: {identical}/{len(deltas)} iterations bitwise identical")
    print(f"  max |Δ| {max(deltas):.6f}   mean |Δ| {sum(deltas) / len(deltas):.6f}")
    if firsts and deltas[0] != 0.0:
        print(f"  they differ from the FIRST shared iteration "
              f"({firsts[0]['iteration']}): "
              f"{firsts[0]['reward_per_step'][0]:.6f} vs "
              f"{firsts[0]['reward_per_step'][1]:.6f}")
        print("  — so this is not drift from a shared start; the very first")
        print("    update already differs. Same seed, same code, same card.")

    # Correlation over the shared span: the shape question, not the identity one.
    left_values = [row["reward_per_step"][0] for row in rows if "reward_per_step" in row]
    right_values = [row["reward_per_step"][1] for row in rows if "reward_per_step" in row]
    n = len(left_values)
    if n > 2:
        mean_l, mean_r = sum(left_values) / n, sum(right_values) / n
        cov = sum((x - mean_l) * (y - mean_r) for x, y in zip(left_values, right_values))
        var_l = sum((x - mean_l) ** 2 for x in left_values)
        var_r = sum((y - mean_r) ** 2 for y in right_values)
        if var_l > 0 and var_r > 0:
            r = cov / math.sqrt(var_l * var_r)
            print(f"  Pearson r over the shared span: {r:+.4f}  "
                  f"(shape, not identity)")
    return {"rows": len(rows), "max_abs_delta": max(deltas), "identical": identical}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tools/cxpolicy.py",
        description="Read a .cxpolicy header; diff two runs' reward curves.",
    )
    parser.add_argument("--policy", help="Describe this .cxpolicy.")
    parser.add_argument("--curve", action="store_true", help="Print a sample of the curve.")
    parser.add_argument("--compare", nargs=2, metavar=("LEFT", "RIGHT"))
    parser.add_argument("--limit", type=int, default=None,
                        help="Compare only iterations below this.")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.compare:
        left, right = (Path(item).expanduser() for item in args.compare)
        result = compare(left, right, args.limit)
        if args.json:
            json.dump(result, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
        return 0
    if args.policy:
        path = Path(args.policy).expanduser()
        header = read_header(path)
        if args.json:
            json.dump(header, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
        else:
            describe(path, header, show_curve=args.curve)
        return 0
    build_parser().print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
