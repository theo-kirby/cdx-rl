#!/usr/bin/env python
"""Hazard 15, measured against a TRAINED POLICY under a position action space.

    /home/theo/cadex-train-venv/bin/python \\
        mechanisms/mg-legs/drivers/hazard15.py \\
        --policy <p.cxpolicy> --task tasks/stand-b8/stand-task.json \\
        [--seeds 12] [--rating-nmm 86]

*The bracing is the resting posture.* Experiment 001 found the policy holding
a motor at 71 % of its 86 N·mm rating with nothing pushing, and called that a
machine that cannot be built; 002 replicated it 3 of 3 at 63–87 %.

## Why this file exists instead of `harness capability`

**The harness's torque columns are wrong under a position action space, and
they are wrong silently.** ``_episodes._torque_columns`` derives peak, mean
and saturation from ``step["action"]`` — the clamped list written to
``data.ctrl`` — which is correct when the action IS torque and meaningless
when it is a joint angle. On `stand-b8` it reports

    limit_nmm       [30, 30, 45, 45, 30, 30, 25, 25, 20, 20]
    peak_torque_nmm [27.6, 29.6, 43.3, 44.3, …]

which are **degrees** against the joints' angle ranges, labelled N·mm and
scored against the wrong limit. A reader who divides 44.3 by an 86 N·mm
rating gets 51 % and has measured nothing. `method.md`'s note that the drop
test inverts under a position action space applies to this instrument too,
and nobody had noticed.

## …and why 003's "hazard 15 dissolves" does not settle it either

003 reports **27.0 N·mm peak, 31 % of rating**, against 002's 63–87 %. But
that figure is `feasibility.py` check 6: *the model's own servo, one episode,
zero action*, holding the nominal pose. It is a property of the mechanism and
the PD gains, and no trained network is involved. 002's 63–87 % is a trained
policy's commanded torque. The two are not the same measurement, so the
comparison does not license "the action space dissolved hazard 15" for any
policy — it licenses it for the servo.

This file measures the thing the hazard is actually about: **what torque the
solver develops while a trained policy holds the machine up with nothing
pushing.** Under position control the policy never commands torque, so the
only honest source is ``data.actuator_force`` and the only honest limit is
``model.actuator_forcerange``, both read off the compiled model.

## What it measures, and the one caveat

``evaluate_episode``'s ``sample`` callback is invoked at the reset pose and
once after every control step, so the force is sampled at the **control**
rate — 50 Hz — and not at the solver's 500 Hz. A spike entirely inside one
control interval is invisible here. That is the same resolution every other
number in this experiment is quoted at, and it is stated rather than hidden.

The disturbance schedule is dropped with ``disturbance: False`` — not scaled
to zero, which still runs the schedule and applies a zero force. `nothing
pushing` means the list is gone.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MODULE_DIR = os.environ.get("CADEX_ENGINE_DEV_TREE", "/home/theo/cadex")
sys.path.insert(0, str(Path(MODULE_DIR) / "src" / "Mod" / "cadex"))
sys.path.insert(0, str(REPO / "harness"))

import CadexDynamics as cd  # noqa: E402
import numpy as np  # noqa: E402

from _episodes import apply_variant  # noqa: E402

#: Seconds to let the reset drop be absorbed before the posture is
#: measured. 003 measured the machine settling for roughly a second; the
#: whole-episode peak is reported beside it so the gap is visible rather
#: than chosen.
SETTLE_SECONDS = 1.0


def measure(policy_path: Path, task: dict, model_xml: bytes, seeds: list[int],
            rating_nmm: float) -> dict:
    """Peak |actuator_force| per motor, over undisturbed episodes."""

    container = cd.decode_policy(policy_path.read_bytes(),
                                 context=policy_path.name)
    header, weights = container["header"], container["weights"]
    # Nothing pushing: the schedule is removed, not zeroed.
    quiet = apply_variant(task, {"disturbance": False})

    peaks: list[np.ndarray] = []
    settled_peaks: list[np.ndarray] = []
    settled_frames: list[np.ndarray] = []
    survived = 0
    limits = None
    control_dt = float(task["episode"]["control_interval_s"])
    settle_steps = int(round(SETTLE_SECONDS / control_dt))

    for seed in seeds:
        model = cd.load_model(model_xml)
        peak = {"value": None, "settled": None}

        def sample(step, data, _final, _action, _peak=peak):
            force = np.abs(np.asarray(data.actuator_force, dtype=float))
            _peak["value"] = (force if _peak["value"] is None
                              else np.maximum(_peak["value"], force))
            # **The window matters more than anything else here.** The reset
            # variation lifts the machine and drops it, and absorbing a 42 mm
            # drop saturates every motor — a peak over the whole episode
            # measures the landing, not the posture. `script.py` documents
            # this exact instrument error against foot lift, found the same
            # day as ADR-107's rotated frame.
            if step is not None and int(step) >= settle_steps:
                _peak["settled"] = (force if _peak["settled"] is None
                                    else np.maximum(_peak["settled"], force))
                # **A peak cannot tell a spike from a brace**, and hazard 15
                # is about the brace: 001's finding was a motor HELD at 71 %
                # for six seconds, not one that touched it once. Every
                # settled frame is kept so a duty cycle can be computed.
                _peak.setdefault("frames", []).append(force)
            return None

        episode = cd.evaluate_episode(
            model, quiet,
            actions=lambda _s, obs: cd.policy_forward(header, weights, obs),
            sample=sample, seed=seed,
        )
        if limits is None:
            # forcerange is in N·m in the compiled model; the project talks
            # N·mm everywhere else.
            limits = np.asarray(model.actuator_forcerange[:, 1],
                                dtype=float) * 1000.0
        if peak["value"] is not None:
            peaks.append(peak["value"] * 1000.0)   # N·m -> N·mm
        if peak["settled"] is not None:
            settled_peaks.append(peak["settled"] * 1000.0)
        if peak.get("frames"):
            settled_frames.append(np.vstack(peak["frames"]) * 1000.0)
        if episode.get("truncated"):
            survived += 1

    stack = np.vstack(peaks)
    worst_per_motor = stack.max(axis=0)
    settled = np.vstack(settled_peaks).max(axis=0)
    allframes = np.vstack(settled_frames)          # (frames, motors), N·mm
    lim = np.asarray(limits, dtype=float)
    # Duty cycle: the fraction of settled frames in which the worst motor is
    # above 90 % of its own forcerange. This is the number hazard 15 is
    # actually about.
    above90 = (allframes >= 0.9 * lim).mean(axis=0)
    mean_force = allframes.mean(axis=0)
    # The rating is a judgment (86 N·mm), the forcerange is what the model
    # was actually given. Report against both; they need not agree.
    return {
        "policy": policy_path.name,
        "episodes": len(seeds),
        "survived": survived,
        "peak_nmm_per_motor": [round(v, 3) for v in worst_per_motor],
        "worst_motor_nmm": round(float(worst_per_motor.max()), 3),
        "worst_motor_index": int(worst_per_motor.argmax()),
        "frac_of_rating": round(float(worst_per_motor.max()) / rating_nmm, 4),
        "mean_across_motors_nmm": round(float(worst_per_motor.mean()), 3),
        "median_episode_peak_nmm": round(
            statistics.median(float(p.max()) for p in peaks), 3),
        "forcerange_nmm": [round(v, 3) for v in limits],
        "frac_of_forcerange": round(
            float((worst_per_motor / limits).max()), 4),
        "rating_nmm": rating_nmm,
        "settle_seconds": SETTLE_SECONDS,
        "settled_peak_nmm_per_motor": [round(v, 3) for v in settled],
        "settled_worst_nmm": round(float(settled.max()), 3),
        "settled_frac_of_rating": round(float(settled.max()) / rating_nmm, 4),
        "settled_mean_across_motors_nmm": round(float(settled.mean()), 3),
        "settled_frames": int(allframes.shape[0]),
        "settled_mean_nmm_per_motor": [round(v, 3) for v in mean_force],
        "settled_mean_frac_of_rating": round(
            float(mean_force.max()) / rating_nmm, 4),
        "settled_frac_above_90pct_per_motor": [round(v, 5) for v in above90],
        "settled_duty_worst": round(float(above90.max()), 5),
    }


#: ``<label>.NNNNNN.cxpolicy``. ``<label>.best.cxpolicy`` and the final
#: ``<label>.cxpolicy`` carry no iteration and are not part of a series.
_PERIODIC = re.compile(r"\.(\d{6})\.cxpolicy$")


def series_checkpoints(run_dir: Path, stride: int) -> list[tuple[int, Path]]:
    """Periodic checkpoints at ``stride``, plus the last one written.

    The last is always included whatever its iteration: it is the newest
    evidence about where the run is heading, and dropping it because 1750 is
    not a multiple of 250 would silently truncate the trend at its most
    informative end.
    """
    found: list[tuple[int, Path]] = []
    for p in sorted(run_dir.glob("*.cxpolicy")):
        m = _PERIODIC.search(p.name)
        if m:
            found.append((int(m.group(1)), p))
    if not found:
        raise SystemExit(f"no periodic checkpoints under {run_dir}")
    found.sort()
    keep = {it: p for it, p in found if stride <= 0 or it % stride == 0}
    keep[found[-1][0]] = found[-1][1]
    return sorted(keep.items())


def trend(iters: list[int], duty: list[float]) -> dict:
    """Least-squares slope of duty against iteration, and the extrapolation.

    Printed unconditionally. The gate's decision rule was written down before
    any of this was looked at (ADR-097): flat or falling means bracing is not
    a function of training time; rising past 25 % duty by extrapolation to
    3600 means a longer run undoes experiment 004's result.
    """
    if len(iters) < 2:
        return {"slope_per_1000": 0.0, "extrapolated_3600": duty[-1] if duty
                else 0.0, "n": len(iters)}
    x = np.asarray(iters, dtype=float)
    y = np.asarray(duty, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return {
        "slope_per_1000": round(float(slope) * 1000.0, 6),
        "intercept": round(float(intercept), 6),
        "extrapolated_3600": round(float(slope * 3600.0 + intercept), 6),
        "first": round(float(y[0]), 6),
        "last": round(float(y[-1]), 6),
        "n": len(iters),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", action="append", type=Path)
    ap.add_argument("--series", type=Path,
                    help="run directory: score its periodic checkpoints at "
                         "--stride and print the duty-cycle trend")
    ap.add_argument("--stride", type=int, default=250,
                    help="series mode: iteration spacing (default 250); "
                         "the last checkpoint is always included")
    ap.add_argument("--task", required=True, type=Path)
    ap.add_argument("--model", type=Path)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--rating-nmm", type=float, default=86.0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not args.policy and not args.series:
        ap.error("one of --policy or --series is required")

    task = json.loads(args.task.read_text())
    model_path = args.model or (args.task.parent / "model-model.xml")
    model_xml = model_path.read_bytes()
    seeds = list(range(args.seeds))

    selected: list[tuple[int | None, Path]] = [(None, p)
                                               for p in (args.policy or [])]
    if args.series:
        selected += series_checkpoints(args.series, args.stride)

    rows = []
    for iteration, p in selected:
        row = measure(p, task, model_xml, seeds, args.rating_nmm)
        if iteration is not None:
            row["iteration"] = iteration
        rows.append(row)

    series_rows = [r for r in rows if "iteration" in r]
    tr = trend([r["iteration"] for r in series_rows],
               [r["settled_duty_worst"] for r in series_rows]) \
        if series_rows else None

    if args.json:
        out = {"driver": "hazard15", "rating_nmm": args.rating_nmm,
               "seeds": seeds, "rows": rows}
        if tr:
            out["duty_trend"] = tr
        print(json.dumps(out, indent=1))
        return 0

    print(f"hazard 15 — peak |actuator_force| with NOTHING PUSHING, "
          f"{len(seeds)} seeds, sampled at the control rate")
    print(f"rating {args.rating_nmm:.0f} N·mm (a judgment, not a datasheet)\n")
    print(f"  {'policy':<30} {'surv':>7} | {'whole episode':>21} | "
          f"{'after ' + str(SETTLE_SECONDS) + ' s':>21}")
    print(f"  {'':<30} {'':>7} | {'worst':>10} {'% rating':>10} | "
          f"{'worst':>10} {'% rating':>10} | {'mean%':>7} {'duty%':>7}")
    for r in rows:
        print(f"  {r['policy']:<30} {r['survived']:>3}/{r['episodes']:<3} | "
              f"{r['worst_motor_nmm']:>10.1f} {r['frac_of_rating']*100:>10.1f} | "
              f"{r['settled_worst_nmm']:>10.1f} "
              f"{r['settled_frac_of_rating']*100:>10.1f} | "
              f"{r['settled_mean_frac_of_rating']*100:>7.1f} "
              f"{r['settled_duty_worst']*100:>7.1f}")
    print("\n  The whole-episode column is the RESET DROP, not the posture. "
          "It saturates\n  every motor and means nothing about hazard 15; "
          "it is printed so that the\n  settled column cannot be mistaken "
          "for a cherry-picked window.")
    print("\n  For reference: 001 measured 71 % of rating, 002 replicated "
          "63–87 % in 3 of 3\n  seeds — both under a TORQUE action space, "
          "where the policy's own command\n  was the torque.")

    if tr:
        print(f"\n  duty trend over {tr['n']} checkpoints "
              f"({series_rows[0]['iteration']} → "
              f"{series_rows[-1]['iteration']}): "
              f"{tr['first']*100:.1f} % → {tr['last']*100:.1f} %, "
              f"slope {tr['slope_per_1000']*100:+.2f} pp / 1000 iterations")
        print(f"  extrapolated to iteration 3600: "
              f"{tr['extrapolated_3600']*100:.1f} % duty")
        print("\n  The rule was written down before this was run: flat or "
              "falling means the\n  bracing is not a function of training "
              "time; rising past 25 % duty at 3600\n  means a longer run "
              "undoes experiment 004's result. A linear fit is the\n  "
              "weakest possible model of the trend — read the per-checkpoint "
              "column too.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
