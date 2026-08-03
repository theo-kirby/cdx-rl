"""``supervise`` — the training supervisor, and the post-mortem reader.

The piece ``MUJOCO.md`` §7 assumes a human provides by watching.

```
supervise --run DIR [--report-only] [--watch] [--patience N]
          [--min-iterations M] [--require-device gpu] [--sigma-floor X]
          [--json]
```

Two modes, one report. ``--report-only`` reads a finished run; ``--watch``
attaches to a live one, polls, acts, and then produces *the same report*.

### The bill it exists to prevent

``/home/theo/cadex-jobs/stand-task-20260802-200109`` — best at iteration
**598** (0.337 reward/step), ran to **2499** (0.146), 14 050 s ≈ 3.9
GPU-hours. About three quarters of the run was spent below its own peak, and
``progress.json`` said so every single iteration, to nobody.

### The line it must not cross

**Stopping the burn is not choosing the checkpoint.**

ADR-099 measured the trainer's reward and real survival as *anti-correlated
across a whole run*: 12/12 survival where the trainer reported its worst
numbers, 0/12 exactly where it reported its best. The checkpoint labelled
``best`` fell in 43 steps of 600, from every seed and every direction, before
the first shove window opened.

So this driver **stops** a run and **never installs** one. Its report ends by
naming the checkpoints worth playing and saying, out loud, that selection is
``compare``'s job. That sentence is in the output, not just in this
docstring, because the report is what gets read.

### Why episode length is printed beside reward everywhere

Hazard 19 is a reward climbing while episode length falls — a policy learning
to score rather than to survive. Its **mirror** is what 200109 actually did:
reward fell from 0.337 to 0.146 while mean episode length rose from 277.7 to
468.1. One of those is a policy going wrong and the other may be a policy
getting better at the thing we care about, and **you cannot tell which from
the reward column alone**. So the two are never printed apart.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

from harness import EXIT_INFRASTRUCTURE, EXIT_OK, EXIT_USAGE  # noqa: E402
from harness.provenance import envelope, load_env_file  # noqa: E402
from harness.runlog import (  # noqa: E402
    WITNESS_FLOOR, load_run, process_gone, read_progress,
)

#: How many rows the curve table shows, besides the named points. Ten is
#: enough to see a shape and few enough to read.
CURVE_ROWS = 10

#: Below this ``action_std``, exploration has collapsed and the run is only
#: burning power. **Read off the data, not invented**:
#: ``stand-task-20260802-200109`` decayed 0.4002 → 0.3375 over 2 500
#: iterations, so 0.02 — five per cent of that run's ``initial_std`` — sits an
#: order of magnitude below anything a healthy run has been observed to do.
#: A floor picked out of the air guesses at where "too small" begins; this one
#: is at least anchored to a run that worked.
SIGMA_FLOOR = 0.02


def quoted_points(run: dict[str, Any]) -> list[dict[str, Any]]:
    """An evenly-spaced sample of the curve, plus the three named points.

    The named ones are not decoration: the reward peak, the episode-length
    peak and the final iteration are the three places this experiment's
    argument lives, and a grid that happened to miss them would produce a
    table that quietly did not say the thing.
    """

    series = run["series"]
    if not series:
        return []
    indices = {0, len(series) - 1}
    step = max(1, len(series) // CURVE_ROWS)
    indices.update(range(0, len(series), step))
    by_iteration = {point["iteration"]: index for index, point in enumerate(series)}
    for named in ("best_reward", "best_episode"):
        point = run.get(named)
        if point is not None and point["iteration"] in by_iteration:
            indices.add(by_iteration[point["iteration"]])

    marks: dict[int, list[str]] = {}
    for label, named in (("best reward", "best_reward"), ("longest episode", "best_episode")):
        point = run.get(named)
        if point is not None:
            marks.setdefault(point["iteration"], []).append(label)
    marks.setdefault(series[-1]["iteration"], []).append("final")

    rows = []
    for index in sorted(indices):
        point = dict(series[index])
        point["mark"] = ", ".join(marks.get(point["iteration"], []))
        rows.append(point)
    return rows


def trend(run: dict[str, Any]) -> dict[str, Any]:
    """Reward against episode length, from the reward peak to the end.

    Three outcomes, and the middle one is the one nothing else would catch:

    ``hazard-19``
        reward up, episode length down. A policy learning to score.
    ``survival-diverges``
        reward down, episode length **up**. What 200109 did. Not a hazard
        with a number yet — it is the observation that motivated experiment
        001 — and it must not be reported as a regression, because it may be
        the opposite of one.
    ``agreeing``
        both moving the same way. The reward is at least tracking something.
    """

    best = run.get("best_reward")
    final = run.get("final")
    if not best or not final or best["iteration"] >= final["iteration"]:
        return {"verdict": "too-short", "detail": "no post-peak span to read"}

    reward_delta = final["reward_per_step"] - best["reward_per_step"]
    episode_from = best.get("episode_steps")
    episode_to = final.get("episode_steps")
    if episode_from is None or episode_to is None:
        return {
            "verdict": "unknown",
            "detail": "this trainer logged no episode length; the reward "
                      "column cannot be read on its own",
            "reward_delta": reward_delta,
        }
    episode_delta = episode_to - episode_from

    if reward_delta > 0 and episode_delta < 0:
        verdict = "hazard-19"
    elif reward_delta < 0 and episode_delta > 0:
        verdict = "survival-diverges"
    else:
        verdict = "agreeing"
    return {
        "verdict": verdict,
        "from_iteration": best["iteration"],
        "to_iteration": final["iteration"],
        "reward_delta": reward_delta,
        "episode_delta": episode_delta,
        "reward_from": best["reward_per_step"],
        "reward_to": final["reward_per_step"],
        "episode_from": episode_from,
        "episode_to": episode_to,
    }


def recommendations(run: dict[str, Any]) -> dict[str, Any]:
    """Which checkpoints are worth playing. **Not** which one to install.

    Three candidates, each for a stated reason, and the run's own bundle
    beside them because ``compare`` refuses to guess at one.
    """

    inventory = run["checkpoints"]
    by_iteration = {
        item["iteration"]: item for item in inventory if item.get("iteration") is not None
    }

    def nearest(target: int | None) -> dict[str, Any] | None:
        if target is None or not by_iteration:
            return None
        key = min(by_iteration, key=lambda value: abs(value - target))
        return by_iteration[key]

    candidates: list[dict[str, Any]] = []
    best_reward = run.get("best_reward")
    best_episode = run.get("best_episode")

    tagged_best = next((item for item in inventory if item.get("tag") == "best"), None)
    if tagged_best is not None:
        candidates.append({
            "reason": "best by the trainer's own reward"
                      + (f" (iteration {best_reward['iteration']})" if best_reward else ""),
            **tagged_best,
        })
    if best_episode is not None:
        near = nearest(best_episode["iteration"])
        if near is not None:
            candidates.append({
                "reason": f"nearest the episode-length peak "
                          f"({best_episode['episode_steps']:.1f} steps at "
                          f"iteration {best_episode['iteration']})",
                **near,
            })
    if run.get("final") is not None:
        final_name = str(run["progress"].get("out") or "")
        final_path = Path(run["run_dir"]) / final_name if final_name else None
        if final_path is not None and final_path.is_file():
            candidates.append({
                "reason": "the final network",
                "iteration": run["final"]["iteration"],
                "path": str(final_path), "name": final_path.name,
                "tag": "final", "exists": True,
            })

    seen: set[str] = set()
    unique = []
    for item in candidates:
        if item.get("path") in seen:
            continue
        seen.add(str(item.get("path")))
        unique.append(item)

    bundle = str(Path(run["run_dir"]) / run["bundle"]) if run["bundle"] else ""
    return {
        "play": unique,
        "compare_command": (
            f"uv run python -m harness compare --dir {run['run_dir']} "
            f"--task {bundle} --seeds 12"
        ) if bundle else "",
        # The line ADR-099 paid for. It is in the output, not only here.
        "note": (
            "These are candidates to PLAY, not a selection. supervise stops "
            "a burn; it never installs a checkpoint. Selection is compare's "
            "job, by measured survival (ADR-099: the trainer's reward and "
            "real survival were anti-correlated across a whole run)."
        ),
    }


def build_report(run: dict[str, Any]) -> dict[str, Any]:
    progress = run["progress"]
    log = run["log"]
    witness = log["witness"]
    low = [item for item in witness if item["factor"] < WITNESS_FLOOR]

    inventory = run["checkpoints"]
    missing = [item for item in inventory if not item.get("exists")]
    mismatched = [
        item for item in inventory
        if item.get("exists") and item.get("digest_matches") is False
    ]

    return {
        "run_dir": run["run_dir"],
        "label": run["label"],
        "device": str(progress.get("device") or ""),
        "state": str(progress.get("state") or ""),
        "error": str(progress.get("error") or ""),
        "iteration": progress.get("iteration"),
        "total": progress.get("total"),
        "wall_time_s": progress.get("wall_time_s"),
        "parameters": (log["terminal"] or {}).get("parameters"),
        "task_sha256": (log["terminal"] or {}).get("task_sha256", ""),
        "model_sha256": (log["terminal"] or {}).get("model_sha256", ""),
        "bundle": run["bundle"],
        "model": run["model"],
        "liveness": run["liveness"],
        "curve": quoted_points(run),
        "series_points": len(run["series"]),
        "best_reward": run["best_reward"],
        "best_episode": run["best_episode"],
        "final": run["final"],
        "max_steps": run["max_steps"],
        "over_budget_iterations": run["over_budget_iterations"],
        "trend": trend(run),
        "witness": witness,
        "witness_below_floor": low,
        "witness_floor": WITNESS_FLOOR,
        "checkpoints": inventory,
        "checkpoints_missing": [item["name"] for item in missing],
        "checkpoints_mismatched": [item["name"] for item in mismatched],
        "log_warnings": log["warnings"],
        "recommendations": recommendations(run),
    }


# ---------------------------------------------------------------------------
# Prose
# ---------------------------------------------------------------------------


def _fmt(value: Any, spec: str = "") -> str:
    if value is None:
        return "—"
    if spec:
        try:
            return format(value, spec)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def report(data: dict[str, Any]) -> None:
    live = data["liveness"]
    print(f"run       {data['run_dir']}")
    print(f"label     {data['label']}   device {data['device'] or '—'}   "
          f"state {data['state'] or '—'}"
          + (f"   error {data['error']}" if data["error"] else ""))
    wall = data["wall_time_s"]
    print(f"progress  iteration {_fmt(data['iteration'])} of {_fmt(data['total'])}"
          + (f"   wall {wall:.0f} s ({wall / 3600:.2f} h)" if wall else ""))
    print(f"artifacts bundle {data['bundle'] or '—'}   model {data['model'] or '—'}"
          f"   parameters {_fmt(data['parameters'])}")

    print()
    print("liveness")
    print(f"  pid            {_fmt(live['pid'])}  alive={_fmt(live['pid_alive'])}")
    print(f"  progress.json  {_fmt(live['progress_mtime'])}  "
          f"age {_fmt(live['progress_age_s'], '.0f')} s")
    print(f"  state          {live['state'] or '—'}  terminal={live['terminal']}")
    if live["stale"]:
        print("  STALE — this run claims to be training and nothing is running it.")
        print("          A stale progress.json is indistinguishable from a live")
        print("          one without this check; do not wait on it.")
    elif live["live"]:
        print("  LIVE — a process is writing here.")

    print()
    print("curve   (episode length is printed beside reward at every point,")
    print("         because reward alone cannot tell surviving from scoring)")
    print(f"  {'iteration':>9} {'reward/step':>12} {'episode':>9} {'sigma':>7} "
          f"{'loss':>10}  mark")
    for point in data["curve"]:
        print(
            f"  {point['iteration']:>9} {point['reward_per_step']:>12.4f} "
            f"{_fmt(point['episode_steps'], '>9.1f')} "
            f"{_fmt(point['sigma'], '>7.4f')} {point['loss']:>10.4g}"
            f"  {point['mark']}"
        )
    print(f"  ({data['series_points']} iterations parsed from train.log)")

    print()
    print(f"peaks   (episode budget {_fmt(data['max_steps'])} steps)")
    if data["over_budget_iterations"]:
        excluded = data["over_budget_iterations"]
        print(f"  {len(excluded)} point(s) logged an episode length above that "
              f"budget — iteration(s) "
              f"{', '.join(str(value) for value in excluded[:8])}"
              + (" …" if len(excluded) > 8 else "") + ".")
        print("  An episode cannot outrun the horizon it is truncated at, so")
        print("  those are not mean episode lengths and are excluded from the")
        print("  peak search. Reported rather than dropped quietly.")
    for label, key in (("reward", "best_reward"), ("episode length", "best_episode")):
        point = data.get(key)
        if point is None:
            print(f"  best {label:<15} —  (this trainer logged none)")
            continue
        print(f"  best {label:<15} iteration {point['iteration']:>6}   "
              f"reward/step {point['reward_per_step']:.4f}   "
              f"episode {_fmt(point['episode_steps'], '.1f')}")
    final = data.get("final")
    if final:
        print(f"  final                iteration {final['iteration']:>6}   "
              f"reward/step {final['reward_per_step']:.4f}   "
              f"episode {_fmt(final['episode_steps'], '.1f')}")

    shape = data["trend"]
    print()
    print(f"trend from the reward peak to the end: {shape['verdict']}")
    if shape["verdict"] == "hazard-19":
        print(f"  reward {shape['reward_delta']:+.4f} while episode length "
              f"{shape['episode_delta']:+.1f} steps.")
        print("  HAZARD 19: the policy is learning to score, not to survive.")
    elif shape["verdict"] == "survival-diverges":
        print(f"  reward {shape['reward_delta']:+.4f} while episode length "
              f"{shape['episode_delta']:+.1f} steps.")
        print("  Survival rose while the trainer's scalar fell. This is NOT a")
        print("  regression by the metric this repository cares about, and a")
        print("  supervisor that stopped at the reward peak would have thrown")
        print("  the longer-surviving policy away. compare settles it.")
    elif shape["verdict"] == "agreeing":
        print(f"  reward {shape['reward_delta']:+.4f}, episode length "
              f"{shape['episode_delta']:+.1f} steps — moving together.")
    else:
        print(f"  {shape.get('detail', '')}")

    print()
    print(f"witness   ({len(data['witness'])} margin(s) recorded, floor "
          f"{data['witness_floor']:.0f}x)")
    if not data["witness"]:
        print("  none in train.log. The trainer prints one when it finishes;")
        print("  a run without one did not get that far.")
    for item in data["witness"]:
        mark = "  BELOW FLOOR — read MUJOCO.md hazard 13" if item["factor"] < data["witness_floor"] else ""
        print(f"  {item['error']:.3e}  {item['factor']:.0f}x inside tolerance{mark}")

    print()
    files = {item["path"] for item in data["checkpoints"]}
    print(f"checkpoints  ({len(data['checkpoints'])} records over "
          f"{len(files)} files; a 'best' record names a file the next one "
          f"overwrites)")
    print(f"  {'iteration':>9} {'tag':>8} {'reward/step':>12} {'bytes':>9} "
          f"sha256    digest")
    for item in data["checkpoints"]:
        digest = str(item.get("observed_sha256") or item.get("declared_sha256") or "")[:8]
        if not item.get("exists"):
            state = "MISSING"
        elif item.get("superseded"):
            state = "superseded"
        elif item.get("digest_matches") is True:
            state = "ok"
        elif item.get("digest_matches") is False:
            state = "MISMATCH"
        else:
            state = "unchecked"
        print(
            f"  {_fmt(item['iteration'], '>9')} {str(item.get('tag') or ''):>8} "
            f"{_fmt(item.get('reward_per_step'), '>12.4f')} "
            f"{_fmt(item.get('declared_bytes'), '>9')} {digest}  {state}"
        )
    if data["checkpoints_missing"]:
        print(f"  MISSING: {', '.join(data['checkpoints_missing'])}")
    if data["checkpoints_mismatched"]:
        print(f"  DIGEST MISMATCH: {', '.join(data['checkpoints_mismatched'])}")

    if data["log_warnings"]:
        print()
        print("log warnings")
        for line in data["log_warnings"]:
            print(f"  {line}")

    advice = data["recommendations"]
    print()
    print("worth playing")
    for item in advice["play"]:
        print(f"  {item.get('name', '?'):<26} {item['reason']}")
    print()
    print(f"  {advice['note']}")
    if advice["compare_command"]:
        print()
        print(f"  {advice['compare_command']}")


# ---------------------------------------------------------------------------
# Watch mode
# ---------------------------------------------------------------------------


def _stop(run_dir: Path, pid: int, reason: str, *, grace: float = 60.0) -> dict[str, Any]:
    """SIGTERM, timed away from the checkpoint writer.

    There is no trainer-side early stop (wishlist #3), so stopping means
    signalling. Losing a half-written ``.cxpolicy`` is cheap; losing the
    run's last complete one is not — so before signalling, wait until the
    newest checkpoint file has stopped growing.
    """

    record: dict[str, Any] = {"reason": reason, "pid": pid, "signalled": False}
    newest = max(
        (path for path in run_dir.glob("*.cxpolicy")),
        key=lambda path: path.stat().st_mtime,
        default=None,
    )
    if newest is not None:
        size = -1
        for _ in range(20):
            current = newest.stat().st_size
            if current == size:
                break
            size, _ = current, time.sleep(0.5)
        record["settled_on"] = newest.name

    try:
        os.kill(pid, signal.SIGTERM)
        record["signalled"] = True
    except (ProcessLookupError, PermissionError) as exc:
        record["error"] = str(exc)
        return record

    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if process_gone(pid):
            record["exited"] = True
            return record
        time.sleep(1.0)
    record["exited"] = False
    return record


def divergence(progress: dict[str, Any], sigma_floor: float) -> dict[str, Any] | None:
    """``DESIGN.md`` §6's ``divergence | loss, action_std | NaN, or σ collapse``.

    Returns the finding, or ``None`` if the run looks healthy.

    **This is stop-the-burn, not checkpoint selection.** Both branches say
    the same thing — the optimiser has left the region where its numbers mean
    anything — and neither says which checkpoint to keep. A diverged run's
    earlier checkpoints are still perfectly good, which is exactly why this
    stops the process rather than deleting anything.

    Why it is not gated behind reward patience: patience is switched *off* by
    default here (experiment 001 Phase A found it stops runs that are
    working). Divergence is the opposite kind of signal — it is not a
    judgement about whether the policy is improving, it is an arithmetic fact
    about whether the numbers are still numbers. A NaN at 02:00 otherwise
    burns the rest of the night, silently, at full power.
    """

    # A non-finite loss or reward is unambiguous: no later iteration recovers
    # from it, because the gradient that produced it has already been applied.
    for field in ("loss", "reward_per_step"):
        value = progress.get(field)
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            return {
                "check": "non-finite",
                "field": field,
                "value": repr(value),
                "reason": f"{field} is {value!r} — the optimiser has diverged",
            }

    sigma = progress.get("action_std")
    if (
        sigma_floor > 0
        and isinstance(sigma, (int, float))
        and math.isfinite(float(sigma))
        and float(sigma) < sigma_floor
    ):
        return {
            "check": "sigma-collapse",
            "field": "action_std",
            "value": float(sigma),
            "floor": sigma_floor,
            "reason": (
                f"action_std {float(sigma):.5f} is below the floor "
                f"{sigma_floor:g} — exploration has collapsed"
            ),
        }

    # Note the deliberate absence of an `else` that reports health. A missing
    # or null action_std means an older trainer that never logged one (see
    # runlog's docstring), and a run whose sigma is unknown must not be
    # reported as a run whose sigma is nought.
    return None


def watch(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Poll ``progress.json`` and act. Returns what the watch did.

    Prints one line per observed iteration change — reward **and** episode
    length, always — so a run that is going wrong is visible while it is
    going wrong rather than in a post-mortem.
    """

    events: list[dict[str, Any]] = []
    last_iteration = -1
    started = time.monotonic()
    stopped: dict[str, Any] | None = None

    print(f"watching {run_dir}  (poll {args.poll:g}s"
          + (f", patience {args.patience}" if args.patience else "")
          + (f", min-iterations {args.min_iterations}" if args.min_iterations else "")
          + ")")

    while True:
        progress = read_progress(run_dir)
        state = str(progress.get("state") or "")
        iteration = progress.get("iteration")
        pid_file = run_dir / "train.pid"
        pid = 0
        if pid_file.is_file():
            try:
                pid = int(pid_file.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                pid = 0

        if isinstance(iteration, int) and iteration != last_iteration:
            last_iteration = iteration
            print(
                f"  it {iteration:>6}/{_fmt(progress.get('total'))}  "
                f"reward/step {_fmt(progress.get('reward_per_step'), '+.4f')}  "
                f"episode {_fmt(progress.get('episode_steps'), '.1f')}  "
                f"sigma {_fmt(progress.get('action_std'), '.4f')}  "
                f"best {_fmt(progress.get('best_iteration'))}"
                f"@{_fmt(progress.get('best_reward_per_step'), '.4f')}  "
                f"{state}"
            )

        # Device guard first: a GPU dispatch that fell back to CPU is not a
        # slow run, it is the wrong run, and every minute of it is wasted.
        device = str(progress.get("device") or "")
        if args.require_device and device and device != args.require_device:
            events.append({"event": "wrong-device", "device": device})
            if pid:
                stopped = _stop(run_dir, pid,
                                f"device is {device!r}, --require-device "
                                f"{args.require_device!r}")
            break

        if state in {"done", "error", "stopped", "failed", "cancelled"}:
            events.append({"event": "terminal", "state": state})
            break

        if pid and process_gone(pid):
            events.append({"event": "process-gone", "pid": pid,
                           "state": state, "iteration": iteration})
            print(f"  process {pid} is gone while state is {state!r} — "
                  "this run is stale, not live.")
            break

        # Divergence before patience, and unconditionally: patience is off by
        # default, so a run left to its own devices overnight has this as its
        # only arithmetic guard.
        diverged = divergence(progress, getattr(args, "sigma_floor", SIGMA_FLOOR))
        if diverged is not None:
            print(f"  STOPPING: {diverged['reason']}")
            events.append({"event": "divergence", "iteration": iteration, **diverged})
            if pid:
                stopped = _stop(run_dir, pid, diverged["reason"])
            break

        best_iteration = progress.get("best_iteration")
        if (
            args.patience
            and isinstance(iteration, int) and isinstance(best_iteration, int)
            and iteration >= (args.min_iterations or 0)
            and iteration - best_iteration > args.patience
        ):
            reason = (
                f"{iteration - best_iteration} iterations since the best "
                f"({best_iteration}), patience {args.patience}"
            )
            print(f"  STOPPING: {reason}")
            events.append({"event": "patience-exceeded", "iteration": iteration,
                           "best_iteration": best_iteration})
            if pid:
                stopped = _stop(run_dir, pid, reason)
            break

        if args.timeout and (time.monotonic() - started) > args.timeout:
            events.append({"event": "watch-timeout", "seconds": args.timeout})
            print(f"  watch timed out after {args.timeout:g}s (the run is "
                  "still going; this stops watching, not training)")
            break

        time.sleep(args.poll)

    return {"events": events, "stopped": stopped}


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harness supervise",
        description="Watch a training run, or read a finished one.",
    )
    parser.add_argument("--run", required=True, help="A training run directory.")
    parser.add_argument(
        "--report-only", action="store_true",
        help="Read and report. The default when --watch is absent.",
    )
    parser.add_argument(
        "--watch", action="store_true",
        help="Attach to a live run: poll, act, then report.",
    )
    parser.add_argument("--poll", type=float, default=10.0, help="Seconds between polls.")
    parser.add_argument(
        "--patience", type=int, default=0,
        help="Stop when this many iterations have passed since the best. "
             "0 disables stopping entirely.",
    )
    parser.add_argument(
        "--min-iterations", type=int, default=0,
        help="Never stop before this iteration — guards the initial plateau, "
             "where an untrained network can look stable.",
    )
    parser.add_argument(
        "--require-device", default="",
        help="Stop immediately if progress.json reports another device.",
    )
    parser.add_argument(
        "--timeout", type=float, default=0.0,
        help="Give up watching after this many seconds. Does not stop the run.",
    )
    parser.add_argument(
        "--sigma-floor", type=float, default=SIGMA_FLOOR,
        help=f"Stop when action_std falls below this. Default {SIGMA_FLOOR:g}, "
             "which is 5%% of the initial_std of the run this floor was read "
             "off. 0 disables the sigma half of the divergence guard; the "
             "non-finite half cannot be disabled.",
    )
    parser.add_argument(
        "--no-verify-checkpoints", action="store_true",
        help="Skip re-hashing every .cxpolicy against its declared digest.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the envelope.")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_env_file()
    args = build_parser().parse_args(argv)
    run_dir = Path(args.run).expanduser().resolve()
    if not run_dir.is_dir():
        print(f"no such run directory: {run_dir}", file=sys.stderr)
        return EXIT_USAGE

    watched: dict[str, Any] | None = None
    if args.watch:
        try:
            watched = watch(run_dir, args)
        except KeyboardInterrupt:
            print("\n  interrupted — reporting on what is there.")
            watched = {"events": [{"event": "interrupted"}], "stopped": None}
        print()

    try:
        run = load_run(run_dir, verify_checkpoints=not args.no_verify_checkpoints)
    except OSError as exc:
        print(f"could not read {run_dir}: {exc}", file=sys.stderr)
        return EXIT_INFRASTRUCTURE

    data = build_report(run)
    if watched is not None:
        data["watch"] = watched

    if args.json:
        json.dump(envelope("supervise", True, data), sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        report(data)

    # A stale run is a finding, not a failure of this driver: it read the
    # directory correctly and the directory is lying. Exit 0 and say so.
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
