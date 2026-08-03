#!/usr/bin/env python
"""Make ``supervise``'s wall-cap projection fire, on purpose, and stay silent
when it should.

``harness/DESIGN.md`` §6: *"every check must be able to fail, and must have
been made to fail once on purpose."* This guard's failure mode is the quiet
one — if it never fires, the seed simply trains for hours and is then
terminated mid-run, leaving checkpoints, no final ``.cxpolicy`` and no witness.
That artefact is indistinguishable from a run stopped on purpose, which is
why experiment 002 already has one (``results/stopped.md``) and why a wall cap
carried unchanged onto a slower card is worth catching at iteration 10.

Same shape as ``fire_divergence_guard.py``: a **synthetic run directory** with
a real live process standing in for the trainer, a ``progress.json`` a real run
would have written, and the real ``watch()`` pointed at it. Nothing is mocked.

The projection **warns and does not stop** — unlike divergence, a slow run is
still a valid run, and whether to keep it is the operator's call. So every
case here also asserts the victim is *still alive* afterwards.

```
uv run python tools/fire_projection_guard.py
```

Exit 0 if the guard fires when it should and is silent when it should be.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from harness.supervise import PROJECTION_AFTER, build_parser, watch  # noqa: E402

#: The wall cap every case is judged against, in seconds. Small so the watch
#: loop ends promptly; the projection arithmetic does not care about scale.
CAP = 3.0

#: A healthy mid-run point. Each case perturbs only the three fields the
#: projection reads — ``iteration``, ``total``, ``wall_time_s`` — so what
#: fired, or did not, is never in question.
HEALTHY: dict[str, Any] = {
    "schema": "cadex-training-progress-v1",
    "label": "synthetic",
    "state": "training",
    "device": "gpu",
    "iteration": 10,
    "total": 1500,
    "wall_time_s": 150.0,
    "reward_per_step": 0.21,
    "episode_steps": 288.4,
    "action_std": 0.3801,
    "loss": 24.7,
    "best_iteration": 8,
    "best_reward_per_step": 0.22,
    "error": "",
    "out": "synthetic.cxpolicy",
    "checkpoints": [],
}

CASES: list[tuple[str, dict[str, Any], bool, str]] = [
    (
        # 13.6 s/iteration is what sb9x actually measured at 2048 environments.
        # 1500 of them is 5.7 h; the cap here stands in for "far less".
        "too slow for the cap",
        {"iteration": 10, "wall_time_s": 150.0, "total": 1500},
        True,
        "13.6 s/it x 1500 = 5.7 h, which does not fit",
    ),
    (
        "comfortably inside the cap",
        {"iteration": 10, "wall_time_s": 0.011, "total": 1500},
        False,
        "1.5 ms/it x 1500 = 1.5 s, which fits in 3 s",
    ),
    (
        "too early to project",
        {"iteration": PROJECTION_AFTER - 1, "wall_time_s": 150.0, "total": 1500},
        False,
        f"iteration {PROJECTION_AFTER - 1} < PROJECTION_AFTER "
        f"{PROJECTION_AFTER}; compile still dominates",
    ),
]


def moving(root: Path) -> dict[str, Any]:
    """The slope path: a ``progress.json`` that advances while ``watch()`` reads it.

    The static cases above leave ``watch()`` with one sample, so it falls back
    to the average and inherits iteration 0's compile cost. A real run gives
    it two, and the slope between them is what it should be believing. This
    case makes that difference visible: iteration 0 costs 65 s and every
    iteration after it costs 1 s, so the average at iteration 12 reads
    ~6 s/iteration while the true rate is 1. Only the slope gets it right.
    """

    run_dir = root / "moving"
    run_dir.mkdir(parents=True, exist_ok=True)
    victim = subprocess.Popen(
        ["sleep", "600"], start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    (run_dir / "train.pid").write_text(str(victim.pid), encoding="utf-8")

    compile_s, steady_s, total = 65.0, 1.0, 1500

    def write(iteration: int) -> None:
        (run_dir / "progress.json").write_text(
            json.dumps(dict(HEALTHY) | {
                "iteration": iteration, "total": total,
                "wall_time_s": compile_s + steady_s * iteration,
            }), encoding="utf-8"
        )

    write(1)
    # 1 s/iteration x 1500 = 1500 s. A cap of 900 s does not fit, so the
    # guard must fire; but the *average* at iteration 12 is 6.1 s/iteration,
    # which would put the projection at 9200 s — right verdict, wrong number.
    cap = 900.0
    args = build_parser().parse_args([
        "--run", str(run_dir), "--watch", "--poll", "0.2",
        "--patience", "0", "--timeout", "6",
    ])
    args.timeout = cap  # the cap being projected against, not the watch budget

    stop = threading.Event()

    def advance() -> None:
        iteration = 1
        while not stop.is_set() and iteration <= 14:
            time.sleep(0.25)
            iteration += 1
            write(iteration)

    mover = threading.Thread(target=advance, daemon=True)
    mover.start()

    print("--- moving progress (slope, not average) " + "-" * 16)
    print(f"  synthetic    iteration 0 costs {compile_s:g}s, "
          f"then {steady_s:g}s each; {total} total, cap {cap:g}s")
    print(f"  average at 12 would read "
          f"{(compile_s + steady_s * 12) / 13:.2f} s/it; true rate {steady_s:g}")

    # watch() has no terminal state to find, so end it by marking the run done
    # once the mover is finished.
    def finish() -> None:
        mover.join()
        (run_dir / "progress.json").write_text(
            json.dumps(dict(HEALTHY) | {"state": "done", "iteration": 14,
                                        "total": total}), encoding="utf-8"
        )

    threading.Thread(target=finish, daemon=True).start()
    result = watch(run_dir, args)
    stop.set()

    detail = next(
        (e for e in result["events"] if e["event"] == "wall-cap-too-short"), None
    )
    alive = victim.poll() is None
    victim.kill()
    victim.wait()

    measured = detail["seconds_per_iteration"] if detail else None
    # The slope must recover the true rate, not the compile-inflated average.
    accurate = measured is not None and abs(measured - steady_s) < 0.2
    passed = detail is not None and accurate and alive

    print(f"  events       {[e['event'] for e in result['events']]}")
    print(f"  measured     {measured} s/iteration  "
          f"(true {steady_s:g}, average would be "
          f"{(compile_s + steady_s * 12) / 13:.2f})")
    print(f"  RESULT       {'PASS' if passed else 'FAIL'}")
    print()
    return {"case": "moving progress", "fired": detail is not None,
            "measured_s_per_iteration": measured, "true_s_per_iteration": steady_s,
            "slope_accurate": accurate, "run_survived": alive, "passed": passed}


def fire(name: str, patch: dict[str, Any], expect: bool, why: str,
         root: Path) -> dict[str, Any]:
    """One case: live process, a progress file, the real ``watch()``."""

    run_dir = root / name.replace(" ", "-")
    run_dir.mkdir(parents=True, exist_ok=True)

    victim = subprocess.Popen(
        ["sleep", "600"], start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    (run_dir / "train.pid").write_text(str(victim.pid), encoding="utf-8")
    (run_dir / "progress.json").write_text(
        json.dumps(dict(HEALTHY) | patch), encoding="utf-8"
    )

    args = build_parser().parse_args([
        "--run", str(run_dir), "--watch", "--poll", "0.2",
        "--patience", "0", "--timeout", str(CAP),
    ])

    print(f"--- {name} " + "-" * max(0, 56 - len(name)))
    print(f"  patched      {patch}")
    print(f"  why          {why}")

    started = time.monotonic()
    result = watch(run_dir, args)
    elapsed = time.monotonic() - started

    events = [event["event"] for event in result["events"]]
    fired = "wall-cap-too-short" in events
    detail = next(
        (e for e in result["events"] if e["event"] == "wall-cap-too-short"), None
    )

    # A projection warns; it must not stop anything.
    alive = victim.poll() is None
    victim.kill()
    victim.wait()

    passed = (fired == expect) and alive

    print(f"  events       {events}")
    if detail:
        print(f"  projection   {detail['seconds_per_iteration']} s/it, "
              f"{detail['projected_seconds']} s needed vs cap "
              f"{detail['timeout']:g} s, reaches iteration "
              f"{detail['reaches_iteration']} of {detail['total']}")
    print(f"  fired        {fired}   (expected {expect})")
    print(f"  run survived {alive}   (a projection warns, it does not stop)")
    print(f"  RESULT       {'PASS' if passed else 'FAIL'}   after {elapsed:.2f}s")
    print()
    return {
        "case": name, "patch": patch, "expected_fire": expect,
        "fired": fired, "events": events, "projection": detail,
        "run_survived": alive, "seconds": round(elapsed, 2), "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="tools/fire_projection_guard.py")
    parser.add_argument("--json", action="store_true")
    options = parser.parse_args()

    print("Firing supervise's wall-cap projection on purpose (DESIGN.md §6).")
    print(f"cap {CAP:g}s, projecting after iteration {PROJECTION_AFTER}\n")

    with tempfile.TemporaryDirectory(prefix="fire-projection-") as raw:
        results = [fire(*case, Path(raw)) for case in CASES]
        results.append(moving(Path(raw)))

    ok = all(item["passed"] for item in results)
    print("=" * 64)
    for item in results:
        print(f"  {item['case']:<28} "
              f"{'fired' if item['fired'] else 'silent':<7}"
              f"{'PASS' if item['passed'] else 'FAIL'}")
    print()
    print("PASS — the projection fires only when the cap is too short"
          if ok else "FAIL")

    if options.json:
        json.dump({"results": results, "passed": ok},
                  sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
