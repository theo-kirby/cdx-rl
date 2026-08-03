#!/usr/bin/env python
"""Make ``supervise``'s divergence guard fire, on purpose, both branches.

``harness/DESIGN.md`` §6: *"every check must be able to fail, and must have
been made to fail once on purpose."* A guard that has never fired is a guard
that is believed rather than known, and the failure mode is silent — the run
it was supposed to stop just keeps going.

So this builds a **synthetic run directory** with a real live process in it
(a ``sleep``, standing in for the trainer), writes a ``progress.json`` that a
diverged run would have written, and points the real ``watch()`` at it. The
guard passes only if the process is actually dead afterwards. Nothing is
mocked: this is ``harness.supervise.watch`` and ``harness.supervise._stop``,
the same code the sweep runs under.

```
uv run python tools/fire_divergence_guard.py
```

Exit 0 if both branches fired and killed their process; 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from harness.supervise import SIGMA_FLOOR, build_parser, divergence, watch  # noqa: E402

#: What a healthy iteration of the run this floor was read off looks like.
#: Each case below perturbs exactly one field of it, so that what fired is
#: never in question.
HEALTHY: dict[str, Any] = {
    "schema": "cadex-training-progress-v1",
    "label": "synthetic",
    "state": "training",
    "device": "gpu",
    "iteration": 300,
    "total": 1500,
    "reward_per_step": 0.21,
    "episode_steps": 288.4,
    "action_std": 0.3801,
    "loss": 24.7,
    "best_iteration": 298,
    "best_reward_per_step": 0.22,
    "error": "",
    "out": "synthetic.cxpolicy",
    "checkpoints": [],
}

CASES: list[tuple[str, dict[str, Any], str]] = [
    (
        "non-finite loss",
        {"loss": float("nan")},
        "non-finite",
    ),
    (
        "sigma collapse",
        {"action_std": 0.004},
        "sigma-collapse",
    ),
]


def fire(name: str, patch: dict[str, Any], expect: str, root: Path) -> dict[str, Any]:
    """One case: live process, poisoned progress file, real ``watch()``."""

    run_dir = root / name.replace(" ", "-")
    run_dir.mkdir(parents=True, exist_ok=True)

    # A real process, in its own session so terminating it cannot reach us.
    victim = subprocess.Popen(
        ["sleep", "600"], start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    (run_dir / "train.pid").write_text(str(victim.pid), encoding="utf-8")

    progress = dict(HEALTHY) | patch
    # allow_nan is the default, and is the whole reason a NaN survives the
    # round trip through progress.json at all.
    (run_dir / "progress.json").write_text(json.dumps(progress), encoding="utf-8")

    args = build_parser().parse_args([
        "--run", str(run_dir), "--watch", "--poll", "0.2",
        "--patience", "0", "--timeout", "20",
    ])

    print(f"--- {name} " + "-" * (56 - len(name)))
    print(f"  patched      {patch}")
    print(f"  victim pid   {victim.pid} (alive: {victim.poll() is None})")

    started = time.monotonic()
    result = watch(run_dir, args)
    elapsed = time.monotonic() - started

    # Did the guard actually kill it? Ask the OS, not the return value.
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and victim.poll() is None:
        time.sleep(0.2)
    dead = victim.poll() is not None
    if not dead:  # do not leave a stray sleep behind on failure
        victim.kill()
        victim.wait()

    events = [event["event"] for event in result["events"]]
    checks = [event.get("check") for event in result["events"]]
    passed = "divergence" in events and expect in checks and dead

    print(f"  events       {events}")
    print(f"  check        {checks}")
    print(f"  stopped      {json.dumps(result['stopped'], sort_keys=True)}")
    print(f"  process dead {dead}   after {elapsed:.2f}s")
    print(f"  RESULT       {'FIRED' if passed else 'DID NOT FIRE'}")
    print()
    return {
        "case": name, "patch": {k: repr(v) for k, v in patch.items()},
        "expected_check": expect, "events": events, "checks": checks,
        "process_dead": dead, "seconds": round(elapsed, 2), "passed": passed,
    }


def control() -> dict[str, Any]:
    """The healthy point must NOT fire — a guard that always fires is useless."""

    verdict = divergence(dict(HEALTHY), SIGMA_FLOOR)
    print("--- control (healthy progress.json) " + "-" * 22)
    print(f"  action_std {HEALTHY['action_std']}, loss {HEALTHY['loss']}, "
          f"floor {SIGMA_FLOOR:g}")
    print(f"  divergence() returned {verdict!r}")
    print(f"  RESULT       {'CORRECTLY SILENT' if verdict is None else 'FALSE POSITIVE'}")
    print()

    # And an older trainer's run, which logs no action_std at all, must read as
    # unknown rather than as zero (runlog's docstring).
    older = dict(HEALTHY)
    older["action_std"] = None
    silent = divergence(older, SIGMA_FLOOR)
    print("--- control (older trainer, no action_std) " + "-" * 15)
    print(f"  divergence() returned {silent!r}")
    print(f"  RESULT       {'CORRECTLY SILENT' if silent is None else 'FALSE POSITIVE'}")
    print()
    return {"healthy_silent": verdict is None, "missing_sigma_silent": silent is None}


def main() -> int:
    parser = argparse.ArgumentParser(prog="tools/fire_divergence_guard.py")
    parser.add_argument("--json", action="store_true")
    options = parser.parse_args()

    print("Firing supervise's divergence guard on purpose (DESIGN.md §6).")
    print(f"sigma floor {SIGMA_FLOOR:g}\n")

    controls = control()
    with tempfile.TemporaryDirectory(prefix="fire-divergence-") as raw:
        root = Path(raw)
        results = [fire(name, patch, expect, root) for name, patch, expect in CASES]

    ok = all(item["passed"] for item in results) and all(controls.values())
    print("=" * 64)
    for item in results:
        print(f"  {item['case']:<18} {'FIRED' if item['passed'] else 'DID NOT FIRE'}"
              f"   (killed pid in {item['seconds']:.2f}s)")
    print(f"  {'healthy control':<18} "
          f"{'silent' if controls['healthy_silent'] else 'FALSE POSITIVE'}")
    print(f"  {'no-sigma control':<18} "
          f"{'silent' if controls['missing_sigma_silent'] else 'FALSE POSITIVE'}")
    print()
    print("PASS — both branches stop a live process" if ok else "FAIL")

    if options.json:
        json.dump({"results": results, "controls": controls, "passed": ok},
                  sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
