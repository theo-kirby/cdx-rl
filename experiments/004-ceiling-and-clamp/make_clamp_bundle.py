#!/usr/bin/env python
"""Derive `tasks/stand-b8-clamp15/` from `tasks/stand-b8/` by capping the
action ranges. Idempotent; prints both digests.

    uv run python experiments/004-ceiling-and-clamp/make_clamp_bundle.py

The cap is **±15°**, chosen because 003's gate measures the position servo
saturating at **16.4° of error** (86 N·mm ÷ 5.236 N·mm/deg) and the trained
policies command up to 44° on a ±45° joint. Every command now sits below the
saturation threshold.

**The MJCF is copied unchanged and that is not an oversight.** Every actuator
carries `ctrllimited="false"`, so the model never clamped `ctrl`; the action
range lived entirely in the bundle's action table. The same file also carries
`forcerange="-0.086 0.086"`, which is where the 86 N·mm figure comes from —
it is a model limit, not the judgment it is often called.

**This is a bundle edit rather than a mechanism edit, and that is required
rather than expedient.** In Cadex a position servo's action range *is* its
joint's physical limits — `CadexDynamics._ACTION_SOURCES` maps
`("position", "angular")` to `angle_limits_degrees` — and the exported MJCF
confirms it, carrying joint ranges that are the same ten numbers as the action
table. Capping the range in `script.py` would narrow the **joint** too, so the
machine could not flex past 15°, which is a different and confounded
experiment: it removes reachable configurations for reasons unrelated to
torque.

There is no way to say "a joint that moves ±45° and a policy that may only
command ±15°" in the mechanism vocabulary. That is `cadex-wishlist.md` #12,
and since 2026-08-05 it is a PR against `/home/theo/cadex-prs` rather than a
wish — `cadex-engine-plan.md` §2 is the spec. Until it lands, editing the
derived bundle is the only expression of this experiment, and 004-B is a
claim about committed bytes rather than a re-derivable one.

**This script is the thing the PR has to reproduce.** When #12 lands,
regenerate the clamped bundle from `script.py` and diff its action table
against this script's output; a byte-identical table retires the workaround.
One known difference to expect, and it is this script's semantics rather than
the PR's: `_clamped_action` below takes `abs()` of both bounds and writes a
**symmetric** ±cap, while the engine surface clamps each bound
independently. That is a no-op for `stand-b8`, whose joint ranges are already
symmetric, and it would not be for a mechanism with an asymmetric joint.

Note what the cap does and does not do. The servo sees **error = command −
actual**, so a joint displaced 45° while commanded −15° still presents 60° of
error. The clamp does not prevent saturation; it removes the policy's ability
to *choose* it. Any saturation that survives is coming from the dynamics
rather than from the network asking for a setpoint three times past the
threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "tasks" / "stand-b8"

# The cap is a parameter because 004's pre-registered fork has a second arm at
# ±25°: still above the 16.4° saturation threshold, but far below the ±45° the
# trained policies were using. Which one runs is decided by the rule in
# `README.md` §7, not by looking at a curve.
CAPS = {15.0: ("stand-b8-clamp15", "stand11"),
        25.0: ("stand-b8-clamp25", "stand12")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cap", type=float, default=15.0, choices=sorted(CAPS))
    args = ap.parse_args(argv)
    dst_name, LABEL = CAPS[args.cap]
    DST = REPO / "tasks" / dst_name
    CAP_DEGREES = args.cap

    task = json.loads((SRC / "stand-task.json").read_text())

    changed = []
    for action in task["actions"]:
        low, high = abs(float(action["low"])), abs(float(action["high"]))
        if max(low, high) > CAP_DEGREES:
            changed.append((action["actuator"], high, CAP_DEGREES))
        action["low"] = -min(low, CAP_DEGREES)
        action["high"] = min(high, CAP_DEGREES)
    task["label"] = LABEL

    DST.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(task, indent=1, sort_keys=True).encode()
    (DST / "stand-task.json").write_bytes(blob)
    shutil.copy2(SRC / "model-model.xml", DST / "model-model.xml")

    parent = hashlib.sha256((SRC / "stand-task.json").read_bytes()).hexdigest()
    model = hashlib.sha256((DST / "model-model.xml").read_bytes()).hexdigest()
    print(f"parent bundle  {parent}")
    print(f"clamped bundle {hashlib.sha256(blob).hexdigest()}")
    print(f"model (copied) {model}")
    print(f"\ncapped at ±{CAP_DEGREES:.0f}°, {len(changed)} of "
          f"{len(task['actions'])} actions narrowed:")
    for name, was, now in changed:
        print(f"  {name:24s} ±{was:5.1f} -> ±{now:5.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
