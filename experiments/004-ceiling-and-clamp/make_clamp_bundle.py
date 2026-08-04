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

**This is a bundle edit, not a mechanism edit, and the difference is a debt.**
The ranges originate in `angle_limits_degrees` on the mechanism, so the honest
version of this change is a `script.py` edit. It is not done here because
`sb1x` cannot build `script.py` at all — its pinned Cadex `06d1374b` rejects
`centre_of_mass_velocity`, and only `mmini` (`560935bd`) accepts the full
observation set. Until that is carried back, 004-B is a claim about committed
bytes rather than a re-derivable one.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "tasks" / "stand-b8"
DST = REPO / "tasks" / "stand-b8-clamp15"
CAP_DEGREES = 15.0
LABEL = "stand11"


def main() -> int:
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
