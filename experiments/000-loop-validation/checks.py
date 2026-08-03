#!/usr/bin/env python
"""Experiment 000's gate: the three checks that replace ``feasibility``.

Runs under the **trainer interpreter**, like everything that imports mujoco::

    /home/theo/cadex-train-venv/bin/python experiments/000-loop-validation/checks.py \\
        --model <path to model-model.xml> --task <path to task-task.json>

``feasibility``'s six checks are not run here, and §5 of this experiment's
README says why: four of the six are about a floating base standing on a
floor, and a grounded pendulum has neither. Running them would produce four
green rows that measured nothing, which is hazard 18 exactly.

What replaces them:

1. **The MJCF loads in stock MuJoCo 3.10.0** and reports 1 dof, 1 actuator,
   2 sensors. Link 3 of the ten.
2. **The bundle's own digests check out** — the model the bundle names is the
   model on disk.
3. **At zero torque from the `solved` keyframe, the arm falls.** A pendulum
   that does not fall under gravity has a units bug, and this is the cheapest
   possible detector for one. The check asserts the hinge moves by a real
   angle, not merely that it moves: a 1e-6 rad drift over two seconds would
   pass a "did it change" test and would still be a broken model.

Every one of these can fail, which is the bar ``harness/DESIGN.md`` §6 sets.
Check 3 was made to fail on purpose during the build by asserting the arm
*rises*, and it reported the fall correctly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

#: How far the arm must swing under gravity in the drop window before this
#: counts as "it fell". 5° is far beyond any plausible solver drift and far
#: below what a 200 mm arm released horizontally actually does.
FALL_THRESHOLD_DEG = 5.0

#: How long to integrate the drop. Two seconds is the task's own episode
#: length, so the check runs the same physics for the same duration the
#: trainer will.
DROP_SECONDS = 2.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--module-dir", default="/home/theo/cadex/src/Mod/cadex")
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    sys.path.insert(0, args.module_dir)
    import mujoco  # noqa: PLC0415
    import CadexDynamics as cd  # noqa: PLC0415

    model_path = Path(args.model)
    task_path = Path(args.task)
    xml = model_path.read_bytes()
    task = json.loads(task_path.read_text(encoding="utf-8"))

    checks: list[dict[str, Any]] = []

    def record(name: str, ok: bool, detail: Any) -> bool:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        return bool(ok)

    # -- 1. the MJCF loads, and is the shape rig.py declared -----------------
    model = cd.load_model(xml)
    shape = {
        "mujoco_version": mujoco.__version__,
        "nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu),
        "nsensor": int(model.nsensor), "nbody": int(model.nbody),
        "njnt": int(model.njnt),
    }
    record(
        "mjcf_loads_1dof_1actuator_2sensors",
        model.nv == 1 and model.nu == 1 and model.nsensor == 2,
        shape,
    )

    # -- 2. the bundle names this model -------------------------------------
    declared = str((task.get("model") or {}).get("sha256") or "")
    observed = hashlib.sha256(xml).hexdigest()
    record(
        "bundle_names_this_model",
        bool(declared) and declared == observed,
        {"declared": declared, "observed": observed,
         "task_sha256": hashlib.sha256(task_path.read_bytes()).hexdigest(),
         "mujoco_version_in_bundle": task.get("mujoco_version")},
    )

    # -- 3. the drop test ---------------------------------------------------
    # From the exported keyframe, no torque, integrate, and demand the hinge
    # has actually moved. `evaluate_episode` is not used here on purpose:
    # this must be the plainest possible MuJoCo, so that a failure points at
    # the model rather than at the engine's episode loop.
    data = mujoco.MjData(model)
    keyframe = str(task["episode"].get("reset_keyframe") or "")
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    start = float(data.qpos[0])
    data.ctrl[:] = 0.0
    steps = int(DROP_SECONDS / float(model.opt.timestep))
    extreme = start
    for _ in range(steps):
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)
        if abs(float(data.qpos[0]) - start) > abs(extreme - start):
            extreme = float(data.qpos[0])
    end = float(data.qpos[0])

    swept_deg = abs(math.degrees(extreme - start))
    record(
        "falls_under_gravity_at_zero_torque",
        swept_deg > FALL_THRESHOLD_DEG,
        {
            "keyframe": keyframe or "(none — reset to zero)",
            "keyframe_found": key_id >= 0,
            "start_rad": start, "end_rad": end, "extreme_rad": extreme,
            "swept_degrees": round(swept_deg, 3),
            "threshold_degrees": FALL_THRESHOLD_DEG,
            "seconds": DROP_SECONDS,
            "solver_timestep_s": float(model.opt.timestep),
            "gravity": [float(value) for value in model.opt.gravity],
        },
    )

    ok = all(item["ok"] for item in checks)
    envelope = {
        "schema": "cdxrl-000-checks-v1", "ok": ok,
        "model": str(model_path), "task": str(task_path), "checks": checks,
    }
    if args.json:
        json.dump(envelope, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        for item in checks:
            mark = "ok  " if item["ok"] else "FAIL"
            print(f"{mark}  {item['check']:<36} "
                  f"{json.dumps(item['detail'], sort_keys=True)}")
        print()
        print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
