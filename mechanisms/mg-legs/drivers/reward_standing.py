#!/usr/bin/env python
"""What does the reward pay a machine that is standing still and doing nothing?

    CADEX_REPO=~/cadex pixi run python ~/cdx-mjc/reward_standing.py

The positive-kernel analogue of hazard 9, and the check B8 has to pass before
anything is dispatched.

Through B7 every shaping term was a COST and the rule was "every term must
read ~0 at the standing pose", because a term that is non-zero there charges
rent for standing still. This project measured what breaking that rule costs:
the same reward written against an absolute channel with a large offset
trained WORSE THAN NOT TRAINING, 4.46 -> 3.66, where the displacement form
went -0.243 -> -0.028.

B8 turns every shaping term into `w * exp(-(e/s)^2)`, which is `w` when the
error is 0. So the rule reads the other way round and is exactly as
diagnostic: EVERY TERM MUST PAY ITS OWN WEIGHT AT THE NOMINAL POSE. A term
that does not is either written against an expression that is non-zero
standing -- a stale measured offset, hazard 9 -- or scaled so narrowly that
it is already falling off at rest.

It also prints the floor: what the reward pays when every kernel has gone to
zero. That number is `alive`'s weight, it is the whole of what makes
surviving beat terminating, and under B7's sign convention it did not exist.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys

import mujoco

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("CADEX_REPO", Path.home() / "cadex"))
sys.path.insert(0, str(REPO / "src" / "Mod" / "cadex"))
sys.path.insert(0, str(HERE))

import CadexDynamics as dyn  # noqa: E402
import compare  # noqa: E402

PROJECT = HERE / "mg-legs.cadex"

#: How far a term may sit from its own weight before this fails. Tight,
#: because at the nominal pose every error expression is exactly 0 by
#: construction and anything else is a defect rather than a tolerance.
TOLERANCE = 1.0e-6


def standing_observation(task, model):
    """Every declared channel, at the reset keyframe, as the reward sees it."""

    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY,
                            str(task["episode"]["reset_keyframe"]))
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    observation = {}
    for row in task["observations"]:
        adr, scale = int(row["adr"]), float(row["scale"])
        for index, name in enumerate(row["channels"]):
            observation[str(name)] = float(data.sensordata[adr + index]) * scale
    return observation


def main(argv: list[str]) -> int:
    bundle_path = compare.newest_task_bundle(PROJECT)
    task, model_factory = compare.load_task(bundle_path)
    print(f"==> bundle {bundle_path}")
    observation = standing_observation(task, model_factory())

    rows = list(task["reward"])
    positive = all(float(row["weight"]) >= 0.0 for row in rows)
    print(f"==> {len(rows)} terms, "
          f"{'ALL POSITIVE' if positive else 'MIXED SIGN'}")

    print("==> at the nominal pose, every term must pay its own weight")
    print("    term            kernel    weight      pays")
    total, budget, wrong = 0.0, 0.0, []
    for row in rows:
        weight = float(row["weight"])
        value = dyn.evaluate_reward(str(row["expression"]), observation,
                                    context="standing")
        pays = weight * value
        total += pays
        budget += weight
        flag = ""
        if abs(pays - weight) > TOLERANCE:
            flag = "   <-- NOT AT ITS WEIGHT"
            wrong.append(str(row["label"]))
        print(f"    {str(row['label']):14s} {value:8.5f} {weight:9.3f} "
              f"{pays:9.4f}{flag}")
    print(f"    {'TOTAL':14s} {'':8s} {budget:9.3f} {total:9.4f}")

    print("==> the floor: what the reward pays when every kernel is 0")
    floor = sum(float(row["weight"]) for row in rows
                if str(row["expression"]).strip() == "1")
    print(f"    {floor:.3f} per step, from the constant term(s)")
    if not positive:
        print("    ...but at least one weight is NEGATIVE, so the total is "
              "not bounded below by\n    this and the suicide mode is "
              "available. That is what B8 exists to remove.")
        return 1
    print("    Every other term is a positive kernel bounded in [0, w], so "
          "the per-step\n    reward is bounded in "
          f"[{floor:.1f}, {budget:.1f}] and ENDING AN EPISODE IS ALWAYS "
          "WORSE THAN\n    CONTINUING IT. There is no arithmetic to keep "
          "balanced.")

    if wrong:
        print(f"==> FAIL: {', '.join(wrong)} do not pay their own weight "
              f"standing.")
        print("    Either the error expression is non-zero at the nominal "
              "pose -- a stale\n    measured offset, which is hazard 9 -- or "
              "the scale is so narrow the kernel\n    is already falling off "
              "at rest. Re-run measure.py.")
        return 1
    print("==> every term is at its maximum standing, so each one can only "
          "be reduced by\n    the machine doing the thing that term is "
          "about.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
