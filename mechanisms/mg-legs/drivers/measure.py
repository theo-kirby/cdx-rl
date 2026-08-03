#!/usr/bin/env python
"""What the kernel says this machine weighs, and where it stands.

    pixi run python ~/cadex-legs/measure.py

Phase A of the standing exercise, and it comes FIRST because ADR-077 is what
happens when an actuator is sized from a guess: the hopper's leg was short by
2.2x for merely holding a crouch, and a training run was spent finding that
out. Nothing here learns anything and nothing here is designed -- it reads
the exported MJCF, whose masses and inertias OCCT computed from the solids,
and prints the numbers the rest of the plan is derived from:

  * per-body mass and the machine's total;
  * the standing centre of mass (X0, Y0, Z0) at the reset keyframe, which is
    the baseline every reward term is written against. Read, never taken
    from the drawing -- the hopper's was 6.3 mm out that way;
  * each joint's height above the floor and the mass it carries, which is
    what the holding torque is computed from;
  * what is touching what at frame 0.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parent / "mg-legs.cadex"
GRAVITY = 9.81


def accepted_outputs() -> Path:
    state = json.loads((PROJECT / "script.json").read_text(encoding="utf-8"))
    outputs = PROJECT / state["accepted_attempt"]["staging"] / "outputs"
    if not outputs.is_dir():
        raise SystemExit(f"FAIL: {outputs} does not exist. Run rebuild.py first.")
    return outputs


def model_path() -> Path:
    outputs = accepted_outputs()
    xml = next(iter(sorted(outputs.glob("*-model.xml"))), None)
    if xml is None:
        raise SystemExit(f"FAIL: no MJCF in {outputs}")
    return xml


def names(model, kind):
    return [mujoco.mj_id2name(model, kind, i) or f"<{i}>"
            for i in range(
                {mujoco.mjtObj.mjOBJ_BODY: model.nbody,
                 mujoco.mjtObj.mjOBJ_JOINT: model.njnt,
                 mujoco.mjtObj.mjOBJ_GEOM: model.ngeom,
                 mujoco.mjtObj.mjOBJ_ACTUATOR: model.nu}[kind])]


def main() -> int:
    xml = model_path()
    print(f"==> {xml}")
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    body_names = names(model, mujoco.mjtObj.mjOBJ_BODY)
    ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ground")
    machine = [i for i in range(model.nbody) if i not in (0, ground)]

    print("==> mass, from the solids (kg)")
    total = 0.0
    for i in machine:
        mass = float(model.body_mass[i])
        total += mass
        inertia = model.body_inertia[i]
        print(f"    {body_names[i]:10s} {mass * 1000:7.2f} g   "
              f"principal I {inertia[0]:.3e} {inertia[1]:.3e} "
              f"{inertia[2]:.3e} kg*m2")
    print(f"    {'TOTAL':10s} {total * 1000:7.2f} g  ->  "
          f"{total * GRAVITY:.3f} N of weight")

    # subtree_com[0] is the whole model including the floor, which has no
    # mass we care about; the pelvis is the free root, so its subtree IS the
    # machine.
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    com = np.array(data.subtree_com[pelvis]) * 1000.0
    print("==> the standing pose, at the exported keyframe")
    print(f"    centre of mass   X0 {com[0]:8.3f}  Y0 {com[1]:8.3f}  "
          f"Z0 {com[2]:8.3f} mm")
    for i in machine:
        pos = np.array(data.xipos[i]) * 1000.0
        print(f"    {body_names[i]:10s} com at "
              f"({pos[0]:8.3f},{pos[1]:8.3f},{pos[2]:8.3f}) mm")

    print("==> joints: axis, range, height, and the mass hung below it")
    joint_names = names(model, mujoco.mjtObj.mjOBJ_JOINT)
    for j in range(model.njnt):
        kind = int(model.jnt_type[j])
        body = int(model.jnt_bodyid[j])
        anchor = np.array(data.xanchor[j]) * 1000.0
        axis = np.array(data.xaxis[j])
        # Everything at or below this joint's child body.
        carried, stack = 0.0, [body]
        while stack:
            b = stack.pop()
            carried += float(model.body_mass[b])
            stack.extend(c for c in range(model.nbody)
                         if int(model.body_parentid[c]) == b and c != b)
        if kind == mujoco.mjtJoint.mjJNT_FREE:
            print(f"    {joint_names[j]:14s} free   carries "
                  f"{carried * 1000:6.1f} g")
            continue
        low, high = np.degrees(model.jnt_range[j])
        limited = bool(model.jnt_limited[j])
        print(f"    {joint_names[j]:14s} axis ({axis[0]:+.2f},{axis[1]:+.2f},"
              f"{axis[2]:+.2f})  z {anchor[2]:7.2f} mm  "
              f"range [{low:7.2f},{high:7.2f}] {'' if limited else '(free)'}"
              f"  carries {carried * 1000:6.1f} g")

    print("==> contacts at frame 0")
    geom_names = names(model, mujoco.mjtObj.mjOBJ_GEOM)
    print(f"    ngeom {model.ngeom}, ncon {data.ncon}, nq {model.nq}, "
          f"nv {model.nv}, nu {model.nu}")
    for c in range(data.ncon):
        contact = data.contact[c]
        pos = np.array(contact.pos) * 1000.0
        print(f"    {geom_names[contact.geom1]:22s} "
              f"{geom_names[contact.geom2]:22s} "
              f"at ({pos[0]:7.2f},{pos[1]:7.2f},{pos[2]:7.2f}) mm  "
              f"gap {contact.dist * 1000:+.4f} mm")

    print("==> geoms")
    for g in range(model.ngeom):
        pos = np.array(data.geom_xpos[g]) * 1000.0
        size = np.array(model.geom_size[g]) * 1000.0
        print(f"    {geom_names[g]:24s} at "
              f"({pos[0]:8.2f},{pos[1]:8.2f},{pos[2]:8.2f}) mm  "
              f"half-size ({size[0]:.2f},{size[1]:.2f},{size[2]:.2f})")

    reward_constants(outputs_bundle(), model, data)
    return 0


def outputs_bundle():
    """The task bundle beside the model, which is what the reward is written
    against. Read rather than restated, for the reason `feasibility.py` gives.
    """

    outputs = accepted_outputs()
    bundle = next(iter(sorted(outputs.glob("*-task.json"))), None)
    if bundle is None:
        raise SystemExit(f"FAIL: no task bundle in {outputs}")
    return json.loads(bundle.read_text(encoding="utf-8"))


def reward_constants(task, model, data) -> None:
    """The measured block the task's reward is built out of, ready to paste.

    B6 measured the foot centroid and the two offsets BY HAND off this same
    keyframe, and wrote in the script that they had been. That is one hand
    step between a rebuild and a reward, and hazard 9 is the record of what
    a stale constant costs: the same reward written against an absolute
    channel with a large offset trained WORSE THAN NOT TRAINING. So the
    whole block is printed here instead, read through the task's OWN
    observation channels rather than off the model -- which is the only way
    to be sure the number the reward will see is the number measured.

    Every offset printed is what the corresponding expression evaluates to
    at the standing pose, and every one of them must be subtracted in the
    expression so the term reads exactly 0 there.
    """

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    channel = {}
    for observation in task["observations"]:
        scale = float(observation["scale"])
        adr = int(observation["adr"])
        for index, name in enumerate(observation["channels"]):
            channel[str(name)] = float(data.sensordata[adr + index]) * scale

    com = [channel["com_x"], channel["com_y"], channel["com_z"]]
    foot = [0.5 * (channel["ft_l_x"] + channel["ft_r_x"]),
            0.5 * (channel["ft_l_y"] + channel["ft_r_y"])]
    omega0 = math.sqrt(9810.0 / com[2])

    print("==> the reward's measured constants, off this bundle's own "
          "channels")
    print(f"    Z0 = {com[2]:.3f}")
    print(f"    X0 = {com[0]:.3f}")
    print(f"    Y0 = {com[1]:.3f}")
    print(f"    FX0, FY0 = {foot[0]:.3f}, {foot[1]:.3f}")
    print(f"    OMEGA0 = {omega0:.4f}")
    print(f"    OVER/XI offsets: x {com[0] - foot[0]:+.3f}  "
          f"y {com[1] - foot[1]:+.3f}   (subtract these)")
    print(f"    collapsed floor 0.5 * Z0 = {0.5 * com[2]:.3f} mm")

    posture = sum(abs(value) for name, value in channel.items()
                  if name.endswith("_a"))
    effort = sum(abs(value) for name, value in channel.items()
                 if name.endswith("_tau"))
    speed = abs(channel["cv_x"]) + abs(channel["cv_y"])
    swirl = abs(channel["cam_x"]) + abs(channel["cam_y"])
    print("    and what the SHAPED quantities read standing -- every one of "
          "them\n    must be ~0, because that is what makes each kernel read "
          "its own weight:")
    print(f"    sum|joint angle| {posture:8.4f} deg    "
          f"sum|actuator force| {effort:8.4f} N*mm")
    print(f"    |cv_xy|          {speed:8.4f} mm/s   "
          f"|cam_xy|            {swirl:8.4f} N*mm*s")


if __name__ == "__main__":
    raise SystemExit(main())
