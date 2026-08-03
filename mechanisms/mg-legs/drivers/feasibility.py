#!/usr/bin/env python
"""Can this machine be held up at all, and does it fall when it is not?

    pixi run python ~/cadex-legs/feasibility.py

The gate ADR-075 did not have and ADR-077 paid for. A training run is hours
of GPU time answering "can a policy stand this thing up", and the mechanism
may already have closed that question -- the hopper's leg was under-actuated
by 2.2x for merely HOLDING a crouch, so no policy could have hopped, and the
run that "found a gait" was reading a collapse. Everything below runs in
seconds against the exported MJCF, opened by stock MuJoCo, with no learning
in it.

Six checks. The first is advisory; each of the other five can fail the gate:

  1. **Arithmetic.** Holding torque is about weight x moment arm, per joint,
     printed beside the limit the script actually gives it. Advisory since
     M9 (ADR-086): it multiplies full body weight by a full moment arm,
     which is one leg carrying everything.
  2. **Gravity compensation, exactly.** `mj_inverse` at the reset pose with
     qvel = qacc = 0 is the precise torque each joint needs to hold the
     stance. Over its limit and the machine CANNOT stand -- no reward will
     find a pose that does not exist.
  3. **The capture point against what a step can reach.** The worst shove
     the task declares is an impulse; ``xi = v / sqrt(g/h)`` is how far
     ahead of the feet the machine would have to be caught. Inside the
     support polygon the ankles do it standing; outside it, a foot has to
     move, and a 195 mm leg swung 45 degrees moves one 138 mm. Beyond
     *that* no policy can recover and the falls would be a mechanism limit
     read as a learning failure. Re-specified in M9b (ADR-087) because the
     task changed from "reject in place" to "catch it however you can" --
     the in-place statics are still computed and still printed, they just
     no longer gate.
  4. **Contact sanity.** Unlike the hopper this machine is SUPPOSED to start
     touching the floor, so the check is that the soles are the only things
     touching and that nothing is interpenetrating.
  5. **The drop test.** The zero action must not score a free episode. What
     "zero action" IS depends on the action space, so since B8 this check
     reads the actuators first:

       * **torque action space** (``kind="motor"``): zero action is motors
         off, and it must make the machine FALL from the reset pose. If it
         stands by itself the task is degenerate and the reward measures
         nothing.
       * **position action space** (``kind="position"``): zero action is
         "hold the nominal pose", and it MUST stand at the reset pose --
         that is the whole premise of the change, and a servo that cannot
         hold its own nominal stance is a gain error. Non-degeneracy is then
         measured where it actually lives: zero action is run against the
         REAL task, with its reset variation, its shoves and its
         randomisation, and it must fail a substantial share of episodes.

     Both branches ask one question -- is there anything to learn? -- and
     the second one asks it of the task rather than of the keyframe.
  6. **The hold test.** A PD stabiliser must keep it upright for a full
     episode. If a PD can stand, PPO can. If it cannot, what is wrong is the
     mechanism or the limits, not the reward. It is NOT expected to survive a
     shove -- see check 3. Under a torque action space the PD is written HERE
     and six gain pairs are swept, which is how B8's gains were chosen; under
     a position action space the PD is IN THE MODEL, a runtime sweep would be
     measuring the same servo six times, and what is reported instead is that
     servo's own settle, peak effort and saturation error.

Exits non-zero when any of them fails. Lives beside the project rather than
in `cadex_tests`: the biped is not a test (ADR-075 section 6).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys

import mujoco
import numpy as np

REPO = Path(os.environ.get("CADEX_REPO", Path.home() / "cadex"))
sys.path.insert(0, str(REPO / "src" / "Mod" / "cadex"))

import CadexDynamics as dyn  # noqa: E402

PROJECT = Path(__file__).resolve().parent / "mg-legs.cadex"
GRAVITY = 9.81

# The moment arm each joint works against.
#
# TWO SETS, because the first version of this gate asked the wrong question
# and failed a machine that four other checks said was sound. FULL_ARM_MM is
# what it used to ask: the whole limb length, which is the arm when the leg
# is HORIZONTAL and the machine hangs off one hip. That is a one-legged iron
# cross, not a stance, and a standing biped never loads a hip that way -- so
# on the MG90S-torque build it read 0.84x at the hip and printed DO NOT
# DISPATCH while `mj_inverse` wanted 2.39 N*mm and the hand-written PD stood
# on 4.5. It is kept and printed because it bounds what this robot could do
# if it ever had to hold a leg out, which is a real limit and a real reason
# not to ask this machine to walk yet.
#
# ARM_MM is the STANDING worst case, and it is what gates:
#
#   hip_roll   30.0  the hip half-width -- the machine over one hip, unchanged
#   hip_pitch  50.0  thigh_len * sin(30 deg): the deepest lean/squat a balance
#   knee       47.5  calf_len * sin(30 deg): task is asked to hold
#   ankle      45.5  the sole's own forward reach -- and this one is not a
#                    choice at all. The centre of pressure cannot leave the
#                    foot, so past weight * 45.5 mm the foot ROLLS instead of
#                    pushing and more servo would buy nothing. At 2.581 N
#                    that is 117 N*mm against an MG90S's 216: the footprint,
#                    not the servo, is what limits this robot.
#
# `ankle_roll` is B2's, and its arm is the LATERAL half-width of the sole --
# 20 mm on a 40 mm foot -- because that is what an ankle roll torque acts
# through. It is not 45.5: that number is how far the sole reaches FORWARD,
# which is the pitch axis's business and not this one's.
ARM_MM = {"hip_roll": 30.0, "hip_pitch": 50.0, "knee": 47.5, "ankle": 45.5,
          "ankle_roll": 20.0}

#: How far ahead of the ankle the sole reaches, from ADR-082. This is the
#: whole of the footprint's authority: the centre of pressure cannot leave
#: the sole, so past `weight x this` the foot rolls instead of pushing.
COP_LIMIT_MM = 45.5
FULL_ARM_MM = {"hip_roll": 30.0, "hip_pitch": 100.0, "knee": 95.0,
               "ankle": 24.5, "ankle_roll": 20.0}

#: The support polygon, measured off the sole and toe geoms of the exported
#: model. Asymmetric front to back because the toe reaches forward and the
#: heel does not reach back: 45.5 mm ahead of the ankle, 24.5 mm behind it.
SUPPORT_MM = {"forward": 45.5, "backward": 24.5, "lateral": 50.0}

#: What the lateral direction can ACTUALLY hold, which is not the polygon.
#:
#: 0.65 THROUGH M9c, and the reasoning was: no ankle roll and no hip yaw, so
#: sideways the centre of pressure could not be driven out to a sole's outer
#: edge -- hip_roll and a weight shift were the whole of it, and two thirds
#: of the geometric half-width was the honest guess.
#:
#: B2 PUT THE ANKLE ROLL IN, which is the mechanism change that number was an
#: argument for. An ankle roll servo drives the centre of pressure laterally
#: the same way the ankle pitch drives it fore and aft, so the cap is no
#: longer the missing actuator -- it is the same thing that caps the sagittal
#: direction, which is the sole itself. 0.90 rather than 1.00 because hip yaw
#: is still absent, so a lateral recovery cannot rotate the stance to face
#: the push, and because a 20 deg roll limit is not the 40 deg the pitch has.
#:
#: It is still an ESTIMATE and it is still named here rather than buried in a
#: formula. `capability.py`'s sagittal-versus-lateral split is what will
#: measure it, and if `lat` is still the worst column after B2 then this
#: number was optimistic and the finding is that ankle roll did not buy what
#: it was bought for.
LATERAL_AUTHORITY = 0.90

#: The leg, as two links, from the joint frames of the exported model.
#
# NOT USED FOR THE STEP REACH ANY MORE (B2). What a swung leg actually
# places is the SOLE, and the sole is now 44 mm below the ankle pitch axis
# rather than 24 -- the ankle roll went in between them -- so a reach
# computed from these two link lengths would understate it by that much and
# keep understating it every time the drawing changes. `stepping_reach`
# measures the hip pitch axis's height off the exported model instead. These
# stay because the two link lengths are still what the leg IS.
THIGH_MM = 100.0     # hip_pitch -> knee
CALF_MM = 95.0       # knee -> ankle

#: How far the hip may swing a leg in one recovery step. hip_pitch's range is
#: -60..+90 deg; 45 deg is what a step actually uses and is inside both ends
#: with room for the knee to be doing something at the same time.
STEP_SWING_DEG = 45.0
JOINTS = ["hip_roll_l", "hip_roll_r", "hip_pitch_l", "hip_pitch_r",
          "knee_l", "knee_r", "ankle_l", "ankle_r",
          "ankle_roll_l", "ankle_roll_r"]


def accepted_outputs() -> Path:
    state = json.loads((PROJECT / "script.json").read_text(encoding="utf-8"))
    outputs = PROJECT / state["accepted_attempt"]["staging"] / "outputs"
    if not outputs.is_dir():
        raise SystemExit(f"FAIL: {outputs} does not exist. Run rebuild.py first.")
    return outputs


def family(name: str) -> str:
    return name.rsplit("_", 1)[0] if name[-2:] in ("_l", "_r") else name


def load():
    outputs = accepted_outputs()
    xml = next(iter(sorted(outputs.glob("*-model.xml"))), None)
    if xml is None:
        raise SystemExit(f"FAIL: no MJCF in {outputs}")
    print(f"==> {xml}")
    model = mujoco.MjModel.from_xml_path(str(xml))
    return model, mujoco.MjData(model), xml


def load_task():
    """The task bundle beside the model, which is where M9's forces live.

    Read rather than restated: the gate has to be robust to what the SCRIPT
    declares, and a copy of those numbers here would be a gate that passes
    while the task asks for something else.
    """

    outputs = accepted_outputs()
    bundle = next(iter(sorted(outputs.glob("*-task.json"))), None)
    if bundle is None:
        raise SystemExit(f"FAIL: no task bundle in {outputs}")
    return json.loads(bundle.read_text(encoding="utf-8"))


def load_task_and_model():
    """The bundle, and a FACTORY for the model it names.

    A factory rather than a model, for the reason `compare.load_task`
    records (ADR-103 section 9): `evaluate_episode` multiplies its domain
    randomisation into the `MjModel` IN PLACE, so a model reused across
    episodes compounds every draw it has ever been given.
    """

    task = load_task()
    outputs = accepted_outputs()
    xml = next(iter(sorted(outputs.glob("*-model.xml"))), None)
    if xml is None:
        raise SystemExit(f"FAIL: no MJCF in {outputs}")
    source = xml.read_bytes()
    return task, lambda: dyn.load_model(source)


def actuator_limits(model) -> dict[str, float]:
    """Each joint's torque limit, in N*mm, by joint name."""
    limits = {}
    for a in range(model.nu):
        joint = int(model.actuator_trnid[a, 0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        limits[str(name)] = float(model.actuator_forcerange[a, 1]) * 1000.0
    return limits


def arithmetic(model) -> bool:
    ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ground")
    mass = sum(model.body_mass[i] for i in range(model.nbody)
               if i not in (0, ground))
    weight = mass * GRAVITY
    print(f"    machine {mass * 1000:.2f} g -> {weight:.3f} N")
    limits = actuator_limits(model)
    ok = True
    print("    joint           arm   standing hold    limit   margin"
          "    | leg horizontal")
    for name in JOINTS:
        arm = ARM_MM[family(name)]
        need = weight * arm            # N*mm
        limit = limits[name]
        margin = limit / need
        if margin < 1.0:
            ok = False
        full_arm = FULL_ARM_MM[family(name)]
        full_need = weight * full_arm
        full_margin = limit / full_need
        print(f"    {name:14s} {arm:4.1f} mm {need:8.1f} N*mm "
              f"{limit:7.1f}  {margin:5.2f}x"
              f"{'  UNDER-ACTUATED' if margin < 1.0 else '        '}"
              f"  | {full_need:6.1f} N*mm {full_margin:5.2f}x")
    print("    the right-hand column does not gate: it is the arm when the "
          "leg is\n    horizontal, which a standing task never reaches. See "
          "ARM_MM.")
    print("    NEITHER COLUMN GATES ANY MORE (M9, ADR-086). Both multiply "
          "FULL body\n    weight by a FULL moment arm, which is one leg "
          "carrying everything --\n    a supported pose, not a stance. "
          "Against the 216 N*mm build it read\n    0.84x at the hip and "
          "printed DO NOT DISPATCH while mj_inverse wanted\n    2.39 and a "
          "hand-written PD stood on 4.5. Re-rating to 86 N*mm makes it\n"
          "    red everywhere, on exactly the check ADR-082 already "
          "established is\n    over-conservative. Hazard 14 says the "
          "failure mode to avoid is learning\n    to click past a red gate, "
          "so the answer is to stop printing a red gate\n    nobody should "
          "obey -- not to keep one and ignore it. It stays PRINTED\n"
          "    because it bounds what this robot could do if it ever had to "
          "hold a leg\n    out, which is a real limit and a real reason not "
          "to ask it to walk yet.")
    return ok


def gravity_compensation(model, data) -> bool:
    """The exact torque holding the reset pose takes, from mj_inverse."""
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_inverse(model, data)
    limits = actuator_limits(model)
    ok = True
    print("    joint            needs        limit     used")
    for name in JOINTS:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        dof = int(model.jnt_dofadr[joint])
        need = abs(float(data.qfrc_inverse[dof])) * 1000.0     # N*m -> N*mm
        limit = limits[name]
        share = need / limit
        if share > 1.0:
            ok = False
        print(f"    {name:14s} {need:9.2f} N*mm {limit:8.1f}  {share * 100:6.1f}%"
              f"{'   OVER THE LIMIT' if share > 1.0 else ''}")
    residual = np.abs(np.array(data.qfrc_inverse[:6]))
    print(f"    the free joint's own residual (unactuated, must be carried "
          f"by contact): {residual.max() * 1000:.2f}")
    return ok


def worst_declared_force(task):
    """The largest force the task can put on the machine at one instant.

    Not the sum of every entry at maximum: that was the first version and it
    was wrong the way the arithmetic column is wrong. Summing this task's
    three entries gives 0.78 N, and holding it for three seconds is 2.34
    N*s of impulse against the **0.042 N*s** a 0.35 N shove lasting 0.12 s
    actually delivers -- 56x, which fails a machine the task never asks that
    of. Hazard 14's whole point is that a red gate nobody should obey is
    worse than no gate.

    So the windows are read. A sustained entry is present at every instant.
    A windowed one can be present anywhere in ``[at_low, at_high +
    duration)``, because its start is drawn -- so the worst instant is the
    one covered by the most windows, found by sweeping the boundaries.
    Returns ``(sustained_n, pulse_n, pulse_seconds, labels)``.
    """

    pushes = list(task.get("disturbance") or ())
    sustained = sum(float(entry["newtons_high"]) for entry in pushes
                    if bool(entry["sustained"]))
    windowed = [entry for entry in pushes if not bool(entry["sustained"])]
    if not windowed:
        return sustained, 0.0, 0.0, [str(e["label"]) for e in pushes]

    edges = sorted({float(entry["at_low_s"]) for entry in windowed}
                   | {float(entry["at_high_s"]) + float(entry["duration_s"])
                      for entry in windowed})
    best, best_labels, best_seconds = 0.0, [], 0.0
    for index in range(len(edges) - 1):
        instant = 0.5 * (edges[index] + edges[index + 1])
        live = [entry for entry in windowed
                if float(entry["at_low_s"]) <= instant
                < float(entry["at_high_s"]) + float(entry["duration_s"])]
        total = sum(float(entry["newtons_high"]) for entry in live)
        if total > best:
            best = total
            best_labels = [str(entry["label"]) for entry in live]
            best_seconds = max(float(entry["duration_s"]) for entry in live)
    return sustained, best, best_seconds, best_labels


def machine_mass_kg(model) -> float:
    """Everything but the world and the ground, which is the robot."""

    ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ground")
    return sum(float(model.body_mass[i]) for i in range(model.nbody)
               if i not in (0, ground))


def joint_span_deg(model, name: str) -> tuple[float, float]:
    """One joint's declared range, in degrees, read off the exported model."""

    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    low, high = (float(v) for v in model.jnt_range[joint])
    return math.degrees(low), math.degrees(high)


def usable_swing_deg(model, name: str) -> float:
    """How far this joint can swing a leg in EITHER direction, capped.

    The smaller of the two ends, because a shove's azimuth is drawn over the
    whole circle and the machine does not get to choose which way it has to
    step. Capped at ``STEP_SWING_DEG`` because the limit is not what a
    recovery step uses -- the knee is doing something at the same time.
    """

    low, high = joint_span_deg(model, name)
    return min(STEP_SWING_DEG, min(abs(low), abs(high)))


def stepping_reach(model, data, task) -> bool:
    """Can this mechanism catch the worst shove the task declares -- by
    stepping if it has to?

    The **third** specification of this check, and the reason is different
    from the reasons the first two were replaced. Those two were wrong:
    they measured nothing, and ADR-086 records them.

    1. ``mj_inverse`` at the reset pose with the force applied. Inverse
       dynamics on a floating base solves for the force needed at *every*
       dof including the six unactuated ones, so a horizontal push at the
       pelvis is absorbed by the free joint's own residual and the leg
       torques come back **bit-identical to the undisturbed case** -- 2.39
       N*mm at every hip, worst azimuth 0 degrees for all eight, which is
       the signature of a number that was never computed.
    2. The hand-written PD, pushed. It fell from every direction on a 0.042
       N*s impulse -- but a **joint-space** PD holds joint angles and has no
       base-attitude feedback at all, so it cannot resist the whole machine
       rotating about its ankles however good the mechanism is. That result
       is about the controller, and gating a mechanism on it would fail
       every mechanism.
    3. The statics of an IN-PLACE rejection: a horizontal force ``F`` at
       height ``h`` is a moment ``F*h``, resisted by whichever is smaller of
       the footprint (``W * COP_LIMIT_MM``, because the centre of pressure
       cannot leave the sole) and the two ankle servos. That one was right,
       and it is still computed and still printed below.

    This version replaces it because **the task changed**, which is a
    different kind of reason and belongs in the record as one (ADR-087). M9
    asked "can it reject this in place", sized the shove to fit that answer,
    and got a policy that never moved its feet -- correctly, because nothing
    ever asked it to. M9b deliberately declares a shove the machine CANNOT
    reject in place, so check 3 as written would print 0.39x and DO NOT
    DISPATCH about a run whose entire point is the thing it is failing.

    The question a stepping task asks is the CAPTURE POINT: after an impulse
    the centre of mass is moving at ``v``, and

        xi = v / omega0,    omega0 = sqrt(g / h)

    is how far ahead of the feet it would have to be caught. Inside the
    support polygon, the ankles can do it standing. Outside, a foot has to
    move -- and a leg of ``THIGH_MM + CALF_MM`` swung through ``swing``
    places one ``leg * sin(swing)`` further out. So:

        in place:  capture point vs the support polygon
        stepping:  capture point vs support + leg * sin(swing)
        gate:      worst declared capture point <= steppable reach

    A sustained force is not an impulse and is not treated as one: it is a
    steady lean, which offsets the centre of pressure by ``F*h/W`` and so
    SHRINKS the polygon the capture point has to land in. It is subtracted
    rather than added to the shove.
    """

    pushes = list(task.get("disturbance") or ())
    if not pushes:
        print("    no disturbance declared -- nothing to be robust to.")
        print("    THIS IS NOT A PASS. A task nothing ever disturbs cannot "
              "distinguish\n    balancing from bracing, which is the whole "
              "finding of ADR-083.")
        return False

    sustained, pulse, pulse_s, labels = worst_declared_force(task)
    mass = machine_mass_kg(model)
    weight = mass * GRAVITY

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    height_mm = com_height(model, data)

    omega0 = math.sqrt(GRAVITY * 1000.0 / height_mm)      # rad/s, mm-consistent
    impulse = pulse * pulse_s                             # N*s
    velocity = impulse / mass if mass > 0.0 else 0.0      # m/s
    capture_mm = velocity * 1000.0 / omega0               # mm

    print(f"    machine {mass * 1000:.1f} g -> {weight:.3f} N, centre of "
          f"mass {height_mm:.1f} mm up  ->  omega0 {omega0:.2f} rad/s")
    print(f"    worst instant: {sustained:.3f} N sustained + {pulse:.3f} N "
          f"for {pulse_s:.2f} s ({', '.join(labels) or 'none'})")
    print(f"    the pulse is {impulse:.3f} N*s -> {velocity:.3f} m/s -> "
          f"CAPTURE POINT {capture_mm:.1f} mm")

    # The in-place statics, kept because it is what says whether a step is
    # merely allowed or actually required -- and because it is the ADR-086
    # number, still true, now reported instead of gating.
    limits = actuator_limits(model)
    from_foot = weight * COP_LIMIT_MM
    from_ankles = limits["ankle_l"] + limits["ankle_r"]
    available = min(from_foot, from_ankles)
    binding = "the foot" if from_foot < from_ankles else "the ankle servos"
    needed = (sustained + pulse) * height_mm
    in_place = available / needed if needed > 0.0 else float("inf")
    print(f"    in-place statics: needs {needed:6.1f} N*mm, has "
          f"{available:6.1f} ({binding} binds) -> {in_place:.2f}x"
          f"{'   A STEP IS REQUIRED' if in_place < 1.0 else ''}")

    # A steady lean costs polygon rather than adding to the impulse.
    lean_mm = sustained * height_mm / weight if weight > 0.0 else 0.0
    print(f"    the {sustained:.3f} N sustained lean spends "
          f"{lean_mm:.1f} mm of polygon before the shove arrives")

    sagittal = usable_swing_deg(model, "hip_pitch_l")
    lateral = usable_swing_deg(model, "hip_roll_l")
    # Measured off the export rather than added up from link lengths: what a
    # swung leg places is the SOLE, and the distance from the hip pitch axis
    # to the sole is the whole leg PLUS the ankle stack, which B2 made 20 mm
    # taller. The floor is z = 0, so the axis's world height is that number.
    hip_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                  "hip_pitch_l")
    leg = float(data.xanchor[hip_joint][2]) * 1000.0
    step_sag = leg * math.sin(math.radians(sagittal))
    step_lat = leg * math.sin(math.radians(lateral))
    print(f"    hip pitch axis {leg:.0f} mm above the sole "
          f"(links {THIGH_MM + CALF_MM:.0f} + the ankle stack); "
          f"hip_pitch swings {sagittal:.0f} deg "
          f"-> {step_sag:.0f} mm of step, hip_roll {lateral:.0f} deg "
          f"-> {step_lat:.0f} mm")

    # Lateral support is still not quite the geometric half-width, but it is
    # much closer than it was: B2's ankle roll drives the centre of pressure
    # sideways the way the ankle pitch drives it fore and aft, so what caps
    # this row is now the sole and the 20 deg roll limit rather than a
    # missing actuator. See LATERAL_AUTHORITY.
    support = {
        "forward": SUPPORT_MM["forward"],
        "backward": SUPPORT_MM["backward"],
        "lateral": SUPPORT_MM["lateral"] * LATERAL_AUTHORITY,
    }
    step = {"forward": step_sag, "backward": step_sag, "lateral": step_lat}

    ok = True
    print("    direction    in place   + step   available   capture   margin")
    for name in ("forward", "backward", "lateral"):
        standing = max(0.0, support[name] - lean_mm)
        reach = standing + step[name]
        margin = reach / capture_mm if capture_mm > 0.0 else float("inf")
        if margin < 1.0:
            ok = False
        verdict = ("in place" if capture_mm <= standing
                   else "NEEDS A STEP" if margin >= 1.0
                   else "OUT OF REACH")
        print(f"    {name:11s} {standing:7.1f} mm {step[name]:7.1f} "
              f"{reach:9.1f} mm {capture_mm:8.1f} mm {margin:6.2f}x  "
              f"{verdict}")
    lateral_pair = (limits.get("ankle_roll_l", 0.0)
                    + limits.get("ankle_roll_r", 0.0))
    print(f"    lateral was the weak axis BY MECHANISM through M9c -- no "
          f"ankle roll, no hip\n    yaw -- and `compare.py` measured it: the "
          f"`lat` column was the worst of the\n    three at every shove "
          f"scale. B2 answered the half of that which was\n    answerable. "
          f"The two ankle rolls now put {lateral_pair:.0f} N*mm across a "
          f"{ARM_MM['ankle_roll']:.0f} mm half-sole,\n"
          f"    so this polygon is {LATERAL_AUTHORITY:.2f} of the geometric "
          f"half-width rather than 0.65. Hip yaw\n    is still absent, so a "
          f"twisting recovery remains out of reach and is not in\n    "
          f"scope. If `lat` is STILL the worst column after this, the ankle "
          f"roll did not\n    buy what it was bought for, and that is the "
          f"finding rather than a reason to\n    iterate.")

    if not ok:
        print("    THE SHOVE IS BEYOND WHAT A STEP CAN CATCH. A policy "
              "cannot learn to recover\n    from a push that no foot "
              "placement available to it could catch -- the falls\n    "
              "would be a mechanism limit read as a learning failure, which "
              "is the mistake\n    ADR-075 names. Make the shove smaller.")
        return False

    # Evidence, not a gate. The undisturbed PD stands this machine on 4.5
    # N*mm (check 6); pushed, it falls from every direction -- because a
    # JOINT-SPACE PD has no base-attitude feedback and cannot resist the
    # whole machine rotating about its ankles, let alone pick a foot up.
    # That is a fact about the controller, not the mechanism, and it is
    # exactly the thing the policy has to learn that the PD does not have.
    print("    (a joint-space PD cannot do this and is not expected to: it "
          "holds joint\n    angles, has no base-attitude feedback, and "
          "would never lift a foot. That gap\n    is what the policy is "
          "being asked to learn, and it is why check 6 passing is\n    not "
          "this passing.)")
    return True


def contacts(model, data) -> bool:
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    ok = True
    print(f"    ngeom {model.ngeom}, ncon {data.ncon}")
    for c in range(data.ncon):
        contact = data.contact[c]
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
                 for g in (contact.geom1, contact.geom2)]
        gap = float(contact.dist) * 1000.0
        sole = all("ground" in n or "foot" in n or "toe" in n for n in names)
        if not sole or gap < -1.0e-6:
            ok = False
        print(f"    {names[0]:22s} {names[1]:22s} gap {gap:+.4f} mm"
              f"{'' if sole else '   NOT A SOLE'}"
              f"{'   INTERPENETRATING' if gap < -1.0e-6 else ''}")
    # Height of every sole above the floor, which is the check that would
    # have caught both of the hopper's collision bugs.
    for name in ("foot_l/collision0", "toe_l/collision0",
                 "foot_r/collision0", "toe_r/collision0"):
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        bottom = (float(data.geom_xpos[g][2]) - float(model.geom_size[g][2])) * 1000.0
        if bottom < -1.0e-6:
            ok = False
        print(f"    {name:22s} bottom at z {bottom:+.4f} mm"
              f"{'   BELOW THE FLOOR' if bottom < -1.0e-6 else ''}")
    return ok


def upright(model, data) -> float:
    """1 - 2*(qx^2 + qy^2) off the free joint, the task's own signal."""
    return 1.0 - 2.0 * (float(data.qpos[4]) ** 2 + float(data.qpos[5]) ** 2)


def com_height(model, data) -> float:
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    return float(data.subtree_com[pelvis][2]) * 1000.0


def run(model, data, controller, seconds: float):
    """Integrate, reporting when it first tips past 45 degrees."""
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    z0 = com_height(model, data)
    step = float(model.opt.timestep)
    fell_at = None
    peak = np.zeros(model.nu)
    t = 0.0
    while t < seconds:
        controller(model, data, t)
        mujoco.mj_step(model, data)
        t += step
        peak = np.maximum(peak, np.abs(np.array(data.actuator_force)))
        if fell_at is None and (upright(model, data) < 0.7
                                or com_height(model, data) < 0.7 * z0):
            fell_at = t
    return fell_at, upright(model, data), com_height(model, data), z0, peak


def servo_controlled(model) -> bool:
    """Is the action space POSITION rather than torque?

    Read off the compiled model rather than off the script, because the
    script is not what the trainer will be handed. A MuJoCo position servo
    is the affine bias -- `biasprm = (0, -kp, -kd)` against `gainprm = kp` --
    and a plain motor has no bias at all.
    """

    return all(int(model.actuator_biastype[a]) == int(mujoco.mjtBias.mjBIAS_AFFINE)
               for a in range(model.nu))


def servo_gains(model) -> tuple[float, float]:
    """(kp, kd) in N*m/rad and N*m*s/rad, off the compiled model."""

    return (float(model.actuator_gainprm[0, 0]),
            -float(model.actuator_biasprm[0, 2]))


def drop(model, data) -> bool:
    def zero(model, data, t):
        data.ctrl[:] = 0.0

    fell, up, z, z0, _peak = run(model, data, zero, 3.0)
    if not servo_controlled(model):
        print(f"    from the reset pose, zero torque for 3 s")
        print(f"    upright {up:+.3f}, com_z {z:.1f} mm of {z0:.1f}")
        if fell is None:
            print("    IT STANDS BY ITSELF. The task is degenerate: a policy "
                  "that emits nothing scores a perfect episode.")
            return False
        print(f"    falls at {fell:.3f} s  <-- correct: there is nothing to "
              f"hold it up but the policy")
        return True

    # A POSITION ACTION SPACE INVERTS THIS CHECK AT THE KEYFRAME, and the
    # inversion is the point of B8 rather than a concession to it. Zero
    # action is the midpoint of every joint's range, the ranges are
    # symmetric about the nominal pose, so zero action means HOLD THE
    # CROUCH -- and the untrained policy stands. That is what the literature
    # buys with this action space and it is why nine runs of searching for
    # gravity compensation stop here.
    print(f"    the action space is POSITION: zero action is 'hold the "
          f"nominal pose'")
    print(f"    from the reset pose, zero action for 3 s")
    print(f"    upright {up:+.3f}, com_z {z:.1f} mm of {z0:.1f}")
    if fell is not None:
        print(f"    IT FALLS, at {fell:.3f} s. A servo that cannot hold its "
              f"own nominal stance\n    is a gain error or a pose error, and "
              f"nothing downstream of it means anything.")
        return False
    print("    it stands  <-- correct, and it is the whole of B8: the "
          "untrained policy\n    already holds the stance, so what is left "
          "to learn is the DEVIATION.")
    return True


def degenerate(task, model_factory, seeds=range(12)) -> bool:
    """Does the zero action score a free episode ON THE REAL TASK?

    The other half of check 5, and under a position action space it is the
    half that carries it. "Zero action stands at the keyframe" is exactly
    what B8 wanted, so the degeneracy question moves to where it now lives:
    the task also has a reset variation that tilts, lifts and throws the
    machine, two shoves and a wind, and if a policy that emits nothing
    survives all of THAT, then the run has nothing to teach and the reward
    is measuring luck.

    Driven through the engine's own episode machinery rather than a
    hand-rolled loop, so what is measured is what the trainer will see:
    the same reset draw, the same disturbance schedule, the same
    randomisation.
    """

    zero = [0.0] * len(task["actions"])
    survived, endings, lengths = 0, {}, []
    for seed in seeds:
        episode = dyn.evaluate_episode(model_factory(), task,
                                       actions=lambda _step, _obs: zero,
                                       seed=seed)
        lengths.append(len(episode["steps"]))
        if episode["truncated"]:
            survived += 1
            ending = "survived"
        else:
            ending = str(episode["termination"])
        endings[ending] = endings.get(ending, 0) + 1
    count = len(list(seeds))
    print(f"    zero action on the DECLARED task, {count} seeds: "
          f"survived {survived}/{count}, mean "
          f"{sum(lengths) / max(1, len(lengths)):.1f} of "
          f"{task['episode']['max_steps']} steps")
    print(f"    {dict(sorted(endings.items()))}")
    if survived == count:
        print("    THE TASK IS DEGENERATE: doing nothing survives every "
              "episode, so there is\n    nothing for a reward to select "
              "for. Open the disturbance or the reset.")
        return False
    print(f"    {count - survived} of {count} episodes are lost by doing "
          f"nothing  <-- there is a task here")
    return True


def pd_hold(model, data, gain: float, damp: float) -> tuple[bool, float, np.ndarray]:
    """A hand-written PD to the reset pose. If this stands, PPO can.

    THE GAINS ARE IN SI, and their scale is the whole subtlety: `data.ctrl`
    on a motor is newton-metres, the joint errors are radians, and these
    links carry about 1e-4 kg*m2 about their own joints. A gain of 10 N*m/rad
    -- which would be modest on a robot arm -- is 100x what this machine
    needs, and the FIRST sweep of this gate ran 2 to 40 and reported that a
    biped which stands perfectly cannot be held up. The binding constraint is
    the damping: explicit damping is stable only while kd * h / I is below
    about one, so with h = 2 ms and I = 1e-4, kd above ~0.05 diverges on its
    own. Measured: kd = 0.05 launches the machine to a 750 N*mm saturation in
    ten milliseconds, kd = 0.02 holds it at 3 N*mm."""
    mujoco.mj_resetDataKeyframe(model, data, 0)
    target = np.array([float(data.qpos[int(model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])])
        for name in JOINTS])
    address = [(int(model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]),
        int(model.jnt_dofadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]))
        for name in JOINTS]
    order = [str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT,
                                   int(model.actuator_trnid[a, 0])))
             for a in range(model.nu)]
    index = [JOINTS.index(name) for name in order]

    def control(model, data, t):
        for a in range(model.nu):
            j = index[a]
            qadr, vadr = address[j]
            error = target[j] - float(data.qpos[qadr])
            data.ctrl[a] = gain * error - damp * float(data.qvel[vadr])

    fell, up, z, z0, peak = run(model, data, control, 6.0)
    # Standing, not merely upright: a machine that has slumped 40 mm and is
    # still nominally the right way up has not held the stance.
    held = fell is None and up > 0.99 and abs(z - z0) < 5.0
    return held, up, z - z0, peak


def hold_servo(model, data) -> tuple[bool, float, float, np.ndarray]:
    """The model's OWN servo, holding the nominal pose for a whole episode.

    Under a position action space `data.ctrl` is a setpoint in radians and
    the loop is inside the solver, so the six-pair sweep below is not a
    controller sweep any more -- it would run the same servo six times and
    print six identical rows. The measurement that still means something is
    what that one servo does over a full episode, so that is what this is.
    """

    def nominal(model, data, t):
        data.ctrl[:] = 0.0

    fell, up, z, z0, peak = run(model, data, nominal, 6.0)
    held = fell is None and up > 0.99 and abs(z - z0) < 5.0
    return held, up, z - z0, peak


def main() -> int:
    model, data, _xml = load()
    verdicts = {}

    print("==> 1. arithmetic: weight x moment arm against the limit "
          "(ADVISORY)")
    advisory = arithmetic(model)

    print("==> 2. gravity compensation at the reset pose (mj_inverse)")
    verdicts["gravity"] = gravity_compensation(model, data)

    print("==> 3. can the mechanism CATCH the worst declared shove -- "
          "stepping if it must?")
    verdicts["disturbed"] = stepping_reach(model, data, load_task())

    print("==> 4. contact sanity at frame 0")
    verdicts["contacts"] = contacts(model, data)

    servo = servo_controlled(model)
    print("==> 5. drop test: the zero action must not score a free episode")
    verdicts["drop"] = drop(model, data)
    if servo and verdicts["drop"]:
        # Under a position action space "it stands at the keyframe" is the
        # ANSWER WANTED, so the degeneracy question has to be asked of the
        # task instead. This is the half that can still fail the gate.
        task, model_factory = load_task_and_model()
        verdicts["drop"] = degenerate(task, model_factory)

    limits = np.array([float(model.actuator_forcerange[a, 1])
                       for a in range(model.nu)])
    if servo:
        gain, damp = servo_gains(model)
        print("==> 6. hold test: the MODEL's servo must hold the stance for a "
              "whole episode")
        print(f"    the PD is in the model: kp {gain:.3f} N*m/rad, "
              f"kd {damp:.4f} N*m*s/rad")
        ok, up, drop_mm, peak = hold_servo(model, data)
        saturated = bool(np.any(peak >= limits - 1.0e-9))
        print("    upright   settle    peak effort   of limit")
        print(f"    {up:+7.3f}  {drop_mm:+7.2f} mm {peak.max() * 1000:8.1f} "
              f"N*mm {limits.max() * 1000 / max(peak.max() * 1000, 1e-9):7.2f}x"
              f"{'  SATURATED' if saturated else ''}"
              f"{'  <-- stands' if ok else '  <-- DOES NOT STAND'}")
        # The number that decides how much of the action range is real. A
        # servo can only ask for `torque_limit` N*mm, so past an error of
        # limit/kp it is saturated and every further degree of setpoint is
        # the same command. B8 took the softest standing gains for exactly
        # this reason -- see the servo block in script.py.
        kp_deg = gain * 1000.0 / math.degrees(1.0)
        print(f"    saturates at {limits.max() * 1000.0 / kp_deg:.1f} deg of "
              f"error ({limits.max() * 1000.0:.0f} N*mm / "
              f"{kp_deg:.3f} N*mm/deg), so that much\n    of each joint's "
              f"range is proportional and the rest is a hard push.")
        verdicts["hold"] = ok
    else:
        print("==> 6. hold test: a PD must keep it upright for a whole episode")
        held = False
        print("    kp N*m/rad  kd N*m*s/rad   upright   settle    peak effort")
        for gain, damp in ((0.1, 0.005), (0.3, 0.01), (1.0, 0.02), (1.0, 0.005),
                           (2.0, 0.02), (2.0, 0.05)):
            ok, up, drop_mm, peak = pd_hold(model, data, gain, damp)
            saturated = bool(np.any(peak >= limits - 1.0e-9))
            print(f"    {gain:9.2f} {damp:12.3f}   {up:+7.3f}  {drop_mm:+7.2f} mm "
                  f"{peak.max() * 1000:8.1f} N*mm "
                  f"{'SATURATED' if saturated else '          '}"
                  f"{'  <-- stands' if ok else ''}")
            held = held or ok
        verdicts["hold"] = held

    print("==> verdict")
    print(f"    {'arithmetic':12s} {'pass' if advisory else 'red'}"
          f"   (advisory, does not gate -- see the note under check 1)")
    for name in ("gravity", "disturbed", "contacts", "drop", "hold"):
        print(f"    {name:12s} {'pass' if verdicts[name] else 'FAIL'}")
    if all(verdicts.values()):
        if servo:
            print("==> the mechanism can be held up, its servos hold the "
                  "nominal stance, and the\n    declared task still beats a "
                  "policy that does nothing. Training is worth "
                  "dispatching.")
        else:
            print("==> the mechanism can be held up, falls when it is not, and "
                  "stands under a hand-written PD. Training is worth "
                  "dispatching.")
        return 0
    print("==> DO NOT DISPATCH A TRAINING RUN. A policy cannot learn a "
          "balance the mechanism cannot perform.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
