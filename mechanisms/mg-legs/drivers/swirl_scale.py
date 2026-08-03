#!/usr/bin/env python
"""What are the SHAPED quantities worth on this machine? -- the reward's scales.

    CADEX_REPO=~/cadex pixi run python ~/cdx-mjc/swirl_scale.py

Named for the one quantity it was written to measure and kept for the other
three, because they all need the same answer from the same sweep.

B7 added a reward term on the machine's own angular momentum about its own
centre of mass (ADR-116), and B8 turned every shaping term into a positive
bounded kernel:

    swirl   B7  -0.3 * tanh((abs(cam_x) + abs(cam_y)) / K)
            B8  +0.3 * exp(-(((abs(cam_x) + abs(cam_y)) / K)^2))

The sign convention changed; WHAT K MEANS DID NOT. It is the magnitude the
term is ABOUT -- under tanh the value where it reaches 0.762, under the
Gaussian the value where it has fallen to exp(-1) = 0.368 of its weight --
and in both cases half the samples above it and half below is what makes the
term a gradient rather than a constant. Pick it too small and the term is
saturated the instant anything moves, which is what B6's single `capture` at
40 mm turned out to be at the new shove band: a saturated term is not a
strong term, it is a constant. Pick it too large and it never says anything.
So it is MEASURED, off the accepted bundle, before its weight is fixed --
the same rule `measure.py` exists for.

FOUR QUANTITIES SINCE B8, all summed the same way the reward sums them:

    swirl     abs(cam_x) + abs(cam_y)          N*mm*s
    arrest    abs(cv_x)  + abs(cv_y)           mm/s
    posture   sum over ten joints of abs(angle)   deg
    effort    sum over ten actuators of abs(force) N*mm

`posture` and `effort` were carried through six runs as WEIGHTS on unmeasured
linear costs, where a wrong scale is a wrong price and nothing more. As
kernel widths they decide where the term has a gradient at all, so they are
measured for the first time here.

Three numbers for each:

1. **At the reset keyframe**, which must be ~0. Hazard 9: a term that is
   non-zero at the standing pose charges rent for standing still, and this
   project has already measured that as the difference between training and
   training WORSE than not training. For a positive kernel the same defect
   reads the other way round -- a term that is not at its FULL WEIGHT when
   the machine is standing is mis-scaled -- and it is the same check.
   Momentum and velocity at rest are zero by definition; joint angles are
   zero because the assembly is drawn at the nominal pose; actuator force is
   zero because the servos are at their setpoint. All four are checks that
   the channels are wired to what they say.

2. **Under the declared shove**, swept across the band the task declares, at
   eight azimuths, over the 1.5 s after the push lands. This is the
   distribution the scales come from.

3. **Per newton**, so the numbers can be re-derived if the band moves again.

**The controller holds the stance, and that is a stated limitation.** No
policy exists for this task, and a stance-holding controller has no
base-attitude feedback and will never lift a foot. What it does is hold the
stance while the shove lands, which is what this measurement needs: the
state in the moment AFTER the machine is hit, which is when these terms have
to have a gradient. It under-samples the end of a real recovery, where a foot
has come down and the momentum is being shed. Read the numbers as the top of
the range rather than as the whole of it.

**WHICH controller depends on the action space, and since B8 this reads the
model rather than assuming.** Under a torque action space it is
`feasibility.py` check 6's hand-written PD, in SI, at the solver step. Under
a position action space the PD IS THE MODEL -- `data.ctrl` is a setpoint in
radians, not a torque -- so holding the stance is `ctrl = 0` and writing
torques into it would command millirad setpoints and measure nothing. That
is the same mistake in a new place as ADR-107's, and it is one line.

**The loop is at the solver step, not the control step, and that is not a
detail.** `pd_hold`'s own docstring works out why: explicit damping is
stable only while kd*h/I is under about one, and at h = 2 ms with kd = 0.02
this machine sits just inside that. Driven through the task's control
interval instead -- zero-order hold across five or ten solver steps -- the
same gains diverge and the machine tips in 0.7 s with nothing having pushed
it. Measured here before this file was written the other way.

The forces, the duration, the body and the point of application all come out
of the accepted bundle, and the force goes in exactly where the engine puts
it: `xfrc_applied` on the declared body, at its centre of mass, in the world
frame.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import statistics
import sys

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("CADEX_REPO", Path.home() / "cadex"))
sys.path.insert(0, str(HERE))

import compare  # noqa: E402

PROJECT = HERE / "mg-legs.cadex"

#: The gains `feasibility.py` check 6 reports as standing with 3.3 N*mm of
#: peak effort. SI, because `data.ctrl` on a motor is newton-metres and the
#: joint errors are radians -- see `pd_hold`'s docstring for why 10 N*m/rad
#: would be a hundred times what these links need.
KP = 1.0
KD = 0.02

#: When the push lands and how long after it counts as the recovery. 1.0 s
#: leaves the PD time to settle from the keyframe; 1.5 s is the upper end of
#: the 1-2 s a stumble-and-recover takes on this machine and is inside the
#: 2.0 s of room the B7 windows leave between shoves.
SHOVE_AT_S = 1.0
RECOVERY_S = 1.5

#: Eight compass points. The first shove's declared arc is 210-330 deg, but
#: `shove2` draws the whole circle, so the sweep does too.
AZIMUTHS = tuple(range(0, 360, 45))

#: Where in the band to sample. The ends are the declared ones; the middle
#: three are there so the per-newton column has a shape rather than a slope
#: through two points.
FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


#: The four sums the reward shapes, each as the list of channel names it adds
#: the absolute value of. Written as NAMES and resolved against the bundle's
#: own `adr`/`scale` below, because that pair is what a trainer slices
#: `sensordata` with -- reading it any other way would measure a different
#: thing than the one training uses.
QUANTITIES = (
    ("swirl", "N*mm*s", ("cam_x", "cam_y")),
    ("arrest", "mm/s", ("cv_x", "cv_y")),
    ("posture", "deg", "*_a"),
    ("effort", "N*mm", "*_tau"),
)


def channel_slices(task):
    """(name -> (adr, scale)) for every channel the bundle declares."""

    slices = {}
    for row in task["observations"]:
        adr, scale = int(row["adr"]), float(row["scale"])
        for index, name in enumerate(row["channels"]):
            slices[str(name)] = (adr + index, scale)
    return slices


def quantity_slices(task):
    """Each shaped sum, as the list of (adr, scale) it adds abs() over."""

    slices = channel_slices(task)
    resolved = []
    for label, unit, spec in QUANTITIES:
        if isinstance(spec, str):
            names = sorted(name for name in slices
                           if name.endswith(spec.lstrip("*")))
        else:
            names = list(spec)
        missing = [name for name in names if name not in slices]
        if missing or not names:
            raise SystemExit(
                f"FAIL: this bundle declares no {missing or spec!r} channel. "
                "Rebuild the project first."
            )
        resolved.append((label, unit, [slices[name] for name in names]))
    return resolved


def read(data, columns) -> float:
    """One shaped sum, off `sensordata`, exactly as the reward reads it."""

    return sum(abs(float(data.sensordata[adr]) * scale)
               for adr, scale in columns)


def shove_entry(task):
    """The first non-sustained disturbance, as (body_id, band, duration)."""

    for entry in task["disturbance"]:
        if not bool(entry["sustained"]):
            return (int(entry["body_id"]),
                    (float(entry["newtons_low"]), float(entry["newtons_high"])),
                    float(entry["duration_s"]))
    raise SystemExit("FAIL: the bundle declares no impulsive disturbance")


def servo_controlled(model) -> bool:
    """Is `data.ctrl` a SETPOINT rather than a torque?

    Read off the compiled model, the same way `feasibility.py` reads it: a
    MuJoCo position servo is the affine bias, `biasprm = (0, -kp, -kd)`, and
    a plain motor has no bias at all.
    """

    return all(int(model.actuator_biastype[a]) == int(mujoco.mjtBias.mjBIAS_AFFINE)
               for a in range(model.nu))


def pd_controller(model, data):
    """Hold every actuated joint at the keyframe pose, whatever `ctrl` means.

    Under a POSITION action space the loop is already inside the solver and
    the keyframe pose is the setpoint zero, so holding the stance is writing
    nothing -- and writing a torque into it instead would command a setpoint
    of a few thousandths of a radian and measure a machine nobody built.
    """

    if servo_controlled(model):
        def hold():
            data.ctrl[:] = 0.0

        return hold

    address, target = [], []
    for actuator in range(model.nu):
        joint = int(model.actuator_trnid[actuator, 0])
        qadr = int(model.jnt_qposadr[joint])
        vadr = int(model.jnt_dofadr[joint])
        address.append((qadr, vadr))
        target.append(float(data.qpos[qadr]))

    def control():
        for actuator, (qadr, vadr) in enumerate(address):
            data.ctrl[actuator] = (
                KP * (target[actuator] - float(data.qpos[qadr]))
                - KD * float(data.qvel[vadr])
            )

    return control


def episode(model, task, quantities, *, body, newtons, azimuth_deg, duration):
    """One push onto a held stance. Returns a trace per shaped quantity."""

    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY,
                            str(task["episode"]["reset_keyframe"]))
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    control = pd_controller(model, data)

    radians = math.radians(azimuth_deg)
    force = (newtons * math.cos(radians), newtons * math.sin(radians), 0.0)
    step = float(model.opt.timestep)

    traces = {label: [] for label, _unit, _columns in quantities}
    tipped_at = None
    t = 0.0
    while t < SHOVE_AT_S + RECOVERY_S:
        data.xfrc_applied[:] = 0.0
        if SHOVE_AT_S <= t < SHOVE_AT_S + duration:
            for axis in range(3):
                data.xfrc_applied[body, axis] = force[axis]
        control()
        mujoco.mj_step(model, data)
        t += step
        if t >= SHOVE_AT_S:
            for label, _unit, columns in quantities:
                traces[label].append(read(data, columns))
        # The task's own `tipped`: qx^2 + qy^2 over 0.15 on the free root.
        quat = data.qpos[3:7]
        if tipped_at is None and float(quat[1]) ** 2 + float(quat[2]) ** 2 > 0.15:
            tipped_at = t
    return traces, tipped_at


def percentiles(values, fractions=(0.5, 0.75, 0.9, 1.0)):
    ordered = sorted(values)
    if not ordered:
        return [float("nan")] * len(fractions)
    return [ordered[min(len(ordered) - 1, int(round(f * (len(ordered) - 1))))]
            for f in fractions]


def main(argv: list[str]) -> int:
    bundle_path = compare.newest_task_bundle(PROJECT)
    task, model_factory = compare.load_task(bundle_path)
    print(f"==> bundle {bundle_path}")
    names = compare.channel_names(task)
    quantities = quantity_slices(task)
    body, band, duration = shove_entry(task)
    model = model_factory()
    print(f"    {len(names)} channels; "
          f"{'POSITION' if servo_controlled(model) else 'TORQUE'} action "
          f"space, so the stance is held by "
          f"{'the model' if servo_controlled(model) else 'a written PD'}")
    print(f"    declared shove {band[0]:g}-{band[1]:g} N for {duration:g} s "
          f"on body {body}")
    for label, unit, columns in quantities:
        print(f"    {label:8s} sums {len(columns):2d} channel(s) in {unit}")

    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY,
                            str(task["episode"]["reset_keyframe"]))
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    print("==> 1. at the reset keyframe, every one of which must be ~0")
    for label, unit, columns in quantities:
        at_rest = read(data, columns)
        print(f"    {label:8s} {at_rest:12.6f} {unit}")
        if at_rest > 1.0e-6:
            print(f"    FAIL: `{label}` is not ~0 standing, so its kernel is "
                  f"not at full weight\n    when the machine is doing "
                  f"nothing wrong (hazard 9).")
            return 1
    print("    all ~0, so every kernel reads its own weight at the standing "
          "pose.")

    print(f"==> 2. over the {RECOVERY_S:g} s after the push, "
          f"{len(AZIMUTHS)} azimuths x {len(FRACTIONS)} force levels")
    per_newton = []
    for fraction in FRACTIONS:
        newtons = band[0] + fraction * (band[1] - band[0])
        samples = {label: [] for label, _unit, _columns in quantities}
        tipped = 0
        for azimuth in AZIMUTHS:
            traces, tipped_at = episode(
                model_factory(), task, quantities,
                body=body, newtons=newtons, azimuth_deg=azimuth,
                duration=duration,
            )
            for label in samples:
                samples[label].extend(traces[label])
            tipped += tipped_at is not None
        per_newton.append({"newtons": newtons, "tipped": tipped,
                           "samples": samples})
    for label, unit, _columns in quantities:
        print(f"    {label} ({unit})")
        print("    newtons    median      p75      p90      max   tipped")
        for row in per_newton:
            p50, p75, p90, p100 = percentiles(row["samples"][label])
            print(f"    {row['newtons']:6.2f} N {p50:9.2f} {p75:8.2f} "
                  f"{p90:8.2f} {p100:8.2f}   {row['tipped']}/{len(AZIMUTHS)}")

    print("==> 3. per newton, so this is re-derivable if the band moves")
    for label, unit, _columns in quantities:
        slopes = [percentiles(row["samples"][label], (0.5,))[0] / row["newtons"]
                  for row in per_newton if row["newtons"]]
        print(f"    {label:8s} median of the medians "
              f"{statistics.median(slopes):8.2f} {unit} per newton")

    # THE SCALES COME FROM THE RECOVERY REGIME, not from the whole band, and
    # the criterion is measured rather than picked: the force levels at which
    # the held stance actually goes over. Below that the machine is standing
    # and there is nothing for these terms to shape -- so folding it in would
    # drag every scale down and saturate the terms exactly where a step is
    # the only answer. That is B6's single `capture` mistake, generalised.
    recovering = [row for row in per_newton
                  if row["tipped"] > len(AZIMUTHS) // 2]
    if not recovering:
        print("==> the held stance never went over at any declared force. "
              "Either the band is")
        print("    too small for these terms to be about anything, or the "
              "controller is doing")
        print("    more than it should. Do not pick a scale from this run.")
        return 1
    print("==> the scales")
    print(f"    the recovery regime is {recovering[0]['newtons']:.2f} N and "
          f"up -- where the held stance\n    goes over -- and each scale is "
          f"its median there.")
    for label, unit, _columns in quantities:
        everything = [value for row in recovering
                      for value in row["samples"][label]]
        whole = [value for row in per_newton
                 for value in row["samples"][label]]
        print(f"    {label:8s} {percentiles(everything, (0.5,))[0]:9.2f} "
              f"{unit:8s} over {len(everything)} samples   "
              f"(whole band would say "
              f"{percentiles(whole, (0.5,))[0]:.2f})")
    print("    Half the recovery samples sit below each of those and half "
          "above, so every\n    kernel has a gradient where a recovery "
          "actually happens rather than a\n    constant. The whole-band "
          "figures in brackets are numbers about standing\n    still, which "
          "is what they would be measuring if the absorbable end of the\n"
          "    band were folded in.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
