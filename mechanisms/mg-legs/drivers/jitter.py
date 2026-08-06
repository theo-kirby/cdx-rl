#!/usr/bin/env python
"""How STILL is it, and how much does the sole slide? Experiment 006's instrument.

    /home/theo/cadex-train-venv/bin/python \\
        mechanisms/mg-legs/drivers/jitter.py \\
        --policy <p.cxpolicy> --task tasks/stand-b8-clamp25/stand-task.json \\
        [--seeds 12] [--disturbance declared|none] [--open-loop off|replay|frozen]

**Run under the trainer interpreter, never `uv run`** — the same rule as
`hazard15.py`, and for the same reason: this imports the engine, which imports
mujoco, which cdx-rl's own venv deliberately does not pin.

## Why this file exists

Watching the trained policies play, two things are wrong with *how* the machine
stands, and no published number in this repository was built to see either:
the legs jitter continuously, and a push is answered with a long sequence of
tiny foot corrections rather than one decisive step. `harness steps` measures
*that* a foot moved and *how far*; nothing measures how much the machine is
fidgeting while it does not.

Of the nine reward kernels the bundle carries, `arrest` (CoM velocity) and
`swirl` (centroidal angular momentum) **both cancel for antiphase leg
jitter** — which is what jitter mostly is. `posture` is on joint *angles* and
`effort` on torque. Nothing in the reward sees it, which is a reason to expect
it rather than a reason to be surprised by it.

## The four things it reports, and the units rule

**Every emitted key carries its unit in its own name.** That is not a style
preference: `harness/_episodes._torque_columns` derives `peak_torque_nmm` from
`step["action"]`, which under a position action space is a joint angle in
*degrees*, and it labels it N·mm. It has been silently wrong on every table
built from it. So there is no key here called `jitter`.

* **Command jitter** — mean and p95 ``|Δaction|`` per control step per joint,
  in **degrees**, plus the sign-reversal rate per second. Both are needed and
  neither means anything alone: a policy sweeping a joint quickly has a large
  Δ and near-zero reversals, and a chattering one has both. At this bundle's
  50 Hz the reversal rate saturates at **50 /s** — one direction change per
  control step, since a full oscillation carries two.
* **Joint jitter** — mean and RMS ``|q̇|`` per joint in **deg/s** off
  ``data.qvel``, and ``Σ|q̇|`` over the ten actuated joints. This is exactly
  the quantity a `quiet` reward kernel would sum, so this driver's job is to
  **size that kernel's σ**, not to guess it.
* **Sole slip** — tangential speed of each foot while in contact, and
  integrated slip in mm per contact-second.
* **Centre of pressure** inside the sole, read against the box derived from
  the model's own geoms. The cheap way to ask the foot question without
  building a foot.

## Two caveats that are stated rather than hidden

**The sampling rate.** ``evaluate_episode``'s ``sample`` hook fires at the
**control** rate — 50 Hz — not the solver's 500 Hz. Slip inside one control
interval is integrated as if it were constant over that interval. That biases
the integral; it does not invent it. Every other number in this experiment is
quoted at the same resolution.

**The frames the support polygon is quoted in.** ``harness/profiles/mg-legs.json``
declares ``forward 45.5 / backward 24.5 / lateral 50.0``, and those three
numbers are **not in the same frame**. Derived from the compiled model:

* 45.5 / 24.5 are the sole's reach measured from the **ankle bracket** origin
  — the foot body sits 12.25 mm forward of it, so in the *foot's own* frame
  the same box is +33.25 / −36.75;
* ±50 is the **double-support** half-width — the stance spans two feet 60 mm
  apart — while one sole is only **±20 mm** wide.

A driver that compared a per-foot CoP's lateral excursion against 50 would
report 40 % utilisation where the true figure is 100 %. So the per-foot box is
**derived from the model** here and the profile's numbers are printed beside
it, labelled, rather than used.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
MODULE_DIR = os.environ.get("CADEX_ENGINE_DEV_TREE", "/home/theo/cadex")

#: Seconds to let the reset drop be absorbed before anything is believed.
#:
#: Same constant and same discipline as `hazard15.py`. The reset variation
#: lifts the machine and drops it, and absorbing 42 mm saturates every motor
#: and slams both soles into the floor — a slip integral over the whole
#: episode measures the landing, not the standing. Whole-episode figures are
#: printed **beside** settled ones, never instead of them, so that the settled
#: window cannot be mistaken for a cherry-picked one.
SETTLE_SECONDS = 1.0

#: The bundle carries this literal in every ``*_v`` row's ``scale``. It is read
#: from the bundle rather than typed, so this driver and the reward kernel it
#: sizes cannot disagree; this value is the fallback when a row is absent.
DEG_PER_RAD = 57.29577951308232


# ---------------------------------------------------------------------------
# The pure half. No mujoco, no engine, no filesystem — so it is importable and
# testable under cdx-rl's own interpreter, which has no mujoco pin.
#
# `harness/_steps.py` is laid out this way and `harness/test_steps.py` is what
# it buys; `hazard15.py` imports the engine at module scope and has no test as
# a direct result. The divider is load-bearing.
# ---------------------------------------------------------------------------


def settle_frames(interval_s: float, settle_s: float = SETTLE_SECONDS) -> int:
    """The settle window in control steps, from a DURATION.

    The same lesson `harness/_steps.min_airborne_steps` encodes: 003 halved the
    control rate from 100 Hz to 50 Hz, and any threshold written as a step
    count would silently have doubled in meaning and made every figure
    incomparable with the baseline it was being compared against.
    """

    return max(0, int(round(settle_s / interval_s)))


def command_deltas(actions: list, interval_s: float) -> np.ndarray:
    """``|Δaction|`` per control step per joint, in the bundle's own unit.

    **The reset frame is DROPPED, not zero-filled**, and that is the whole
    subtlety. ``evaluate_episode`` passes ``action=None`` at the reset pose
    because no action has been taken there. Treating that as a row of zeros
    manufactures one enormous Δ at step 0 — the entire nominal pose, appearing
    as a single-step command jump — which is the reset-drop instrument error
    in a new place, and it would dominate a p95 over a short episode.

    Returns an ``(n-1, njoints)`` array, empty if fewer than two commands were
    issued.
    """

    issued = [a for a in actions if a is not None]
    if len(issued) < 2:
        return np.zeros((0, len(issued[0]) if issued else 0), dtype=float)
    arr = np.asarray(issued, dtype=float)
    return np.abs(np.diff(arr, axis=0))


def sign_reversals_per_s(signed_deltas: np.ndarray, interval_s: float) -> np.ndarray:
    """Direction changes per second, per joint.

    This is what separates *chatter* from *fast tracking*. A magnitude
    statistic cannot: a policy sweeping a joint quickly and a policy
    oscillating it both show a large mean ``|Δ|``, and only the second reverses.

    **The ceiling is ``1 / interval_s``, not ``1 / (2 · interval_s)``** — 50 /s
    at this bundle's 50 Hz, not 25. The first draft printed 25 by analogy with
    the Nyquist *frequency*, and then measured 38.6 against it, which is the
    kind of impossible reading that means the yardstick is wrong rather than
    the machine. A full oscillation carries **two** direction changes, so a
    command alternating every control step reverses once per step.

    A zero Δ is not a direction, so it neither counts as a reversal nor breaks
    the run on either side of it.
    """

    if signed_deltas.shape[0] < 2:
        return np.zeros(signed_deltas.shape[1] if signed_deltas.ndim > 1 else 0)
    signs = np.sign(signed_deltas)
    out = np.zeros(signs.shape[1], dtype=float)
    for j in range(signs.shape[1]):
        column = [s for s in signs[:, j] if s != 0.0]
        out[j] = sum(1 for a, b in zip(column, column[1:]) if a != b)
    seconds = signed_deltas.shape[0] * interval_s
    return out / seconds if seconds > 0 else out


def summarise(values: np.ndarray) -> dict:
    """mean / rms / p95 / max of a 1-D sample, as plain floats.

    RMS and mean are both reported because they differ on exactly the
    distribution this driver cares about: a joint that is still most of the
    time and spikes occasionally has a low mean and a high RMS, and calling
    either one "the jitter" alone would hide the other.
    """

    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        return {"mean": 0.0, "rms": 0.0, "p95": 0.0, "max": 0.0, "n": 0}
    return {
        "mean": float(np.mean(np.abs(arr))),
        "rms": float(np.sqrt(np.mean(arr * arr))),
        "p95": float(np.percentile(np.abs(arr), 95)),
        "max": float(np.max(np.abs(arr))),
        "n": int(arr.size),
    }


def tangential_speed(v_point: np.ndarray, normal: np.ndarray) -> float:
    """Speed of a contact point in the plane of its own contact.

    ``v_point`` is the world velocity of the material point on the foot that
    is currently touching; ``normal`` is the contact normal. The component
    along the normal is separation or penetration, not sliding, so it comes
    out: a foot pressing straight down is not slipping however fast it presses.
    """

    n = np.asarray(normal, dtype=float)
    norm = float(np.linalg.norm(n))
    if norm < 1.0e-12:
        return float(np.linalg.norm(v_point))
    n = n / norm
    v = np.asarray(v_point, dtype=float)
    return float(np.linalg.norm(v - np.dot(v, n) * n))


def centre_of_pressure(points: list, forces: list):
    """Force-weighted mean of contact positions, or ``None``.

    ``None`` rather than a divide when the total normal force is
    negligible — a foot carrying no load has no centre of pressure, and
    returning the unweighted centroid there would draw a confident point
    through an airborne foot.
    """

    if not points:
        return None
    f = np.asarray(forces, dtype=float)
    total = float(f.sum())
    if total < 1.0e-9:
        return None
    p = np.asarray(points, dtype=float)
    return (p * f[:, None]).sum(axis=0) / total


def cop_margins(cop: np.ndarray, box: dict) -> dict:
    """How much sole is left in each direction, in mm.

    ``box`` is ``{"x": [lo, hi], "y": [lo, hi]}`` in the same frame as ``cop``,
    derived from the model's own geoms rather than from the profile.

    **A CoP outside the box reports a NEGATIVE margin rather than clamping to
    zero.** Clamping would make "on the edge" and "off the edge" the same
    reading, and the second is a machine that is rolling over.
    """

    x, y = float(cop[0]), float(cop[1])
    return {
        "forward_mm": box["y"][1] - y,
        "backward_mm": y - box["y"][0],
        "lateral_mm": min(box["x"][1] - x, x - box["x"][0]),
    }


def sigma_from(samples) -> float:
    """The σ a `quiet` kernel should use: the MEDIAN of the samples given.

    `swirl_scale.py`'s rule, restated: *half the samples above it and half
    below is what makes the term a gradient rather than a constant*, because a
    saturated term is not a strong term, it is a constant.

    **Which samples are handed in is the whole decision, and it inverts
    `swirl_scale.py`'s own convention deliberately.** That driver measures its
    scales over the *recovery* regime only, excluding the frames where the
    machine is merely standing, because folding those in halves every scale.
    `quiet` is the one term whose entire subject **is** the machine standing
    still, so its σ must come from exactly the regime the others throw away.
    Sizing it off recoveries would make it too wide to tell jitter from
    stillness at all. The caller passes the settled frames; this function does
    not choose them, and the test pins that it does not silently concatenate.
    """

    arr = np.asarray(list(samples), dtype=float).ravel()
    if arr.size == 0:
        return 0.0
    return float(np.median(arr))


def trend(iters: list, values: list) -> dict:
    """Least-squares slope of a series against iteration.

    Same shape as `hazard15.trend`, including the ``degenerate`` early return
    that keeps the key set identical for a run with one checkpoint — an early
    return with a different shape is how that driver printed a KeyError
    instead of a table.
    """

    first = round(float(values[0]), 6) if values else 0.0
    last = round(float(values[-1]), 6) if values else 0.0
    if len(iters) < 2:
        return {"slope_per_1000": 0.0, "intercept": last, "first": first,
                "last": last, "n": len(iters), "degenerate": True}
    slope, intercept = np.polyfit(np.asarray(iters, dtype=float),
                                  np.asarray(values, dtype=float), 1)
    return {
        "slope_per_1000": round(float(slope) * 1000.0, 6),
        "intercept": round(float(intercept), 6),
        "first": first, "last": last, "n": len(iters), "degenerate": False,
    }


# ---------------------------------------------------------------------------
# The half that needs the engine.
# ---------------------------------------------------------------------------


def _engine():
    """Import the engine on demand, so the pure half above stays importable.

    ``hazard15.py`` does this at module scope. Doing it here instead is what
    lets ``test_jitter.py`` run under ``uv run pytest`` with no mujoco.
    """

    sys.path.insert(0, str(Path(MODULE_DIR) / "src" / "Mod" / "cadex"))
    sys.path.insert(0, str(REPO / "harness"))
    sys.path.insert(0, str(REPO))
    import CadexDynamics as cd  # noqa: E402

    from _episodes import apply_variant  # noqa: E402

    return cd, apply_variant


def _joint_columns(cd, model, task):
    """Resolve the bundle's ten actions to qvel DOF addresses, in bundle order.

    ``data.qvel`` is indexed by **DOF address**, not by joint id, and the two
    differ the moment a model carries a free joint — which this one does, at
    the pelvis. Getting this wrong reads the pelvis's linear velocity as a
    knee's.

    Refuses loudly on a missing joint, the way ``_steps._geom_groups`` refuses
    a missing body: a joint that resolves to nothing would silently contribute
    zero jitter and make the machine look stiller than it is.
    """

    mujoco = cd._mujoco_module()
    names, addresses, missing = [], [], []
    for action in task["actions"]:
        name = str(action["joint"])
        jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
        if jid < 0:
            missing.append(name)
            continue
        names.append(name)
        addresses.append(int(model.jnt_dofadr[jid]))
    if missing:
        raise SystemExit(
            f"the bundle names joints this model does not have: {missing}. "
            f"A joint that resolves to no DOF contributes no velocity, so "
            f"every episode would report a stiller machine rather than an "
            f"error."
        )
    return names, addresses


def _sole_boxes(cd, model, profile):
    """The per-foot sole box, in each foot's OWN frame, from the model's geoms.

    Rigid: the toe is welded to the foot, so this box does not move as the
    ankle flexes, which is what makes it the right frame for "where on the
    sole is the pressure".

    Returns ``{label: {"body": id, "x": [lo, hi], "y": [lo, hi]}}`` in mm.
    """

    mujoco = cd._mujoco_module()
    boxes = {}
    for label, body_names in profile["feet"].items():
        root = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                     body_names[0]))
        if root < 0:
            raise SystemExit(f"profile foot {label!r} names no such body")
        lo = np.array([math.inf, math.inf])
        hi = np.array([-math.inf, -math.inf])
        for name in body_names:
            body = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
            # Offset of this body's frame within the root foot's frame. The
            # chain is short and rigid, so walking body_pos up to the root is
            # exact; it is also the only route that does not need a pose.
            offset = np.zeros(3)
            walk = body
            while walk != root and walk > 0:
                offset = np.asarray(model.body_pos[walk], dtype=float) + offset
                walk = int(model.body_parentid[walk])
            for g in range(int(model.ngeom)):
                if int(model.geom_bodyid[g]) != body:
                    continue
                centre = np.asarray(model.geom_pos[g], dtype=float) + offset
                size = np.asarray(model.geom_size[g], dtype=float)
                lo = np.minimum(lo, (centre[:2] - size[:2]) * 1000.0)
                hi = np.maximum(hi, (centre[:2] + size[:2]) * 1000.0)
        boxes[label] = {"body": root,
                        "x": [float(lo[0]), float(hi[0])],
                        "y": [float(lo[1]), float(hi[1])]}
    return boxes


def _play(cd, model, task, header, weights, seed, columns, boxes, ground,
          feet_geoms, interval_s, deg_per_rad, action_fn=None):
    """One episode, returning the per-control-step record this driver needs."""

    mujoco = cd._mujoco_module()
    _names, addresses = columns
    nv = int(model.nv)
    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))
    wrench = np.zeros(6)

    frames = {"action": [], "qvel": [], "slip": [], "cop": [], "contact": []}

    def sample(step, data, _final, action):
        frames["action"].append(None if action is None
                                else [float(v) for v in action])
        frames["qvel"].append([abs(float(data.qvel[a])) * deg_per_rad
                               for a in addresses])

        per_foot_slip, per_foot_cop, per_foot_contact = {}, {}, {}
        for label, ids in feet_geoms.items():
            points, forces, speeds = [], [], []
            for c in range(int(data.ncon)):
                contact = data.contact[c]
                g1, g2 = int(contact.geom1), int(contact.geom2)
                if not ((g1 in ids and g2 in ground)
                        or (g2 in ids and g1 in ground)):
                    continue
                point = np.asarray(contact.pos, dtype=float)
                # MuJoCo's contact frame is row-major; row 0 is the normal.
                normal = np.asarray(contact.frame[0:3], dtype=float)
                mujoco.mj_contactForce(model, data, c, wrench)
                fn = abs(float(wrench[0]))
                # The world velocity of the MATERIAL POINT on the foot that is
                # touching, via the translational Jacobian at that point. This
                # is exact and needs no assumption about where a body's
                # reported velocity is anchored.
                mujoco.mj_jac(model, data, jacp, jacr, point,
                              boxes[label]["body"])
                v_point = jacp @ np.asarray(data.qvel, dtype=float)
                points.append(point * 1000.0)
                forces.append(fn)
                speeds.append(tangential_speed(v_point * 1000.0, normal))
            per_foot_contact[label] = bool(points)
            # Load-weighted, so a barely-touching toe does not count as much
            # sliding as a fully-loaded heel.
            if points and sum(forces) > 1.0e-9:
                w = np.asarray(forces) / sum(forces)
                per_foot_slip[label] = float(np.dot(w, speeds))
            else:
                per_foot_slip[label] = 0.0
            cop = centre_of_pressure(points, forces)
            if cop is None:
                per_foot_cop[label] = None
            else:
                body = boxes[label]["body"]
                rot = np.asarray(data.xmat[body], dtype=float).reshape(3, 3)
                origin = np.asarray(data.xpos[body], dtype=float) * 1000.0
                per_foot_cop[label] = (rot.T @ (cop - origin)).tolist()
        frames["slip"].append(per_foot_slip)
        frames["cop"].append(per_foot_cop)
        frames["contact"].append(per_foot_contact)
        return None

    if action_fn is None:
        if header is None:
            zeros = [0.0] * len(task["actions"])

            def action_fn(_step, _obs, _z=zeros):
                return _z
        else:
            def action_fn(_step, obs):
                return cd.policy_forward(header, weights, obs)

    episode = cd.evaluate_episode(model, task, actions=action_fn,
                                  sample=sample, seed=int(seed))
    return episode, frames


def measure(policy_path, task: dict, model_xml: bytes, seeds: list,
            profile: dict, *, disturbance: str = "declared",
            open_loop: str = "off") -> dict:
    """Jitter, slip and centre of pressure over ``seeds`` episodes.

    ``policy_path`` of ``None`` measures the **zero-action servo** — the model
    holding its nominal pose with no network in it at all. That is the floor
    every jitter figure has to be read against: under a position action space
    zero action is not "limp", it is "hold the nominal pose", and whatever
    ``Σ|q̇|`` the servo and the contacts produce on their own is not something
    a reward term can remove.
    """

    cd, apply_variant = _engine()
    mujoco = cd._mujoco_module()

    if policy_path is None:
        header = weights = None
        row_label = "zero-action servo"
    else:
        container = cd.decode_policy(policy_path.read_bytes(),
                                     context=policy_path.name)
        header, weights = container["header"], container["weights"]
        row_label = policy_path.name

    played = (apply_variant(task, {"disturbance": False})
              if disturbance == "none" else task)

    interval_s = float(task["episode"]["control_interval_s"])
    settle = settle_frames(interval_s)
    # From the bundle, not typed: the reward kernel this sizes reads the same
    # channels through the same scale.
    scales = [float(row["scale"]) for row in task["observations"]
              if str(row.get("kind")) == "velocity"]
    deg_per_rad = scales[0] if scales else DEG_PER_RAD

    whole = {"delta": [], "qvel": [], "slip": [], "reversals": []}
    settled = {"delta": [], "qvel": [], "slip": [], "reversals": []}
    sigma_samples: list[float] = []
    cop_rows: dict = {}
    contact_seconds: dict = {}
    slip_integral: dict = {}
    frozen_rms: list[float] = []
    closed_rms: list[float] = []
    survived = 0
    boxes = None

    for seed in seeds:
        model = cd.load_model(model_xml)   # per episode (ADR-103 §9)
        columns = _joint_columns(cd, model, task)
        boxes = _sole_boxes(cd, model, profile)
        feet_geoms, ground = {}, set()
        for label, body_names in profile["feet"].items():
            ids = []
            for name in body_names:
                body = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                             name))
                ids += [g for g in range(int(model.ngeom))
                        if int(model.geom_bodyid[g]) == body]
            feet_geoms[label] = ids
        gbody = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                      profile["ground"]))
        ground = {g for g in range(int(model.ngeom))
                  if int(model.geom_bodyid[g]) == gbody}

        episode, frames = _play(cd, model, played, header, weights, seed,
                                columns, boxes, ground, feet_geoms,
                                interval_s, deg_per_rad)
        if episode.get("truncated"):
            survived += 1

        deltas = command_deltas(frames["action"], interval_s)
        issued = [a for a in frames["action"] if a is not None]
        signed = (np.diff(np.asarray(issued, dtype=float), axis=0)
                  if len(issued) >= 2 else np.zeros((0, 0)))
        qvel = np.asarray(frames["qvel"], dtype=float)

        whole["delta"].append(deltas)
        whole["qvel"].append(qvel)
        if signed.shape[0] >= 2:
            whole["reversals"].append(sign_reversals_per_s(signed, interval_s))

        # The settled window. `deltas` is one shorter than the frame list and
        # is indexed from the FIRST issued command, so the same wall-clock
        # instant is a different row index in the two arrays; both offsets are
        # derived rather than assumed equal.
        d_from = max(0, settle - 1)
        settled["delta"].append(deltas[d_from:])
        settled["qvel"].append(qvel[settle:])
        if signed[d_from:].shape[0] >= 2:
            settled["reversals"].append(
                sign_reversals_per_s(signed[d_from:], interval_s))

        # Σ|q̇| over the ten actuated joints, per settled frame. THIS is the
        # quantity a `quiet` kernel sums, so these are the samples σ is the
        # median of.
        if qvel[settle:].size:
            sigma_samples += list(qvel[settle:].sum(axis=1))
            closed_rms.append(float(np.sqrt(np.mean(
                qvel[settle:].sum(axis=1) ** 2))))

        for label in profile["feet"]:
            planted = [i for i in range(settle, len(frames["contact"]))
                       if frames["contact"][i][label]]
            contact_seconds[label] = (contact_seconds.get(label, 0.0)
                                      + len(planted) * interval_s)
            speeds = [frames["slip"][i][label] for i in planted]
            slip_integral[label] = (slip_integral.get(label, 0.0)
                                    + sum(speeds) * interval_s)
            settled["slip"].append(np.asarray(speeds, dtype=float))
            cops = [frames["cop"][i][label] for i in planted
                    if frames["cop"][i][label] is not None]
            cop_rows.setdefault(label, []).extend(cops)

        # Open loop: the SAME seed, so the reset draw and the whole
        # disturbance schedule are identical and the only difference is where
        # the command came from.
        if open_loop != "off" and issued:
            if open_loop == "replay":
                sequence = issued

                def act(step, _obs, _s=sequence):
                    return _s[min(int(step), len(_s) - 1)]
            else:
                # **The settled slice can be empty**, and then `mean(axis=0)`
                # over a (0,) array collapses to a scalar NaN rather than a
                # ten-vector — which reaches the engine as `'float' object is
                # not iterable` from inside `evaluate_episode`, three frames
                # away from the cause. An episode that ended before the settle
                # window has no settled mean to hold, so it falls back to the
                # whole command sequence and, failing that, is skipped.
                window = issued[max(0, settle - 1):] or issued
                if not window:
                    continue
                held = [float(v) for v in
                        np.asarray(window, dtype=float).mean(axis=0)]

                def act(_step, _obs, _h=held):
                    return _h

            model2 = cd.load_model(model_xml)
            _episode2, frames2 = _play(cd, model2, played, header, weights,
                                       seed, columns, boxes, ground,
                                       feet_geoms, interval_s, deg_per_rad,
                                       action_fn=act)
            q2 = np.asarray(frames2["qvel"], dtype=float)[settle:]
            if q2.size:
                frozen_rms.append(float(np.sqrt(np.mean(
                    q2.sum(axis=1) ** 2))))

    def stack(chunks):
        usable = [c for c in chunks if getattr(c, "size", 0)]
        return np.vstack(usable) if usable else np.zeros((0, 0))

    w_delta, s_delta = stack(whole["delta"]), stack(settled["delta"])
    w_qvel, s_qvel = stack(whole["qvel"]), stack(settled["qvel"])
    s_sum = s_qvel.sum(axis=1) if s_qvel.size else np.zeros(0)
    w_sum = w_qvel.sum(axis=1) if w_qvel.size else np.zeros(0)
    reversals = (np.vstack(settled["reversals"]).mean(axis=0)
                 if settled["reversals"] else np.zeros(0))
    slip_all = np.concatenate([s for s in settled["slip"] if s.size]) \
        if any(s.size for s in settled["slip"]) else np.zeros(0)

    row = {
        "policy": row_label,
        "episodes": len(seeds),
        "survived": survived,
        "disturbance": disturbance,
        "settle_seconds": SETTLE_SECONDS,
        "control_interval_s": interval_s,
        # --- command jitter, DEGREES ---
        "cmd_delta_mean_deg": round(float(np.mean(s_delta)) if s_delta.size
                                    else 0.0, 5),
        "cmd_delta_p95_deg": round(float(np.percentile(s_delta, 95))
                                   if s_delta.size else 0.0, 5),
        "cmd_delta_mean_deg_s": round((float(np.mean(s_delta)) / interval_s)
                                      if s_delta.size else 0.0, 4),
        "cmd_delta_per_joint_deg": [round(v, 5) for v in
                                    (s_delta.mean(axis=0) if s_delta.size
                                     else [])],
        "cmd_reversals_per_s": round(float(reversals.mean())
                                     if reversals.size else 0.0, 4),
        "cmd_reversals_per_joint_per_s": [round(v, 4) for v in reversals],
        "cmd_reversal_ceiling_per_s": round(1.0 / interval_s, 4),
        "whole_cmd_delta_mean_deg": round(float(np.mean(w_delta))
                                          if w_delta.size else 0.0, 5),
        # --- joint jitter, DEG/S ---
        "qvel_abs_mean_deg_s": round(float(np.mean(s_qvel)) if s_qvel.size
                                     else 0.0, 4),
        "qvel_per_joint_mean_deg_s": [round(v, 4) for v in
                                      (s_qvel.mean(axis=0) if s_qvel.size
                                       else [])],
        "sum_abs_qvel_mean_deg_s": round(float(np.mean(s_sum))
                                         if s_sum.size else 0.0, 4),
        "sum_abs_qvel_rms_deg_s": round(float(np.sqrt(np.mean(s_sum ** 2)))
                                        if s_sum.size else 0.0, 4),
        "sum_abs_qvel_p95_deg_s": round(float(np.percentile(s_sum, 95))
                                        if s_sum.size else 0.0, 4),
        "whole_sum_abs_qvel_mean_deg_s": round(float(np.mean(w_sum))
                                               if w_sum.size else 0.0, 4),
        # --- the number a `quiet` kernel needs ---
        "sigma_recommended_deg_s": round(sigma_from(sigma_samples), 4),
        "sigma_samples": len(sigma_samples),
        # --- slip ---
        "slip_speed_mean_mm_s": round(float(np.mean(slip_all))
                                      if slip_all.size else 0.0, 4),
        "slip_speed_p95_mm_s": round(float(np.percentile(slip_all, 95))
                                     if slip_all.size else 0.0, 4),
        "contact_seconds": {k: round(v, 3) for k, v in contact_seconds.items()},
        "slip_mm_per_contact_s": {
            k: round(slip_integral[k] / contact_seconds[k], 4)
            if contact_seconds.get(k) else 0.0
            for k in slip_integral},
    }

    # Per-joint contribution, which is what selects the `quiet` channel set.
    if s_qvel.size:
        share = s_qvel.mean(axis=0) / max(float(s_qvel.mean(axis=0).sum()),
                                          1.0e-12)
        row["qvel_share_per_joint"] = [round(float(v), 5) for v in share]
        row["joints"] = _joint_names(task)

    # Centre of pressure, against the box the MODEL declares.
    if boxes:
        row["sole_box_mm"] = {k: {"x": v["x"], "y": v["y"]}
                              for k, v in boxes.items()}
        row["profile_support_polygon_mm"] = profile.get("support_polygon_mm")
        cop_out = {}
        for label, cops in cop_rows.items():
            if not cops:
                continue
            arr = np.asarray(cops, dtype=float)
            margins = [cop_margins(c, boxes[label]) for c in arr]
            cop_out[label] = {
                "frames": int(arr.shape[0]),
                "forward_max_mm": round(float(arr[:, 1].max()), 3),
                "backward_max_mm": round(float(arr[:, 1].min()), 3),
                "lateral_max_mm": round(float(np.abs(arr[:, 0]).max()), 3),
                "min_forward_margin_mm": round(
                    min(m["forward_mm"] for m in margins), 3),
                "min_backward_margin_mm": round(
                    min(m["backward_mm"] for m in margins), 3),
                "min_lateral_margin_mm": round(
                    min(m["lateral_mm"] for m in margins), 3),
                "frac_frames_within_2mm_of_edge": round(float(np.mean(
                    [1.0 if min(m["forward_mm"], m["backward_mm"],
                                m["lateral_mm"]) <= 2.0 else 0.0
                     for m in margins])), 5),
            }
        row["cop"] = cop_out

    if open_loop != "off" and closed_rms and frozen_rms:
        closed = float(np.mean(closed_rms))
        frozen = float(np.mean(frozen_rms))
        row["open_loop_mode"] = open_loop
        row["closed_loop_rms_deg_s"] = round(closed, 4)
        row["open_loop_rms_deg_s"] = round(frozen, 4)
        # **The decisive ratio.** Near 0 means the servo and the contacts ring
        # under a perfectly steady setpoint and no reward term can quiet them;
        # near 1 means the setpoint is the source and a `quiet` kernel is
        # aimed at the right thing.
        row["quiet_headroom"] = round(1.0 - frozen / closed, 4) \
            if closed > 1.0e-9 else 0.0
    return row


def _joint_names(task: dict) -> list:
    return [str(a["joint"]) for a in task["actions"]]


def main() -> int:
    ap = argparse.ArgumentParser(description="jitter, slip and centre of "
                                             "pressure for mg-legs")
    ap.add_argument("--policy", action="append", type=Path)
    ap.add_argument("--series", type=Path,
                    help="run directory: score its periodic checkpoints at "
                         "--stride and print the trend")
    ap.add_argument("--stride", type=int, default=250)
    ap.add_argument("--task", required=True, type=Path)
    ap.add_argument("--model", type=Path)
    ap.add_argument("--profile", default="mg-legs",
                    help="mechanism profile, by name or path")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--disturbance", choices=("declared", "none"),
                    default="declared",
                    help="'none' DROPS the schedule (not scale 0.0): the "
                         "pure standing limit cycle. 'declared' is the "
                         "regime a `quiet` kernel needs a gradient in, and "
                         "is what σ is sized off.")
    ap.add_argument("--open-loop", choices=("off", "replay", "frozen"),
                    default="off",
                    help="'frozen' replays with the setpoint held at its "
                         "settled mean — the floor the servo and contacts "
                         "impose. 'replay' rewrites the same command "
                         "sequence open-loop.")
    ap.add_argument("--zero-action", action="store_true",
                    help="also measure the ZERO-ACTION SERVO — the model "
                         "holding its nominal pose with no network in it. "
                         "Under a position action space that is the floor "
                         "every jitter figure has to be read against.")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not args.policy and not args.series and not args.zero_action:
        ap.error("one of --policy, --series or --zero-action is required")

    task = json.loads(args.task.read_text())
    model_path = args.model or (args.task.parent / "model-model.xml")
    model_xml = model_path.read_bytes()
    seeds = list(range(args.seeds))

    # **Refuse rather than assume the unit.** Under a torque action space
    # every "deg" key below would be a lie in exactly the way
    # `_episodes._torque_columns` is a lie under a position one.
    units = {str(a.get("unit")) for a in task["actions"]}
    kinds = {str(a.get("kind")) for a in task["actions"]}
    if units != {"deg"} or kinds != {"position"}:
        raise SystemExit(
            f"this driver reports command jitter in DEGREES and this bundle "
            f"declares units {sorted(units)} / kinds {sorted(kinds)}. "
            f"`harness/_episodes._torque_columns` is what happens when an "
            f"instrument assumes the action's unit instead of reading it: it "
            f"labels degrees as N·mm and every table built on it is wrong."
        )

    sys.path.insert(0, str(REPO))
    from harness.episodes import BundleError, series_checkpoints  # noqa: E402

    # Shared rather than copied — the profile's `feet` map carries `$comment`
    # keys and `load_profile` is what strips them, along with checking that the
    # four required keys are present. A second reader of the same file is a
    # second place for the same drift.
    from harness.steps import load_profile  # noqa: E402

    profile = load_profile(args.profile)

    selected = []
    if args.zero_action:
        # First, so every table reads against the floor rather than needing
        # the reader to remember it.
        selected.append((None, None))
    selected += [(None, p) for p in (args.policy or [])]
    if args.series:
        try:
            selected += series_checkpoints(args.series, args.stride)
        except BundleError as exc:
            raise SystemExit(str(exc)) from None

    rows = []
    for iteration, p in selected:
        row = measure(p, task, model_xml, seeds, profile,
                      disturbance=args.disturbance, open_loop=args.open_loop)
        if iteration is not None:
            row["iteration"] = iteration
        rows.append(row)

    series_rows = [r for r in rows if "iteration" in r]
    tr = trend([r["iteration"] for r in series_rows],
               [r["sum_abs_qvel_rms_deg_s"] for r in series_rows]) \
        if series_rows else None

    if args.json:
        out = {"driver": "jitter", "seeds": seeds, "task": str(args.task),
               "disturbance": args.disturbance, "open_loop": args.open_loop,
               "rows": rows}
        if tr:
            out["qvel_trend"] = tr
        print(json.dumps(out, indent=1))
        return 0

    print(f"jitter — how still is it, {len(seeds)} seeds, disturbance="
          f"{args.disturbance}, sampled at the control rate\n")
    print(f"  {'policy':<28} {'surv':>7} | {'command, DEGREES':>26} | "
          f"{'joints, DEG/S':>24}")
    print(f"  {'':<28} {'':>7} | {'mean Δ':>8} {'p95 Δ':>8} {'rev/s':>8} | "
          f"{'Σ|q̇| mean':>11} {'Σ|q̇| rms':>11}")
    for r in rows:
        print(f"  {r['policy']:<28} {r['survived']:>3}/{r['episodes']:<3} | "
              f"{r['cmd_delta_mean_deg']:>8.3f} {r['cmd_delta_p95_deg']:>8.3f} "
              f"{r['cmd_reversals_per_s']:>8.2f} | "
              f"{r['sum_abs_qvel_mean_deg_s']:>11.2f} "
              f"{r['sum_abs_qvel_rms_deg_s']:>11.2f}")

    print(f"\n  reversal ceiling at this control rate: "
          f"{rows[0]['cmd_reversal_ceiling_per_s']:.1f} /s. A large mean Δ "
          f"with a low\n  reversal rate is fast TRACKING; large with a high "
          f"rate is CHATTER. Neither\n  number decides it alone.")

    print(f"\n  σ a `quiet` kernel should use (median settled Σ|q̇|):")
    for r in rows:
        print(f"    {r['policy']:<28} {r['sigma_recommended_deg_s']:>10.3f} "
              f"deg/s   over {r['sigma_samples']} settled frames")

    print(f"\n  sole slip while planted, and centre of pressure:")
    for r in rows:
        print(f"    {r['policy']}")
        print(f"      slip {r['slip_speed_mean_mm_s']:.2f} mm/s mean, "
              f"{r['slip_speed_p95_mm_s']:.2f} p95   "
              f"{r['slip_mm_per_contact_s']}")
        for label, c in (r.get("cop") or {}).items():
            print(f"      cop {label}: fwd {c['forward_max_mm']:+7.2f}  "
                  f"back {c['backward_max_mm']:+7.2f}  "
                  f"lat {c['lateral_max_mm']:6.2f} mm   "
                  f"min margin {min(c['min_forward_margin_mm'], c['min_backward_margin_mm'], c['min_lateral_margin_mm']):+6.2f}  "
                  f"at-edge {c['frac_frames_within_2mm_of_edge']*100:.1f} %")
        if r.get("sole_box_mm"):
            box = list(r["sole_box_mm"].values())[0]
            print(f"      sole box from the MODEL, foot frame: "
                  f"x {box['x'][0]:+.2f}…{box['x'][1]:+.2f}  "
                  f"y {box['y'][0]:+.2f}…{box['y'][1]:+.2f} mm")
            print(f"      the profile declares "
                  f"{r.get('profile_support_polygon_mm')}")
            print(f"      — and those are NOT the same frame: 45.5/24.5 are "
                  f"ankle-bracket-relative\n        (the foot sits 12.25 mm "
                  f"forward of it) and 50 is the DOUBLE-support\n        "
                  f"half-width, where one sole is ±20 mm.")

    if any("quiet_headroom" in r for r in rows):
        print(f"\n  open loop ({args.open_loop}): does the jitter survive a "
              f"steady setpoint?")
        for r in rows:
            if "quiet_headroom" not in r:
                continue
            print(f"    {r['policy']:<28} closed "
                  f"{r['closed_loop_rms_deg_s']:>8.2f}  open "
                  f"{r['open_loop_rms_deg_s']:>8.2f}  headroom "
                  f"{r['quiet_headroom']*100:>6.1f} %")
        print(f"\n    The rule was written down before this was run "
              f"(ADR-097): headroom below\n    30 % means the jitter is the "
              f"servo and the contacts rather than the policy's\n    "
              f"setpoint, and a `quiet` reward kernel is aimed at the wrong "
              f"thing.")

    print(f"\n  The whole-episode columns are the RESET DROP, not the "
          f"posture: whole Σ|q̇|\n  mean is "
          f"{rows[0]['whole_sum_abs_qvel_mean_deg_s']:.1f} deg/s against the "
          f"settled {rows[0]['sum_abs_qvel_mean_deg_s']:.1f}. They are "
          f"printed\n  so the settled window cannot be mistaken for a "
          f"cherry-picked one.")

    if tr and tr.get("degenerate"):
        print(f"\n  trend: only {tr['n']} checkpoint(s) — none to fit.")
    elif tr:
        print(f"\n  settled Σ|q̇| rms over {tr['n']} checkpoints "
              f"({series_rows[0]['iteration']} → "
              f"{series_rows[-1]['iteration']}): "
              f"{tr['first']:.2f} → {tr['last']:.2f} deg/s, slope "
              f"{tr['slope_per_1000']:+.2f} per 1000 iterations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
