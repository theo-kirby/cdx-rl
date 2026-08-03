#!/usr/bin/env python
"""Did it STEP? Runs under the TRAINER interpreter. Imports nothing of cdx-rl.

The sibling of :mod:`harness._episodes`, and the far side of the same bridge.
``compare`` and ``capability`` reduce an episode to *did it survive*, and
survival is not the behaviour a push-recovery task is asking for: a machine
that rejects every shove on its ankles and a machine that catches itself with
a step both read as ``12/12``, and only the second is what was wanted.

This reads the thing itself, off MuJoCo's own contact list rather than off a
summary: **when did a foot leave the floor, how far did it travel while it
was off, and where did it come down.**

**Contacts, not heights, and ADR-107 is why.** A foot lift was reported there
as "never leaves the ground" by one instrument and as *5.91 mm at 2.090 s* by
another, and the difference was that the first read a statistic over an
interval and the second read the trace. Contacts are what the solver actually
computed; a height threshold is a number somebody picked. Both are reported,
so the threshold can be argued with.

Job spec on stdin, NDJSON rows on stdout::

    {"schema": "cdxrl-steps-job-v1",
     "module_dir": "…/src/Mod/cadex",
     "model_xml": "<path>", "task_path": "<path>",
     "profile": {"feet": {"foot_l": ["foot_l", "toe_l"],
                          "foot_r": ["foot_r", "toe_r"]},
                 "ground": "ground",
                 "step_mm": 10.0, "min_airborne_s": 0.030},
     "episodes": [{"id": "…", "policy": "<path>", "seed": 0}, …]}

Rows are ``{"row": "episode"|"error"|"done", …}``.

**The detection itself is pure and lives in :func:`lifts`**, which imports
nothing at all — so it can be imported and unit-tested under any interpreter,
including cdx-rl's own venv that has no mujoco. ``harness/test_steps.py``
does exactly that.
"""

from __future__ import annotations

import json
import math
import os
import sys
import traceback
from typing import Any

JOB_SCHEMA = "cdxrl-steps-job-v1"
ROW_SCHEMA = "cdxrl-steps-row-v1"

_STATE: dict[str, Any] = {}


def emit(row: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(row) + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# The pure half.
# ---------------------------------------------------------------------------

def min_airborne_steps(interval_s: float, min_airborne_s: float) -> int:
    """The airborne threshold in control steps, rounded UP.

    **A DURATION, AND IT USED TO BE A STEP COUNT.** The original was
    ``MIN_AIRBORNE_STEPS = 3`` with "three at 100 Hz is 30 ms" written beside
    it — and experiment 003 halved the control rate to 50 Hz, which would
    silently have made the same 3 a **60 ms** threshold. Every step count it
    produced would then have been measured against a different criterion than
    the baseline it was being compared with, and the comparison is the whole
    point of the number. So the threshold is the physical quantity and the
    step count is derived from the bundle's own control interval.
    """

    return max(1, math.ceil(min_airborne_s / interval_s - 1.0e-9))


def lifts(frames: list[dict[str, Any]], interval_s: float, *,
          step_mm: float, min_airborne_s: float) -> list[dict[str, Any]]:
    """Every airborne interval of every foot, as takeoff → landing.

    Pure: takes the trace, returns the events. No mujoco, no engine, no
    filesystem — which is what makes it testable without a GPU box.

    **The episode starts airborne, by design**, and that has to be handled
    here rather than by the caller. A reset variation that lifts the machine
    and drops it leaves both feet out of contact at step 0 of every episode,
    so the first landing is the drop rather than a step. Caught by calibrating
    against a known result: it reported a 30 mm "step" at 0.00 s in four
    episodes of four, which is the reset. **A foot cannot step off a floor it
    has not reached**, so an interval only counts once that foot has been
    planted at least once.
    """

    if not frames:
        return []
    feet = sorted(frames[0]["down"])
    minimum = min_airborne_steps(interval_s, min_airborne_s)
    found: list[dict[str, Any]] = []

    for foot in feet:
        planted = [f["z"][foot] for f in frames if f["down"][foot]]
        base = min(planted) if planted else 0.0

        start: int | None = None
        planted_yet = False
        for index, frame in enumerate(frames):
            if not frame["down"][foot]:
                if start is None and planted_yet:
                    start = index
                continue
            planted_yet = True
            if start is None:
                continue
            if index - start >= minimum:
                found.append(_lift(frames, foot, start, index, base,
                                   interval_s, step_mm))
            start = None
        # Still airborne at the end: it did not land, so it is not a step.
        if start is not None and len(frames) - start >= minimum:
            entry = _lift(frames, foot, start, len(frames) - 1, base,
                          interval_s, step_mm)
            entry["landed"] = False
            entry["step"] = False
            found.append(entry)

    return sorted(found, key=lambda entry: entry["takeoff_s"])


def _lift(frames, foot, start, end, base, interval_s, step_mm) -> dict[str, Any]:
    took_off = frames[max(0, start - 1)]["xy"][foot]
    landed = frames[end]["xy"][foot]
    peak = max(frames[i]["z"][foot] for i in range(start, end + 1))
    travel = math.hypot(landed[0] - took_off[0], landed[1] - took_off[1])
    return {
        "foot": foot,
        "takeoff_s": start * interval_s,
        "landing_s": end * interval_s,
        "airborne_s": (end - start) * interval_s,
        "travel_mm": travel,
        "peak_mm": peak - base,
        "landed": True,
        # A STEP is a lift that went somewhere. 10 mm is about a fifth of
        # this machine's half-stance: below that the foot came down where it
        # left, which is a shuffle or a hop and not a step out.
        "step": travel >= step_mm,
    }


# ---------------------------------------------------------------------------
# The half that needs the engine.
# ---------------------------------------------------------------------------

def _geom_groups(cd: Any, model: Any, profile: dict[str, Any]):
    """Resolve the profile's body names to geom ids, and refuse if any miss.

    Named bodies rather than hardcoded ones: this driver is parameterized by
    mechanism, and a profile that names a body the model does not have is a
    configuration error worth failing loudly on rather than a foot that
    silently never touches the ground.
    """

    mujoco = cd._mujoco_module()
    feet: dict[str, list[int]] = {}
    missing: list[str] = []

    def geoms_of(body_name: str) -> list[int]:
        body = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name))
        if body < 0:
            missing.append(body_name)
            return []
        return [g for g in range(int(model.ngeom))
                if int(model.geom_bodyid[g]) == body]

    for label, bodies in profile["feet"].items():
        ids: list[int] = []
        for body_name in bodies:
            ids.extend(geoms_of(body_name))
        feet[label] = ids

    ground = set(geoms_of(profile["ground"]))
    if missing:
        raise RuntimeError(
            f"the profile names bodies this model does not have: "
            f"{sorted(set(missing))}. A foot that resolves to no geom is "
            f"never in contact, so every episode would report a permanent "
            f"lift rather than an error."
        )
    for label, ids in feet.items():
        if not ids:
            raise RuntimeError(f"foot {label!r} resolved to no geoms")
    if not ground:
        raise RuntimeError(f"ground body {profile['ground']!r} has no geoms")
    return feet, ground, {label: profile["feet"][label][0] for label in feet}


def _init(spec: dict[str, Any]) -> None:
    sys.path.insert(0, spec["module_dir"])
    import CadexDynamics as cd  # noqa: E402

    _STATE["cd"] = cd
    _STATE["task"] = json.loads(open(spec["task_path"], encoding="utf-8").read())
    _STATE["model_bytes"] = open(spec["model_xml"], "rb").read()
    _STATE["profile"] = spec["profile"]


def _play(request: dict[str, Any]) -> dict[str, Any]:
    """One episode, as a per-control-step record of every declared foot."""

    cd = _STATE["cd"]
    task = _STATE["task"]
    profile = _STATE["profile"]
    mujoco = cd._mujoco_module()

    container = cd.decode_policy(open(request["policy"], "rb").read())
    header, weights = container["header"], container["weights"]

    # THE POLICY's channel list, not the task's. A header names the inputs
    # its first layer has; a task names what the model publishes; and once a
    # task gains a channel (ADR-116's `cam_*`) the two stop being equal.
    # Without this an older policy cannot be scored on a newer task at all,
    # which is exactly the comparison worth having.
    names = [str(n) for n in header.get("observations") or ()]
    if not names:
        names = [c for row in task["observations"] for c in row["channels"]]

    model = cd.load_model(_STATE["model_bytes"])  # per episode (ADR-103 §9)
    feet, ground, _ = _geom_groups(cd, model, profile)
    bodies = {label: int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                           profile["feet"][label][0]))
              for label in feet}

    frames: list[dict[str, Any]] = []

    def act(_step: int, observation: Any) -> Any:
        return cd.policy_forward(header, weights,
                                 [observation[n] for n in names])

    # `sample` is the engine's own per-step hook: (step, data, final, action)
    # invoked once at the reset pose and once after every control step's
    # integration, with every non-None return joining `samples`. Returning
    # None keeps that list empty and uses the hook purely as an observer,
    # which is what this is — the frames are accumulated here.
    #
    # Using the hook rather than a second episode loop is not a style
    # preference. CadexDynamics already carries three evaluators of the same
    # reward whitelist and a loop written here would be a fourth place for
    # the same drift; `evaluate_episode` is also what produced ADR-099's
    # table, so a number from it is comparable with every number in it.
    def sample(step: int, data: Any, _final: bool, _action: Any) -> None:
        down = {label: False for label in feet}
        for c in range(int(data.ncon)):
            contact = data.contact[c]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            for label, ids in feet.items():
                if (g1 in ids and g2 in ground) or (g2 in ids and g1 in ground):
                    down[label] = True
        frames.append({
            "step": int(step),
            "down": down,
            "xy": {label: [float(data.xpos[bodies[label]][0]) * 1000.0,
                           float(data.xpos[bodies[label]][1]) * 1000.0]
                   for label in feet},
            "z": {label: float(data.xpos[bodies[label]][2]) * 1000.0
                  for label in feet},
        })
        return None

    episode = cd.evaluate_episode(model, task, actions=act, sample=sample,
                                  seed=int(request["seed"]))

    interval = float(task["episode"]["control_interval_s"])
    events = lifts(frames, interval,
                   step_mm=float(profile["step_mm"]),
                   min_airborne_s=float(profile["min_airborne_s"]))
    steps = [e for e in events if e["step"]]
    return {
        "schema": ROW_SCHEMA,
        "row": "episode",
        "id": request["id"],
        "policy": request["policy"],
        "seed": int(request["seed"]),
        "survived": bool(episode["truncated"]),
        "termination": None if episode["truncated"] else str(episode["termination"]),
        "control_steps": len(episode["steps"]),
        "max_steps": int(task["episode"]["max_steps"]),
        "lifts": len(events),
        "steps": len(steps),
        "longest_step_mm": max((e["travel_mm"] for e in steps), default=0.0),
        "highest_lift_mm": max((e["peak_mm"] for e in events), default=0.0),
        "events": events,
    }


def main() -> int:
    spec = json.loads(sys.stdin.readline())
    if spec.get("schema") != JOB_SCHEMA:
        emit({"row": "error", "error": f"bad schema {spec.get('schema')!r}"})
        return 2
    try:
        _init(spec)
    except Exception as exc:  # noqa: BLE001
        emit({"row": "error", "error": f"{type(exc).__name__}: {exc}",
              "traceback": traceback.format_exc()[-4000:]})
        return 1

    for request in spec["episodes"]:
        try:
            emit(_play(request))
        except Exception as exc:  # noqa: BLE001 — one bad episode must not kill a sweep
            emit({"schema": ROW_SCHEMA, "row": "error",
                  "id": request.get("id"), "policy": request.get("policy"),
                  "seed": request.get("seed"),
                  "error": f"{type(exc).__name__}: {exc}",
                  "traceback": traceback.format_exc()[-4000:]})
    emit({"schema": ROW_SCHEMA, "row": "done"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
