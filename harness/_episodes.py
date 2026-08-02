#!/usr/bin/env python
"""Play episodes. Runs under the TRAINER interpreter. Imports nothing of cdx-rl.

This is the far side of the bridge in :mod:`harness.trainer_venv`, and the
**only** file in this repository that imports ``CadexDynamics``. cdx-rl code
executes it as a subprocess and never imports it, which is what keeps the
LGPL boundary a process boundary and cdx-rl's own venv free of a second
mujoco pin.

It reads one JSON job spec on stdin and writes NDJSON rows on stdout::

    {"schema": "cdxrl-episode-job-v1",
     "module_dir": "/home/theo/cadex/src/Mod/cadex",
     "model_xml": "<path>", "task_path": "<path>", "task_sha256": "<hex>",
     "jobs": 16,
     "variants": {"declared": {"disturbance_scale": 1.0}},
     "verify": ["<policy path>", …],
     "episodes": [{"id": "…", "policy": "<path>", "variant": "declared",
                   "seed": 0}, …]}

Rows are ``{"row": "verify"|"episode"|"error"|"done", …}``. One per episode,
flushed, so a caller can report progress over a sweep that takes minutes.

**Why the engine's own runner rather than a re-implementation.**
``CadexDynamics.evaluate_episode`` is what produced ADR-099's table. A second
episode loop written here would be a fourth evaluator of the same reward
whitelist, and the first thing it would do is disagree with the engine about
something small and undocumented. The module imports nothing from FreeCAD —
its own docstring says so, and it defers ``import mujoco`` into function
bodies — so importing it standalone is supported rather than a trick.

**ADR-103 §9 is obeyed structurally.** ``evaluate_episode`` multiplies domain
randomisation into the model *in place* and keeps no baseline, so an
evaluator that loops over one loaded model compounds every draw it has made.
:func:`_play` therefore loads the model from bytes for every single episode.
It costs about four milliseconds against a ~180 ms episode. The
same-file-twice test the drivers run is the check that this stays true.
"""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import sys
import traceback
from typing import Any

JOB_SCHEMA = "cdxrl-episode-job-v1"
ROW_SCHEMA = "cdxrl-episode-row-v1"

#: Set by :func:`_init_worker`, read by :func:`_play`. Module-level because
#: a forked pool worker inherits them once instead of pickling a 30 KB task
#: bundle and a 13 KB XML per episode.
_STATE: dict[str, Any] = {}


def emit(row: dict[str, Any]) -> None:
    """One NDJSON row, flushed. Flushing is the point: an unflushed row is a
    row the caller learns about after the job it was meant to narrate."""

    sys.stdout.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Task variants
# ---------------------------------------------------------------------------


def apply_variant(task: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    """A **copy** of the bundle with the declared variation scaled or removed.

    Four knobs, and each is a question somebody wants answered separately:

    ``disturbance_scale``
        multiplies every ``newtons_low`` / ``newtons_high`` in
        ``disturbance``. This is ``capability``'s sweep axis, and it mutates
        a copy — a driver that scaled the bundle in place would poison every
        later row in the same process and the table would still look fine.
    ``disturbance``
        ``false`` drops the list entirely. Different from scale 0.0 in one
        way that matters: at scale 0.0 the schedule still *runs*, draws a
        start time and applies a zero force, so the episode is otherwise
        identical. Dropping the list removes the schedule. Both rows are
        reported because the difference is a check on the evaluator.
    ``reset_variation``
        ``false`` starts from the keyframe exactly. "Can it stand at all."
    ``randomisation``
        ``false`` removes the mass/friction draws. Only the no-variation row
        sets this; it is what makes that row *nothing varies* rather than
        *nothing pushes*.

    The copy is deep over the parts that are edited and shallow elsewhere,
    which is enough: nothing downstream writes to the untouched lists.
    """

    scale = variant.get("disturbance_scale")
    patched = dict(task)

    if variant.get("disturbance", True) is False:
        patched["disturbance"] = []
    elif scale is not None:
        scaled = []
        for record in task.get("disturbance") or []:
            entry = dict(record)
            entry["newtons_low"] = float(record["newtons_low"]) * float(scale)
            entry["newtons_high"] = float(record["newtons_high"]) * float(scale)
            scaled.append(entry)
        patched["disturbance"] = scaled

    if variant.get("reset_variation", True) is False:
        patched["reset_variation"] = []
    if variant.get("randomisation", True) is False:
        patched["randomisation"] = []
    return patched


# ---------------------------------------------------------------------------
# Per-episode summary
# ---------------------------------------------------------------------------


def _expression_value(cd: Any, expression: str, observation: dict[str, float],
                      *, context: str) -> float | None:
    """One of the bundle's own expressions, on one observation.

    Through ``compile_reward`` — the task's evaluator, not a re-derived one.
    ``drift`` in particular is a formula with an offset in it
    (``abs(com_x - 0.002) + abs(com_y - 1.225)``) and re-deriving "drift" as
    a distance from the origin would produce a number that is wrong by a
    constant and looks entirely plausible.
    """

    try:
        code = cd.compile_reward(expression, names=list(observation), context=context)
        return float(cd.evaluate_reward(code, observation, context=context))
    except Exception:  # noqa: BLE001 — a refusal here is a null column, not a crash
        return None


def _torque_columns(steps: list[dict[str, Any]], actions: list[dict[str, Any]]
                    ) -> dict[str, list[float]]:
    """Peak, mean and saturation per motor, over the episode's own frames.

    ``step["action"]`` is documented by ``evaluate_episode`` as *the clamped
    list actually written to ``data.ctrl``*, in the unit the bundle
    advertised — N·mm here. So this needs no instrumentation and no second
    opinion about units: it is the number that moved the mechanism.

    Hazard 15 is why these are computed unconditionally. A policy can play as
    a clean stand while holding motors at 98 % of a stall rating for six
    seconds, and nothing in the trajectory shows it.
    """

    count = len(actions)
    limits = [max(abs(float(a["low"])), abs(float(a["high"]))) for a in actions]
    peak = [0.0] * count
    total = [0.0] * count
    saturated = [0] * count
    for step in steps:
        for index, value in enumerate(step["action"][:count]):
            magnitude = abs(float(value))
            if magnitude > peak[index]:
                peak[index] = magnitude
            total[index] += magnitude
            if limits[index] > 0.0 and magnitude >= 0.9 * limits[index]:
                saturated[index] += 1
    frames = max(1, len(steps))
    return {
        "peak_torque_nmm": [round(value, 4) for value in peak],
        "mean_torque_nmm": [round(value / frames, 4) for value in total],
        "frac_above_90pct": [round(count_ / frames, 5) for count_ in saturated],
        "limit_nmm": limits,
    }


def summarise(cd: Any, episode: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    """One ``evaluate_episode`` result, as the row a table is built from.

    Everything computed here is printed by some driver. Nothing is collected
    and hidden behind a flag — ADR-106's termination mix was gathered from M9
    onward and never shown, and that cost three runs.
    """

    steps = episode["steps"]
    last = steps[-1]["observation"] if steps else {}
    control_interval = float(task["episode"]["control_interval_s"])
    end_time_s = float(steps[-1]["time_s"]) if steps else 0.0

    termination_values = []
    for term in task.get("termination") or []:
        termination_values.append({
            "label": str(term["label"]),
            "value": _expression_value(
                cd, str(term["expression"]), last,
                context=f"termination {term['label']!r}",
            ),
            "above": term.get("above"),
            "below": term.get("below"),
        })

    drift = None
    for term in task.get("reward") or []:
        if str(term.get("label") or "") == "drift":
            drift = _expression_value(
                cd, str(term["expression"]), last, context="reward 'drift'"
            )
            break

    # How far into its own disturbance schedule this episode got. ADR-106
    # found the second shove window had never once been exercised by reading
    # exactly this against the death step.
    disturbances = []
    for drawn in episode.get("disturbance") or []:
        start = drawn.get("start_s")
        disturbances.append({
            "label": drawn.get("label"),
            "newtons": drawn.get("newtons"),
            "azimuth_rad": drawn.get("azimuth_rad"),
            "start_s": start,
            # A window the episode never lived to see is a window that
            # measured nothing, however carefully it was declared.
            "reached": None if start is None else bool(end_time_s >= float(start)),
        })

    reset = (episode.get("reset_variation") or [{}])[0]

    row: dict[str, Any] = {
        "schema": ROW_SCHEMA,
        "row": "episode",
        "seed": episode.get("seed"),
        "step_count": episode["step_count"],
        "max_steps": int(task["episode"]["max_steps"]),
        "total_reward": episode["total_reward"],
        "reward_per_step": (
            episode["total_reward"] / episode["step_count"] if episode["step_count"] else None
        ),
        "termination": episode["termination"] or "survived",
        "truncated": bool(episode["truncated"]),
        "terminated_step": episode.get("terminated_step"),
        "end_time_s": round(end_time_s, 6),
        "control_interval_s": control_interval,
        "termination_values": termination_values,
        "drift": drift,
        "disturbances": disturbances,
        "reset_variation": {
            "tilt_rad": reset.get("tilt_rad"),
            "azimuth_rad": reset.get("azimuth_rad"),
            "height_m": reset.get("height_m"),
        },
        **_torque_columns(steps, task["actions"]),
    }
    return row


# ---------------------------------------------------------------------------
# The pool
# ---------------------------------------------------------------------------


def _init_worker(spec: dict[str, Any]) -> None:
    """Import the engine's runner once per worker and read the fixed inputs."""

    sys.path.insert(0, spec["module_dir"])
    import CadexDynamics  # noqa: PLC0415 — deliberately per-worker

    with open(spec["model_xml"], "rb") as handle:
        model_xml = handle.read()
    with open(spec["task_path"], "rb") as handle:
        task = json.loads(handle.read().decode("utf-8"))

    _STATE["cd"] = CadexDynamics
    _STATE["model_xml"] = model_xml
    _STATE["task"] = task
    _STATE["variants"] = {
        name: apply_variant(task, variant)
        for name, variant in (spec.get("variants") or {"declared": {}}).items()
    }
    _STATE["policies"] = {}


def _policy(path: str) -> dict[str, Any]:
    """One decoded ``.cxpolicy``, cached per worker.

    Decoding is cheap and caching it is not the point — the point is that
    the *model* is not cached. Confusing the two is the ADR-103 bug.
    """

    cache = _STATE["policies"]
    if path not in cache:
        with open(path, "rb") as handle:
            cache[path] = _STATE["cd"].decode_policy(handle.read(), context=os.path.basename(path))
    return cache[path]


def _play(request: dict[str, Any]) -> dict[str, Any]:
    """One episode. Model reloaded here, deliberately (ADR-103 §9)."""

    cd = _STATE["cd"]
    try:
        task = _STATE["variants"][request["variant"]]
        container = _policy(request["policy"])
        header, weights = container["header"], container["weights"]
        model = cd.load_model(_STATE["model_xml"])  # ← per episode, on purpose

        def act(_step: int, observation: Any) -> Any:
            return cd.policy_forward(header, weights, observation)

        episode = cd.evaluate_episode(
            model, task, actions=act, seed=int(request["seed"])
        )
        row = summarise(cd, episode, task)
        row.update({
            "id": request["id"],
            "policy": request["policy"],
            "variant": request["variant"],
        })
        return row
    except Exception as exc:  # noqa: BLE001 — one bad episode must not kill a sweep
        return {
            "schema": ROW_SCHEMA,
            "row": "error",
            "id": request.get("id"),
            "policy": request.get("policy"),
            "variant": request.get("variant"),
            "seed": request.get("seed"),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-4000:],
        }


def _verify(spec: dict[str, Any], cd: Any, task: dict[str, Any]) -> list[str]:
    """``verify_policy`` per container. Returns the paths that failed.

    A container that fails is **reported and skipped**, never silently
    dropped: a table that is quietly missing a row reads as a table of
    everything there was.
    """

    failed: list[str] = []
    for path in spec.get("verify") or []:
        try:
            with open(path, "rb") as handle:
                container = cd.decode_policy(handle.read(), context=os.path.basename(path))
            detail = cd.verify_policy(
                container, task,
                task_sha256=spec["task_sha256"],
                context=os.path.basename(path),
            )
            emit({
                "schema": ROW_SCHEMA, "row": "verify", "policy": path, "ok": True,
                "detail": {
                    key: detail.get(key)
                    for key in ("parameters", "witness_error", "witness_tolerance",
                                "witness_samples", "action_count", "layers")
                },
            })
        except Exception as exc:  # noqa: BLE001
            failed.append(path)
            emit({
                "schema": ROW_SCHEMA, "row": "verify", "policy": path, "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            })
    return failed


def main() -> int:
    spec = json.loads(sys.stdin.read())
    if spec.get("schema") != JOB_SCHEMA:
        emit({"row": "error", "error": f"unknown job schema {spec.get('schema')!r}"})
        return 2

    sys.path.insert(0, spec["module_dir"])
    import CadexDynamics as cd  # noqa: PLC0415

    with open(spec["task_path"], "rb") as handle:
        task = json.loads(handle.read().decode("utf-8"))

    failed = _verify(spec, cd, task)
    requests = [
        request for request in spec.get("episodes") or []
        if request["policy"] not in failed
    ]

    jobs = int(spec.get("jobs") or 0) or max(1, (os.cpu_count() or 2) - 2)
    jobs = max(1, min(jobs, len(requests) or 1))

    if requests:
        context = multiprocessing.get_context("fork")
        with context.Pool(jobs, initializer=_init_worker, initargs=(spec,)) as pool:
            # chunksize 1: episodes vary from 30 to 600 steps and a static
            # chunking leaves workers idle at the tail of a sweep.
            for row in pool.imap_unordered(_play, requests, chunksize=1):
                emit(row)

    emit({
        "schema": ROW_SCHEMA, "row": "done",
        "episodes": len(requests), "jobs": jobs,
        "skipped_policies": failed,
        "mujoco_version": getattr(cd._mujoco_module(), "__version__", ""),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
