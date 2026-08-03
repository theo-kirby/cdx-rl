"""Read a training run directory into a series, an inventory and a liveness.

**Both sources are needed and neither is sufficient**, which is the whole
reason this file exists rather than a one-line ``json.load``.

``progress.json``
    The trainer **rewrites it every iteration**, so it holds the *current*
    point and nothing before it. What only it has: the ``checkpoints`` list
    with a per-checkpoint ``reward_per_step``, ``sha256`` and ``bytes``, plus
    ``state``, ``device``, ``wall_time_s``, ``best_iteration`` and
    ``best_reward_per_step``.

``train.log``
    The **actual curve** — one line per iteration, 2 500 of them for the run
    experiment 001 is about — plus the ``witness agrees to …`` lines and, at
    the end of a run that finished, a terminal JSON blob carrying
    ``task_sha256``, ``model_sha256``, ``witness_error`` and ``parameters``.
    What it does not have: any checkpoint digest.

So: the log is the history, the progress file is the state and the manifest.

**The formats move, and the parser must not.** The two ``job-task-*`` runs on
this box were written by an older trainer whose iteration lines carry no
``episode`` and no ``sigma`` at all::

    iteration    0  reward/step +0.29999  loss +2.31516                    ← old
    iteration    0  reward/step -0.941203  loss +134.64  episode 85.3  sigma 0.4002

and two of the ``stand-task-*`` runs have ``episode_steps: null`` in their
progress file for the same reason. Every field after ``loss`` is therefore
optional here, and a missing one reads as ``None`` rather than as zero — a
run whose episode length is unknown must not be reported as a run whose
episode length was nought.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

#: ``iteration N reward/step X loss Y [episode Z] [sigma S]``. Everything
#: after ``loss`` is optional; see the module docstring.
ITERATION_RE = re.compile(
    r"^iteration\s+(?P<iteration>\d+)\s+"
    r"reward/step\s+(?P<reward>[-+]?[\d.]+(?:[eE][-+]?\d+)?)\s+"
    r"loss\s+(?P<loss>[-+]?[\d.]+(?:[eE][-+]?\d+)?)"
    r"(?:\s+episode\s+(?P<episode>[\d.]+(?:[eE][-+]?\d+)?))?"
    r"(?:\s+sigma\s+(?P<sigma>[\d.]+(?:[eE][-+]?\d+)?))?"
)

#: ``witness agrees to 2.820e-07 (355x inside the engine's tolerance)``.
#: The margin is on the trainer's stderr rather than in ``progress.json``,
#: which is why the log is tailed for it at all.
#:
#: **The comma group is load-bearing.** The trainer thousands-separates the
#: factor, so a very good margin prints as ``1,141x`` — and a pattern of
#: ``[\d.]+`` matches only margins under a thousand. Experiment 000's run
#: reported exactly that, and the first version of this regex silently
#: reported "no witness margin recorded" for a run whose margin was eleven
#: times the floor. A check that quietly finds nothing is worse than no
#: check (``DESIGN.md`` §6), and this one was caught only because the run it
#: was pointed at happened to be very clean.
WITNESS_RE = re.compile(
    r"witness agrees to\s+(?P<error>[\d.]+(?:[eE][-+]?\d+)?)\s*"
    r"\((?P<factor>[\d,.]+)x inside"
)

#: Hazard 13's floor. Under this the witness is recording what the GPU
#: rounded the network to rather than what the network is, and the right
#: response is to stop and read the hazard rather than to keep training.
WITNESS_FLOOR = 100.0

#: A ``state`` that means the trainer is no longer expected to write.
TERMINAL_STATES = {"done", "error", "stopped", "failed", "cancelled"}


def read_progress(run_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """``progress.json``, or ``{}`` if there is none.

    Absent is a legitimate state — a run dispatched thirty seconds ago has
    not written one yet — so this returns empty rather than raising, and the
    caller reports the absence.
    """

    path = Path(str(run_dir)) / "progress.json"
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # A half-written progress file is what you get by reading the one
        # file the trainer rewrites every iteration at the wrong moment.
        # It is not a corrupt run.
        return {}
    return loaded if isinstance(loaded, dict) else {}


def parse_log(run_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """The curve, the witness margins, and the terminal blob if there is one."""

    path = Path(str(run_dir)) / "train.log"
    result: dict[str, Any] = {
        "path": str(path), "exists": path.is_file(),
        "series": [], "witness": [], "terminal": {}, "warnings": [],
    }
    if not path.is_file():
        return result

    series: list[dict[str, Any]] = []
    witness: list[dict[str, Any]] = []
    terminal: dict[str, Any] = {}
    warnings: list[str] = []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            match = ITERATION_RE.match(line)
            if match:
                point = match.groupdict()
                series.append({
                    "iteration": int(point["iteration"]),
                    "reward_per_step": float(point["reward"]),
                    "loss": float(point["loss"]),
                    "episode_steps": (
                        float(point["episode"]) if point["episode"] else None
                    ),
                    "sigma": float(point["sigma"]) if point["sigma"] else None,
                })
                continue
            found = WITNESS_RE.search(line)
            if found:
                witness.append({
                    "error": float(found.group("error")),
                    "factor": float(found.group("factor").replace(",", "")),
                })
                continue
            stripped = line.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    blob = json.loads(stripped)
                except ValueError:
                    continue
                if isinstance(blob, dict) and "task_sha256" in blob:
                    terminal = blob
            elif stripped and (
                "Traceback" in stripped or "Error" in stripped
                or stripped.startswith("Failed to import")
            ):
                warnings.append(stripped[:300])

    result.update({
        "series": series, "witness": witness, "terminal": terminal,
        "warnings": warnings[:40],
    })
    return result


def liveness(run_dir: str | os.PathLike[str], progress: dict[str, Any]) -> dict[str, Any]:
    """Is anything still writing here?

    **The check exists because of one directory.**
    ``job-task-20260801-155047/progress.json`` reads ``state: "training"`` at
    iteration 748 of 4000, and its process died 33 hours ago. Without a
    liveness check a stale progress file is *indistinguishable* from a live
    one — same schema, same fields, a plausible iteration count — and a
    supervisor attached to it waits forever for a number that will never
    change.

    Three independent signals, all reported:

    * ``train.pid`` — is that process alive? ``kill(pid, 0)`` rather than
      parsing ``ps``. A pid can of course have been recycled onto some other
      process, so this is evidence and not proof, which is why the other two
      are printed beside it rather than instead of it.
    * ``progress.json``'s mtime — how long since anything wrote.
    * ``state`` — what the trainer last said about itself.
    """

    root = Path(str(run_dir))
    facts: dict[str, Any] = {
        "state": str(progress.get("state") or ""),
        "iteration": progress.get("iteration"),
        "total": progress.get("total"),
        "pid": None,
        "pid_alive": None,
        "progress_mtime": None,
        "progress_age_s": None,
    }

    pid_file = root / "train.pid"
    if pid_file.is_file():
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            pid = 0
        if pid > 0:
            facts["pid"] = pid
            try:
                os.kill(pid, 0)
                facts["pid_alive"] = True
            except ProcessLookupError:
                facts["pid_alive"] = False
            except PermissionError:
                # Alive, owned by somebody else. Still alive.
                facts["pid_alive"] = True

    progress_file = root / "progress.json"
    if progress_file.is_file():
        mtime = progress_file.stat().st_mtime
        facts["progress_mtime"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(mtime)
        )
        facts["progress_age_s"] = round(time.time() - mtime, 1)

    terminal = facts["state"] in TERMINAL_STATES
    facts["terminal"] = terminal
    # Stale: it claims to be running and nothing is running it. That claim
    # is the failure mode; a finished run with a dead process is just a
    # finished run.
    facts["stale"] = bool(
        facts["state"] and not terminal and facts["pid_alive"] is False
    )
    facts["live"] = bool(not terminal and facts["pid_alive"] is True)
    return facts


def checkpoint_inventory(
    run_dir: str | os.PathLike[str], progress: dict[str, Any], *, verify: bool = True
) -> list[dict[str, Any]]:
    """Every checkpoint the run declared, with its digest checked on disk.

    ``verify`` re-hashes the files. It is 15 MB of reading for the longest
    run on this box and it is a check that can fail, which is the bar
    ``DESIGN.md`` §6 sets: a green light from a check that computes nothing
    is worse than no check.

    **``best`` is a path, not a checkpoint, and the list says so badly.** The
    trainer appends a record every time the best-so-far improves, and every
    one of those records names the *same* file — ``<label>.best.cxpolicy`` —
    which it has just overwritten. ``stand-task-20260802-200109`` has eleven
    ``best`` records and one ``best`` file. So only the **last** record for
    any given path can be expected to match what is on disk; the earlier ten
    describe bytes that no longer exist anywhere.

    That is ``superseded``, and it is emphatically not ``MISMATCH``. Getting
    this wrong would print ten alarming rows about a healthy run, which is
    the fastest way to teach a reader to skip the column.
    """

    from harness.provenance import sha256_file

    root = Path(str(run_dir))
    records = [
        record for record in progress.get("checkpoints") or []
        if isinstance(record, dict)
    ]
    #: The index of the last record naming each path — the only one whose
    #: digest the file on disk can still be.
    latest: dict[str, int] = {}
    for index, record in enumerate(records):
        latest[str(record.get("path") or "")] = index

    digests: dict[str, str] = {}
    inventory: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        relative = str(record.get("path") or "")
        path = root / relative
        superseded = latest.get(relative) != index
        item: dict[str, Any] = {
            "iteration": record.get("iteration"),
            "tag": record.get("tag"),
            "path": str(path),
            "name": path.name,
            "reward_per_step": record.get("reward_per_step"),
            "declared_sha256": record.get("sha256"),
            "declared_bytes": record.get("bytes"),
            "exists": path.is_file(),
            "superseded": superseded,
        }
        if item["exists"] and verify:
            if str(path) not in digests:
                digests[str(path)] = sha256_file(path)
            item["observed_sha256"] = digests[str(path)]
            item["observed_bytes"] = path.stat().st_size
            item["digest_matches"] = (
                None if superseded
                else item["observed_sha256"] == str(record.get("sha256") or "")
            )
        inventory.append(item)
    return inventory


def _extremum(series: list[dict[str, Any]], key: str, *, largest: bool = True
              ) -> dict[str, Any] | None:
    points = [
        point for point in series
        if point.get(key) is not None and not point.get("over_budget")
    ]
    if not points:
        return None
    return (max if largest else min)(points, key=lambda point: point[key])


def task_budget(run_dir: str | os.PathLike[str]) -> int | None:
    """``episode.max_steps`` from the run's own bundle, if one is beside it.

    Needed for one thing, and it is not decoration: **four of the eight runs
    on this box log an episode length at iteration 0 that exceeds their own
    horizon** — 1343.0, 1365.3, 1280.0 and 1412.4 steps against a 600-step
    budget, one point each, always the first. Whatever that first statistic
    is, it is not a mean episode length: an episode cannot run longer than
    the horizon it is truncated at.

    Left in, it wins the ``best episode length`` search outright and the
    report announces that every one of those runs peaked at iteration 0 —
    a confident, prominent, wrong headline. So points above the budget are
    excluded from the peak search and **counted**, and the reporter prints
    the count. Dropping them quietly would be the same bug with better
    manners.
    """

    root = Path(str(run_dir))
    for bundle in sorted(root.glob("*-task.json")):
        try:
            loaded = json.loads(bundle.read_text(encoding="utf-8"))
            return int(loaded["episode"]["max_steps"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


def load_run(run_dir: str | os.PathLike[str], *, verify_checkpoints: bool = True
             ) -> dict[str, Any]:
    """Everything :mod:`harness.supervise` needs about one run directory.

    The peaks are computed here rather than in the reporter because two of
    them are the point of the whole exercise: ``best_reward`` is what the
    trainer optimises and ``best_episode`` is what survival looks like from
    the log, and the run this repository was built to explain has them 1 200
    iterations apart.
    """

    root = Path(str(run_dir)).expanduser().resolve()
    progress = read_progress(root)
    log = parse_log(root)
    series = log["series"]

    budget = task_budget(root)
    over_budget = []
    if budget:
        for point in series:
            length = point.get("episode_steps")
            if length is not None and length > budget:
                point["over_budget"] = True
                over_budget.append(point["iteration"])

    best_reward = _extremum(series, "reward_per_step", largest=True)
    best_episode = _extremum(series, "episode_steps", largest=True)
    final = series[-1] if series else None

    bundles = sorted(str(p.name) for p in root.glob("*-task.json"))
    models = sorted(str(p.name) for p in root.glob("*-model.xml"))

    return {
        "run_dir": str(root),
        "label": str(progress.get("label") or "") or root.name,
        "progress": progress,
        "log": log,
        "series": series,
        "best_reward": best_reward,
        "best_episode": best_episode,
        "final": final,
        "max_steps": budget,
        "over_budget_iterations": over_budget,
        "liveness": liveness(root, progress),
        "checkpoints": checkpoint_inventory(root, progress, verify=verify_checkpoints),
        "bundle": bundles[0] if bundles else "",
        "model": models[0] if models else "",
    }
