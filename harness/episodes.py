"""Shared plumbing for the two drivers that play policies.

``compare`` and ``capability`` ask the same three questions of the same
machinery — *where is the bundle, where is the model, and what do a hundred
episode rows add up to* — so the answers live here once rather than twice
with a slow divergence between them.

Nothing in this module imports ``mujoco``; it builds job specs for
``harness/_episodes.py`` and folds up the rows that come back.
"""

from __future__ import annotations

import json
import re
import statistics
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from harness.provenance import sha256_file

#: ``stand8.000600.cxpolicy`` → ``000600``. The tag is the *count* of
#: iterations at the write, so the iteration index it names is one less —
#: ``progress.json`` records ``{"iteration": 599, "tag": "000600"}``. That
#: off-by-one is why the checkpoint list is preferred over the filename
#: whenever there is one: a row that quoted iteration 600 against a curve
#: point at 599 would be quietly out by one everywhere.
TAG_RE = re.compile(r"\.(\d{4,})\.cxpolicy$")

#: ``<label>.NNNNNN.cxpolicy``, exactly six digits. Stricter than
#: :data:`TAG_RE` on purpose: ``<label>.best.cxpolicy`` and the final
#: ``<label>.cxpolicy`` carry no iteration and are not part of a *series*.
PERIODIC_RE = re.compile(r"\.(\d{6})\.cxpolicy$")


class BundleError(RuntimeError):
    """The task bundle, the model or the policies do not line up."""


def load_bundle(task_path: str | Path) -> tuple[dict[str, Any], str]:
    """The bundle and its sha256.

    The digest is what ``verify_policy`` checks a container's claim against,
    so it is computed from the *file the caller named* rather than from any
    copy that happens to be in a project store.
    """

    path = Path(str(task_path)).expanduser().resolve()
    if not path.is_file():
        raise BundleError(
            f"No task bundle at {path}. --task is mandatory and means the "
            "run's own bundle: a rebuild is keyed by script digest and "
            "replaces script_artifacts/, so a finished run's task can vanish "
            "from the project while its checkpoints sit beside you."
        )
    try:
        bundle = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise BundleError(f"{path} is not readable JSON: {exc}") from exc
    if not isinstance(bundle, dict) or "episode" not in bundle:
        raise BundleError(f"{path} is not a cadex-training-task bundle.")
    return bundle, sha256_file(path)


def resolve_model(bundle: dict[str, Any], search: Iterable[Path]) -> Path:
    """The MJCF the bundle was built against, by **digest**, not by name.

    The bundle records ``model.sha256``; the run directory holds an XML. If
    they disagree, every episode played would be an episode of a different
    mechanism, and nothing downstream would say so — the policy would still
    verify against the bundle, the numbers would still be numbers. So the
    match is on the digest and a mismatch is refused here.
    """

    declared = str((bundle.get("model") or {}).get("sha256") or "")
    candidates = [path for path in search if path.is_file()]
    if not candidates:
        raise BundleError("No *-model.xml found beside the run or the bundle.")
    for path in candidates:
        if not declared or sha256_file(path) == declared:
            return path
    found = {path.name: sha256_file(path)[:12] for path in candidates}
    raise BundleError(
        f"No model beside these files digests to the bundle's own "
        f"model.sha256 {declared[:12]}…; found {found}. Playing a policy "
        "against a model the bundle was not built from measures a different "
        "mechanism and nothing downstream would notice."
    )


def checkpoint_index(run_dir: Path) -> dict[str, dict[str, Any]]:
    """``{filename: {iteration, trainer_reward_per_step}}`` from ``progress.json``.

    Preferred over the filename tag because it is the trainer's own index,
    and because the ``best`` file carries no tag at all.

    The trainer's ``reward_per_step`` comes along because **it is the other
    half of ADR-099's claim**. That the trainer's scalar and real survival
    were anti-correlated across a run is a statement about two columns, and a
    table carrying only the measured one cannot restate it, confirm it, or
    contradict it.
    """

    path = run_dir / "progress.json"
    if not path.is_file():
        return {}
    try:
        progress = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    mapping: dict[str, dict[str, Any]] = {}
    for record in progress.get("checkpoints") or []:
        if isinstance(record, dict) and record.get("path") is not None:
            # Later records win: `best` names one file many times, and the
            # last claim is the one the bytes on disk belong to.
            mapping[str(record["path"])] = {
                "iteration": int(record.get("iteration", -1)),
                "trainer_reward_per_step": record.get("reward_per_step"),
            }
    final = str(progress.get("out") or "")
    if final and progress.get("iteration") is not None:
        mapping[Path(final).name] = {
            "iteration": int(progress["iteration"]),
            "trainer_reward_per_step": progress.get("reward_per_step"),
        }
    return mapping


def discover_policies(run_dir: Path) -> list[dict[str, Any]]:
    """Every ``.cxpolicy`` in a run directory, ordered by iteration.

    Ordered rather than globbed because the table is read as a *curve*, and
    ``stand8.best.cxpolicy`` sorting between ``000850`` and ``000900`` by
    name would put the reward peak in the wrong place on it.
    """

    known = checkpoint_index(run_dir)
    found: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("*.cxpolicy")):
        record = known.get(path.name) or {}
        iteration = record.get("iteration")
        if iteration is None:
            tag = TAG_RE.search(path.name)
            iteration = int(tag.group(1)) - 1 if tag else -1
        role = "best" if ".best." in path.name else (
            "final" if TAG_RE.search(path.name) is None else "checkpoint"
        )
        found.append({
            "name": path.name, "path": str(path),
            "iteration": iteration, "role": role,
            "trainer_reward_per_step": record.get("trainer_reward_per_step"),
            "bytes": path.stat().st_size,
        })
    found.sort(key=lambda item: (item["iteration"], item["name"]))
    return found


def series_checkpoints(run_dir: Path, stride: int) -> list[tuple[int, Path]]:
    """Periodic checkpoints at ``stride``, plus the last one written.

    The last is always included whatever its iteration: it is the newest
    evidence about where the run is heading, and dropping it because 1750 is
    not a multiple of 250 would silently truncate the trend at its most
    informative end.

    ``stride <= 0`` keeps every periodic checkpoint, which is how a caller
    that wants to *select* one by iteration rather than plot a trend asks for
    the full list.

    **This lived in ``mechanisms/mg-legs/drivers/hazard15.py`` and moved here
    when ``capture`` became its second caller.** It is pure — a glob, a
    regular expression and a sort — and it stays in the module cdx-rl's own
    venv can import, with ``hazard15`` re-exporting it under the trainer
    interpreter. A third copy of this glob is exactly the drift
    ``harness/_stats.py`` was created to stop.
    """

    found: list[tuple[int, Path]] = []
    for path in sorted(Path(run_dir).glob("*.cxpolicy")):
        match = PERIODIC_RE.search(path.name)
        if match:
            found.append((int(match.group(1)), path))
    if not found:
        raise BundleError(f"no periodic checkpoints under {run_dir}")
    found.sort()
    keep = {iteration: path for iteration, path in found
            if stride <= 0 or iteration % stride == 0}
    keep[found[-1][0]] = found[-1][1]
    return sorted(keep.items())


# ---------------------------------------------------------------------------
# Folding rows into a row
# ---------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float | None:
    clean = [value for value in values if value is not None]
    return statistics.fmean(clean) if clean else None


def _median(values: Sequence[float]) -> float | None:
    clean = [value for value in values if value is not None]
    return statistics.median(clean) if clean else None


def aggregate(rows: list[dict[str, Any]], bundle: dict[str, Any]) -> dict[str, Any]:
    """A group of episode rows, as one line of a table.

    ``survival`` is ``truncated`` — the episode used its whole budget — and
    not "did not report a termination label". The two are the same today and
    the first is what the bundle means by surviving.
    """

    actions = bundle["actions"]
    motors = len(actions)
    limits = [max(abs(float(a["low"])), abs(float(a["high"]))) for a in actions]
    max_steps = int(bundle["episode"]["max_steps"])

    survived = [row for row in rows if row["truncated"]]
    mix: dict[str, int] = {}
    for row in rows:
        mix[row["termination"]] = mix.get(row["termination"], 0) + 1

    # Torque, per motor, across the group. Peak is a max of maxima — the
    # worst instant any episode saw. Mean and saturation are means of the
    # per-episode figures, so a short episode and a long one count equally,
    # which is the right weighting for "what does this policy do".
    peak = [0.0] * motors
    mean_torque: list[list[float]] = [[] for _ in range(motors)]
    saturation: list[list[float]] = [[] for _ in range(motors)]
    for row in rows:
        for index in range(min(motors, len(row["peak_torque_nmm"]))):
            peak[index] = max(peak[index], float(row["peak_torque_nmm"][index]))
            mean_torque[index].append(float(row["mean_torque_nmm"][index]))
            saturation[index].append(float(row["frac_above_90pct"][index]))

    per_motor_mean = [_mean(values) or 0.0 for values in mean_torque]
    per_motor_sat = [_mean(values) or 0.0 for values in saturation]

    tilt_values = []
    for row in rows:
        for term in row["termination_values"]:
            if term["label"] == "tipped" and term["value"] is not None:
                tilt_values.append(float(term["value"]))

    return {
        "episodes": len(rows),
        "survival": len(survived) / len(rows) if rows else 0.0,
        "survived": len(survived),
        "max_steps": max_steps,
        "steps_mean": _mean([row["step_count"] for row in rows]),
        "steps_median": _median([row["step_count"] for row in rows]),
        "steps_min": min((row["step_count"] for row in rows), default=None),
        "steps_max": max((row["step_count"] for row in rows), default=None),
        "reward_mean": _mean([row["total_reward"] for row in rows]),
        "reward_per_step_mean": _mean([row["reward_per_step"] for row in rows]),
        "final_tipped_mean": _mean(tilt_values),
        "drift_mean": _mean([row["drift"] for row in rows]),
        "termination_mix": dict(sorted(mix.items())),
        "peak_torque_nmm": [round(value, 3) for value in peak],
        "mean_torque_nmm": [round(value, 3) for value in per_motor_mean],
        "frac_above_90pct": [round(value, 5) for value in per_motor_sat],
        "limit_nmm": limits,
        "peak_torque_worst": max(peak) if peak else 0.0,
        "peak_frac_of_limit": max(
            (value / limit if limit else 0.0 for value, limit in zip(peak, limits)),
            default=0.0,
        ),
        "mean_frac_of_limit": max(
            (value / limit if limit else 0.0
             for value, limit in zip(per_motor_mean, limits)),
            default=0.0,
        ),
        "saturation_worst": max(per_motor_sat) if per_motor_sat else 0.0,
    }


#: A policy commanding this little of its limit is not balancing (hazard 15's
#: opposite). 5 % of 86 N·mm is 4.3 N·mm — well below what holding this
#: mechanism upright takes.
UNTRAINED_TORQUE_FRACTION = 0.05

#: Hazard 15's thresholds. Either is enough to say the policy is bracing:
#: a quarter of frames pinned above 90 % of the rating, or a *mean* command
#: at half of it — 216 N·mm was a momentary number no real servo holds for
#: six seconds, and this is the column that catches its equivalent.
BRACING_SATURATION = 0.25
BRACING_MEAN_FRACTION = 0.50


def flags(row: dict[str, Any]) -> list[str]:
    """What this row says about itself that a reader would otherwise miss.

    Printed on the row, never in a footnote: the thing about hazard 15 is
    that the trajectory looks fine.
    """

    found: list[str] = []
    if row["peak_frac_of_limit"] < UNTRAINED_TORQUE_FRACTION:
        found.append(
            f"near-zero command (peak {row['peak_torque_worst']:.1f} N·mm, "
            f"{row['peak_frac_of_limit'] * 100:.1f}% of limit)"
            + (" with high survival — doing nothing, not balancing"
               if row["survival"] > 0.5 else "")
        )
    if row["saturation_worst"] >= BRACING_SATURATION:
        found.append(
            f"hazard 15: a motor is above 90% of its limit on "
            f"{row['saturation_worst'] * 100:.0f}% of frames"
        )
    if row["mean_frac_of_limit"] >= BRACING_MEAN_FRACTION:
        found.append(
            f"hazard 15: a motor's MEAN command is "
            f"{row['mean_frac_of_limit'] * 100:.0f}% of its limit"
        )
    return found
