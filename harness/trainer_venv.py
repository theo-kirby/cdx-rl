"""The bridge to the one interpreter allowed to import ``mujoco``.

**The structural constraint this file exists to serve.** cdx-rl's own venv
has no ``mujoco`` and no ``jax``, deliberately: a second mujoco pin would be
a second answer to "which simulator produced this number", and the whole
point of the recorded runs is that there is one answer. But ``CadexDynamics``
— the engine's own reference runner, the thing ADR-099's table came from —
needs ``mujoco``.

So episode evaluation happens in the **trainer venv, as a subprocess**::

    cdx-rl/.venv                       /home/theo/cadex-train-venv
    ┌──────────────────────────┐ spawn ┌────────────────────────────────┐
    │ harness/compare.py       │──────▶│ harness/_episodes.py           │
    │ harness/capability.py    │       │   sys.path += cadex/src/Mod/…  │
    │   job spec (JSON, stdin) │       │   import CadexDynamics         │
    │   rows (NDJSON, stdout) ◀─┼───────│   multiprocessing.Pool(N)      │
    └──────────────────────────┘       └────────────────────────────────┘

``_episodes.py`` is the only file in this repository that imports
``CadexDynamics``. It is *executed*, never imported, by cdx-rl code — so the
LGPL boundary is a process boundary rather than a link, and cdx-rl's venv
stays small. Parallelism lives on the far side, where the interpreter is
already the right one.

Why NDJSON rather than one JSON document at the end: a sweep of 600 episodes
takes a couple of minutes and a caller that has to wait for the last one
before seeing the first cannot report progress, and cannot tell a slow job
from a wedged one. One row per episode, flushed, answers both.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(REPO_ROOT / "tools"))

from smoke import EXPECTED_TRAINER_PINS  # noqa: E402

#: The job spec and row schemas. Bumped together; ``_episodes.py`` checks the
#: request's and stamps its replies with the reply one.
JOB_SCHEMA = "cdxrl-episode-job-v1"
ROW_SCHEMA = "cdxrl-episode-row-v1"

EPISODES_SCRIPT = Path(__file__).resolve().parent / "_episodes.py"

#: The other far side. ``steps`` needs MuJoCo's contact list per control
#: step, which no summary carries, so it has its own trainer-side script
#: rather than widening ``_episodes.py``'s row schema for one caller.
STEPS_SCRIPT = Path(__file__).resolve().parent / "_steps.py"

#: A pin probe that does **not** import jax. ``importlib.metadata`` reads the
#: installed distributions' metadata off disk; importing jax initialises CUDA
#: and costs several seconds, which is a lot to pay before every sweep for a
#: fact that is written in a file.
_PIN_PROBE = r"""
import json, sys
import importlib.metadata as md
out = {"python": ".".join(str(n) for n in sys.version_info[:3])}
for name in ("mujoco", "mujoco-mjx", "jax"):
    try:
        out[name] = md.version(name)
    except Exception as exc:
        out[name] = "missing: %s" % exc
print(json.dumps(out))
"""


class TrainerVenvError(RuntimeError):
    """The trainer interpreter is absent, mispinned, or would not run."""


class EpisodeJobFailed(RuntimeError):
    """``_episodes.py`` refused the job or died part-way through it."""


def trainer_python(venv: str | os.PathLike[str] | None = None) -> Path:
    """The interpreter every episode runs under.

    ``CADEX_TRAIN_VENV``, or the box default. Never created, never repaired:
    its pins are what makes the recorded runs reproducible and rebuilding it
    would silently invalidate every number already in the graph.
    """

    root = Path(str(venv or os.environ.get("CADEX_TRAIN_VENV", "/home/theo/cadex-train-venv")))
    interpreter = root / "bin" / "python"
    if not interpreter.is_file():
        raise TrainerVenvError(
            f"No trainer interpreter at {interpreter}. Set CADEX_TRAIN_VENV, "
            "or see config/env.example. Do not create one: the pins in the "
            "existing venv are the reproducibility guarantee for every run "
            "already recorded."
        )
    return interpreter


_pins_checked: dict[str, dict[str, str]] = {}


#: Pin drift that an evaluation run acknowledged rather than failed on.
#:
#: Populated only by ``check_pins(strict=False)``, and every driver that
#: allows it must copy this into its envelope. See ``UNPINNED_ENV``.
unpinned_drift: dict[str, dict[str, Any]] = {}

#: The one environment variable that turns the pin check into a warning.
#:
#: **It exists because there are three boxes now and only two of them train.**
#: `mg-legs` is authored on a macOS laptop that has never had — and cannot
#: usefully be given — the Linux trainer venv's exact pins, and `method.md`
#: §10 says checkpoint comparison runs *locally*, in stock MuJoCo, in
#: seconds. Refusing to evaluate anything on the authoring box makes the
#: cheapest instrument in the method unavailable on the machine where the
#: mechanism is designed.
#:
#: **It never applies to training.** ``tools/train.py`` does not consult it;
#: a run dispatched against unpinned versions would be comparable to nothing
#: and the graph would not say so. This only lets a driver *play* a policy.
#:
#: The drift is not hidden when it is allowed: it is printed on stderr on
#: every invocation and it is written into the driver's JSON envelope, so a
#: number produced this way carries the reason it might differ.
UNPINNED_ENV = "CDXRL_ALLOW_UNPINNED_EVAL"


def check_pins(venv: str | os.PathLike[str] | None = None, *,
               strict: bool | None = None) -> dict[str, str]:
    """Assert the trainer venv is *the* trainer venv. Returns the versions.

    Loud rather than advisory. An evaluator running against mujoco 3.9 and
    reporting a survival number beside runs trained on 3.10 is producing a
    comparison nobody asked for and nothing in the output would say so.

    ``strict`` defaults to *not* ``CDXRL_ALLOW_UNPINNED_EVAL``. When it is
    false a mismatch warns, records itself in :data:`unpinned_drift`, and
    proceeds — see :data:`UNPINNED_ENV` for when that is legitimate and when
    it is not.

    Cached per process — the answer cannot change under us mid-sweep, and a
    subprocess spawn per driver invocation is enough.
    """

    if strict is None:
        strict = os.environ.get(UNPINNED_ENV, "").strip().lower() not in (
            "1", "true", "yes", "on")

    interpreter = trainer_python(venv)
    key = str(interpreter)
    if key in _pins_checked:
        return _pins_checked[key]
    try:
        completed = subprocess.run(
            [key, "-c", _PIN_PROBE],
            capture_output=True, text=True, timeout=120, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise TrainerVenvError(f"Could not probe {interpreter}: {exc}") from exc
    if completed.returncode != 0:
        raise TrainerVenvError(
            f"{interpreter} could not report its versions:\n"
            + (completed.stderr or completed.stdout)[-2000:]
        )
    try:
        found = json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise TrainerVenvError(
            f"{interpreter} produced no readable probe output: "
            f"{completed.stdout[-500:]!r}"
        ) from exc

    wrong = {
        name: {"found": found.get(name), "expected": expected}
        for name, expected in EXPECTED_TRAINER_PINS.items()
        if found.get(name) != expected
    }
    if wrong:
        message = (
            f"{interpreter} is not the pinned trainer venv: "
            + json.dumps(wrong, sort_keys=True)
            + "\nEvery number in the graph was measured against "
            + json.dumps(EXPECTED_TRAINER_PINS, sort_keys=True)
            + ". Do not rebuild the venv to fix this — find out which venv "
            "this is."
        )
        if strict:
            raise TrainerVenvError(message)
        unpinned_drift[key] = wrong
        print(f"WARNING: {message}\n"
              f"Proceeding because {UNPINNED_ENV} is set. This is an "
              f"EVALUATION-only escape:\nthe drift is recorded in this "
              f"driver's envelope and any number it produces carries it.",
              file=sys.stderr)
    _pins_checked[key] = found
    return found


def cadex_module_dir() -> Path:
    """``<CADEX_REPO>/src/Mod/cadex`` — where ``CadexDynamics`` lives.

    The *checkout*, always, and not a staged payload: the payload on this box
    has no ``CadexDynamics.py`` at all. ``_episodes.py`` puts this on its
    ``sys.path`` and imports one module out of it.
    """

    repo = Path(os.environ.get("CADEX_REPO", "/home/theo/cadex"))
    module_dir = repo / "src" / "Mod" / "cadex"
    if not (module_dir / "CadexDynamics.py").is_file():
        raise TrainerVenvError(
            f"{module_dir} has no CadexDynamics.py. CADEX_REPO must name the "
            "Cadex *checkout*; the staged engine payload predates the whole "
            "MuJoCo surface and does not carry this module."
        )
    return module_dir


def run_episode_job(
    job: Mapping[str, Any],
    *,
    on_row: Callable[[dict[str, Any]], None] | None = None,
    venv: str | os.PathLike[str] | None = None,
    timeout: float = 7200.0,
) -> list[dict[str, Any]]:
    """``run_bridge_job`` against ``_episodes.py``. The original entry point."""

    return run_bridge_job(EPISODES_SCRIPT, job, schema=JOB_SCHEMA,
                          on_row=on_row, venv=venv, timeout=timeout)


def run_bridge_job(
    script: Path,
    job: Mapping[str, Any],
    *,
    schema: str = JOB_SCHEMA,
    on_row: Callable[[dict[str, Any]], None] | None = None,
    venv: str | os.PathLike[str] | None = None,
    timeout: float = 7200.0,
) -> list[dict[str, Any]]:
    """Run one job under the trainer interpreter; return its rows.

    ``script`` is the far side — ``_episodes.py`` or ``_steps.py``. Both read
    one JSON spec on stdin and write NDJSON on stdout, and both are executed
    as subprocesses and never imported, which is what keeps cdx-rl's own venv
    free of a second mujoco pin and the LGPL boundary a process boundary.

    ``on_row`` is called with each row as it arrives, which is how a driver
    prints progress over a sweep that takes minutes.

    Anything the far side writes to stderr is kept and attached to the
    exception if it fails, because a traceback from ``CadexDynamics`` is the
    most useful thing in the room when a bundle and a policy disagree.
    """

    check_pins(venv)
    interpreter = trainer_python(venv)
    spec = {
        "schema": schema,
        "module_dir": str(cadex_module_dir()),
        **dict(job),
    }
    try:
        process = subprocess.Popen(
            [str(interpreter), str(script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # The far side must not inherit a JAX_PLATFORMS or an MUJOCO_GL
            # from a caller that set one for the trainer: episode evaluation
            # is stock MuJoCo on the CPU and opens no renderer.
            env={k: v for k, v in os.environ.items() if k not in ("MUJOCO_GL",)},
        )
    except OSError as exc:
        raise TrainerVenvError(f"Could not spawn {script}: {exc}") from exc

    rows: list[dict[str, Any]] = []
    try:
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(json.dumps(spec) + "\n")
        process.stdin.flush()
        process.stdin.close()
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                # Not a row. Keep it visible rather than swallowing it: an
                # unparseable line here is a warning from a library that
                # decided to print, and it belongs in the record.
                row = {"row": "noise", "text": line}
            rows.append(row)
            if on_row is not None:
                on_row(row)
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        raise EpisodeJobFailed(
            f"The episode job did not finish within {timeout:g}s."
        ) from None
    finally:
        stderr = ""
        if process.stderr is not None:
            stderr = process.stderr.read()
            process.stderr.close()

    if process.returncode != 0:
        raise EpisodeJobFailed(
            f"{script.name} exited {process.returncode}.\n"
            f"stderr:\n{stderr[-4000:]}"
        )
    failures = [row for row in rows if row.get("row") == "error"]
    if failures:
        raise EpisodeJobFailed(
            f"{script.name} reported errors: "
            + json.dumps(failures[:5], sort_keys=True)
            + (f"\nstderr:\n{stderr[-2000:]}" if stderr.strip() else "")
        )
    return rows
