#!/usr/bin/env python
"""Dispatch ``cadex_train.py`` into a run directory, under supervision.

```
uv run python tools/train.py --bundle PATH --label NAME
       [--iterations N] [--envs N] [--checkpoint-every N]
       [--seed N | --seeds N [N …]] [--detach]
       [--cpu] [--supervise] [--patience N] [--require-device gpu]
```

**Why this exists rather than a shell line.** ``remote_train.sh`` dispatches
to another box and owns the run-directory convention; nothing on sb1x
dispatches *locally* in a shape ``supervise`` can attach to. This makes the
run directory, puts the bundle and the model in it, starts the trainer, and
writes the ``train.pid`` that liveness checking depends on.

The directory it builds is deliberately the same shape as the eight in
``/home/theo/cadex-jobs``:

```
jobs/<label>-<YYYYMMDD>-<HHMMSS>/
    <name>-task.json      the bundle, copied
    <name>-model.xml      the MJCF, copied beside it
    <label>.cxpolicy      the final policy
    <label>.NNNNNN.cxpolicy   checkpoints
    progress.json         rewritten every iteration
    train.log             stdout and stderr, merged
    train.pid             so supervise can tell live from stale
```

**The model is copied, and that is load-bearing.** ``cadex_train.py`` resolves
``model.path`` relative to the bundle's *grandparent* first and its own
directory second, then **checks the digest** against what the bundle
recorded. A run directory that carries the bundle without the model does not
fail at dispatch; it fails after the trainer has started, which on a GPU run
is a wasted allocation.

**`--cpu` is real and costs nothing.** There is no CPU guard in
``cadex_train.py`` — ``--allow-cpu`` belongs to ``remote_train.sh``, not to
the trainer — so ``JAX_PLATFORMS=cpu`` in the environment is all it takes.
That is what makes experiment 000 free.

### Every hyperparameter is passed, and the defaults are a *run*, not the trainer's

The first version of this file passed four flags — ``--iterations``,
``--envs``, ``--checkpoint-every``, ``--seed`` — and let the other ten fall
back to ``cadex_train.py``'s own defaults. **Six of those ten differ from what
``stand-task-20260802-200109`` actually ran**, recovered from that run's
``.cxpolicy`` header (``training.hyperparameters``):

===============  ==================  ==============
flag             trainer default     200109 used
===============  ==================  ==============
``envs``         256                 **2048**
``unroll``       20                  **40**
``epochs``       4                   **5**
``discount``     0.97                **0.995**
``gae_lambda``   0.95                **0.97**
``entropy``      1.0e-3              **2.0e-3**
``initial_std``  0.3                 **0.4**
===============  ==================  ==============

A replication sweep dispatched through the old signature would have run a
**different algorithm** from the run it exists to replicate, the seeds would
have been comparable to each other and to nothing else, and **no output would
have shown the confound** — the trainer records what it was given, not what
it was meant to be given. So every one of the fourteen is an explicit flag
here, each defaulting to :data:`RUN_200109`, and the resolved set is written
to ``hyperparameters.json`` in the run directory so section 6 of an
experiment README can be checked against what actually ran.

### `--detach` and `--seeds`

``method.md`` step 8 says *"Detached, always."* ``--detach`` re-execs this
file with ``start_new_session=True`` and returns immediately, so the sweep
outlives the terminal that started it.

``--seeds`` runs each seed **sequentially in one process** — ``cloud.md``:
*"One run at a time on this card"* — each into its own run directory, each
supervised in turn, with a ``sweep.json`` manifest updated after every run.
``--seed`` drives both ``jax.random.PRNGKey`` and the episode-variation draws
(``cadex_train.py:477``, ``random.Random(base_seed + environment_index)``),
so each seed is a fully independent replicate of the whole experiment rather
than a different initialisation of the same one.

### The runtime section, and why a dispatcher owns it

Three settings here are properties of the **box**, not of the experiment:
:data:`TRAINER_STACK_MB`, :data:`TRAINER_XLA_PREALLOCATE` and the launcher
shim. None changes a number the trainer computes. All decide whether the
process lives long enough to write one, and on sb9x at 2048 environments the
answer without them is no — in two ways that do not look like crashes:

* **during tracing**, before iteration 0, as a bare ``SIGSEGV`` with no
  traceback and a ``progress.json`` still reading ``state: starting``. The
  stack limit fixes this one outright; and
* **after a checkpoint write**, leaving a run directory holding every
  checkpoint, no final ``.cxpolicy``, no ``reward_curve`` and no witness
  margin — which is exactly the artefact a SIGTERMed run leaves, so it reads
  as "stopped early", not "crashed". This one is **still open**.

The second is the dangerous one, and it is why these live in the dispatcher
rather than in a shell line or a README. A launch recipe that has to be
remembered is the same shape as the hyperparameter confound above and the
unchecked ``trainer_sha256`` below it: the fact was known, it was written
down somewhere, and **nothing enforced it**. The resolved set is written to
``runtime.json`` beside ``hyperparameters.json``, and into ``sweep.json``
with the hostname, because "which box" is now a real axis — see
``tasks/stand-b2/README.md``.

**They are necessary and not sufficient.** With all three on, 2048
environments completes cleanly at 1 and 3 iterations and still exits ``-11``
at 40, as ``train()`` returns and before the final policy is written. That is
open; ``cloud.md`` §1 has the measurements. Do not read a clean short run as
proof — every one of these faults is scale-dependent, which is exactly how
the first version of this docstring came to claim more than it had measured,
and how it also came to blame a 12 GB card for a task that uses 777 MiB.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from harness import (  # noqa: E402
    EXIT_INFRASTRUCTURE, EXIT_OK, EXIT_SALVAGEABLE, EXIT_USAGE,
)
from harness.provenance import load_env_file, sha256_file  # noqa: E402
from harness.trainer_venv import TrainerVenvError, check_pins, trainer_python  # noqa: E402

#: What ``stand-task-20260802-200109`` actually ran, read out of its own
#: ``.cxpolicy`` header rather than assumed. These are the defaults for every
#: hyperparameter flag below, so that a run dispatched with no tuning flags
#: replicates that run by construction instead of by luck. See the module
#: docstring for the six that differ from ``cadex_train.py``'s own defaults.
#:
#: ``checkpoint_every`` is deliberately **not** here: 200109 used 50, and it
#: changes what is saved rather than what is computed.
RUN_200109: dict[str, Any] = {
    "envs": 2048,
    "unroll": 40,
    "epochs": 5,
    "hidden": [64, 64],
    "learning_rate": 3.0e-4,
    "discount": 0.995,
    "gae_lambda": 0.97,
    "clip": 0.2,
    "entropy": 2.0e-3,
    "value_weight": 0.5,
    "initial_std": 0.4,
}

#: The fourteen that describe a run, in the order the manifest prints them.
#: ``hidden`` is a list; everything else is a scalar.
HYPERPARAMETERS = (
    "seed", "iterations", "envs", "unroll", "epochs", "hidden",
    "learning_rate", "discount", "gae_lambda", "clip", "entropy",
    "value_weight", "initial_std", "checkpoint_every",
)

#: How much stack the *trainer* gets, in MiB.
#:
#: MJX traces deeply. ``make_constraint`` fans out over every contact pair
#: through nested ``vmap``, and on a 2048-environment biped that recursion
#: overflows an 8 MiB stack **during tracing**, before iteration 0 — a bare
#: ``SIGSEGV`` with no Python traceback and no line in ``progress.json``,
#: which reads like a broken card rather than a resource limit.
#:
#: 8 MiB is the Linux default, and the fix is not ``ulimit -s unlimited``:
#: glibc reads ``RLIMIT_STACK`` when it *creates a thread*, and for
#: ``RLIM_INFINITY`` it substitutes its own 8 MiB default. JAX traces on its
#: worker threads, so "unlimited" leaves the faulting thread exactly as it
#: was. The limit has to be a large **finite** number for a thread to inherit
#: it. Measured on sb9x: 8 MiB segfaults, 256 MiB traces clean.
TRAINER_STACK_MB = 256

#: XLA's BFC allocator grabs 75 % of the card up front. Growing the pool on
#: demand instead delays sb9x's post-checkpoint ``SIGSEGV`` from the *first*
#: checkpoint to the last, which is the difference between a run that
#: produces one checkpoint and a run that produces all of them.
#:
#: It is **not a fix**, and the reason first written here — that a 12 GB card
#: leaves too little outside the pool — was wrong. Peak device usage with
#: preallocation off is **777 MiB of 12 282**, six per cent of the card; the
#: 9 131 MiB seen with it on is the pool, not demand. Memory is not the
#: constraint. See ``cloud.md`` §1 for what is still open.
TRAINER_XLA_PREALLOCATE = "false"

#: How bad each exit code is, for picking a sweep's verdict from its seeds.
#:
#: Needed because the numbers are identifiers, not a scale:
#: ``EXIT_SALVAGEABLE`` is 4 and ``EXIT_INFRASTRUCTURE`` is 1, but a seed that
#: crashed *after* writing every checkpoint is a better outcome than one that
#: died importing jax. ``max()`` over the raw codes would report the whole
#: sweep as salvageable when one seed produced nothing at all.
SEVERITY: dict[int, int] = {
    EXIT_OK: 0,
    EXIT_SALVAGEABLE: 1,
    EXIT_INFRASTRUCTURE: 2,
    EXIT_USAGE: 3,
}

#: The shim that turns CPython's cyclic collector off for the trainer. A
#: collection landing mid-trace segfaults on this box; the artefacts survive
#: but the **exit code** does not, and that is the only signal a sweep has
#: for "did this seed work". Its docstring has the measurements.
TRAINER_LAUNCH = REPO_ROOT / "tools" / "trainer_launch.py"


def _stack_soft_mb() -> float | None:
    """This process's soft stack limit in MiB; ``None`` if unlimited."""

    soft, _hard = resource.getrlimit(resource.RLIMIT_STACK)
    return None if soft == resource.RLIM_INFINITY else soft / (1024 * 1024)


def _raise_child_stack(stack_mb: int):
    """A ``preexec_fn`` that gives the trainer ``stack_mb`` MiB of stack.

    Runs in the forked child, between ``fork`` and ``exec``, so it changes
    the limit for the trainer alone and leaves this process — and any other
    seed in the sweep — untouched.
    """

    def apply() -> None:
        wanted = stack_mb * 1024 * 1024
        _soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
        if hard != resource.RLIM_INFINITY:
            wanted = min(wanted, hard)
        resource.setrlimit(resource.RLIMIT_STACK, (wanted, hard))

    return apply


def post_mortem(run_dir: Path, out: Path, returncode: int,
                asked: int) -> dict[str, Any]:
    """What a non-zero exit actually cost, measured from the run directory.

    sb9x's open fault (``cloud.md`` §1) kills the trainer with ``SIGSEGV`` as
    ``train()`` returns — *after* every checkpoint is on disk. Judged by exit
    code alone that is indistinguishable from a run that died importing jax,
    and the difference is four hours of GPU time and a complete experiment:

    * every ``.cxpolicy`` the trainer wrote is a **complete, witness-checked
      file**. ``checked_policy`` runs the witness *before* a byte is written,
      so a checkpoint that exists is a checkpoint that passed;
    * the ``.best`` header carries the ``reward_curve`` up to its own
      iteration, and ``train.log`` carries the whole series; and
    * ``compare`` — which is how a checkpoint gets selected at all
      (ADR-099) — consumes exactly these files. Verified end to end on a
      crashed run: it played both checkpoints, measured per-motor torque and
      the hazard-15 column, and returned a verdict.

    What is genuinely lost is ``<label>.cxpolicy``, the **final iteration's**
    network — which is not the selection target, because 001 and 002 both
    measured that the last iteration is not the best checkpoint.

    So this reports rather than guesses, and the caller maps it to a distinct
    exit code. A run with no checkpoints at all is *not* salvageable and must
    not be quietly promoted.
    """

    checkpoints = sorted(
        p for p in run_dir.glob("*.cxpolicy") if p.name != out.name
    )
    final = out.is_file()
    report: list[str] = []
    salvageable = False

    if returncode == 0:
        return {"salvageable": True, "checkpoints": len(checkpoints),
                "final_policy": final, "report": []}

    if not checkpoints:
        report.append(f"  no checkpoints — this run produced nothing. "
                      f"Read {run_dir / 'train.log'}.")
    else:
        salvageable = True
        names = ", ".join(p.name for p in checkpoints[:4])
        more = "" if len(checkpoints) <= 4 else f", +{len(checkpoints) - 4} more"
        report.append(f"  SALVAGEABLE: {len(checkpoints)} checkpoint(s) on disk "
                      f"({names}{more}).")
        report.append("  Each was witness-checked before it was written, and "
                      "`compare` reads exactly these.")
        if final:
            report.append("  The final policy is present too; only the exit "
                          "code is wrong.")
        else:
            curve_in = (".best header and in train.log"
                        if any(p.name.endswith(".best.cxpolicy")
                               for p in checkpoints)
                        else "train.log (no .best file was written)")
            report.append(f"  Missing only {out.name} — the LAST iteration's "
                          f"network, which is not the checkpoint you select "
                          f"(ADR-099). The reward curve survives in the "
                          f"{curve_in}.")
        report.append(f"  Next: uv run python -m harness compare --dir "
                      f"{run_dir} --task {run_dir / 'stand-task.json'}")

    return {"salvageable": salvageable, "checkpoints": len(checkpoints),
            "final_policy": final, "iterations_asked": asked, "report": report}


def _fresh(path: Path) -> Path:
    """``path``, or ``path-2``, ``path-3``… — whichever does not yet exist."""

    candidate, suffix = path, 1
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            suffix += 1
            candidate = path.with_name(f"{path.name}-{suffix}")


def make_run_dir(label: str, jobs_root: Path, seed: int | None = None) -> Path:
    """``<jobs>/<label>[-s<seed>]-<YYYYMMDD>-<HHMMSS>/``, created.

    UTC in the name, to the second, with ``-2``, ``-3``… appended if that is
    somehow taken. An earlier version of this docstring asserted a collision
    "has never happened because a dispatch takes longer than a second to set
    up" — true of a *training* dispatch, and false the moment two of them are
    scripted back to back, which is exactly what an agent driving this on
    another box will do. Crashing on ``exist_ok=False`` there loses nothing
    but is a stupid way to lose a night.

    The ``-s<seed>`` infix appears only in a sweep. A single run keeps the
    plain ``<label>-<stamp>`` shape the eight directories in
    ``/home/theo/cadex-jobs`` use, so nothing downstream has to learn a second
    convention to read one.
    """

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    stem = label if seed is None else f"{label}-s{seed}"
    return _fresh(jobs_root / f"{stem}-{stamp}")


def stage(bundle_path: Path, run_dir: Path) -> tuple[Path, Path]:
    """Copy the bundle and its model into the run directory. Returns both.

    The model is found by the digest the bundle recorded, not by name — the
    same rule ``harness/episodes.resolve_model`` follows, for the same reason.
    """

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    reference = bundle.get("model") or {}
    declared = str(reference.get("sha256") or "")
    relative = Path(str(reference.get("path") or ""))

    candidates = [
        bundle_path.parent.parent / relative,
        bundle_path.parent / relative.name,
        *sorted(bundle_path.parent.glob("*-model.xml")),
    ]
    model = next(
        (path for path in candidates
         if path.is_file() and (not declared or sha256_file(path) == declared)),
        None,
    )
    if model is None:
        raise FileNotFoundError(
            f"No MJCF matching the bundle's model.sha256 {declared[:12]}… near "
            f"{bundle_path}. Looked at: "
            + ", ".join(str(item) for item in candidates)
        )

    staged_bundle = run_dir / bundle_path.name
    staged_model = run_dir / model.name
    shutil.copy2(bundle_path, staged_bundle)
    shutil.copy2(model, staged_model)
    return staged_bundle, staged_model


def resolved_hyperparameters(args: argparse.Namespace, seed: int | None) -> dict[str, Any]:
    """The fourteen numbers this run will actually be trained with.

    Written into the run directory rather than only passed on a command line,
    because the command line is not in the run directory and the question
    "did this replicate 200109?" is asked months later.
    """

    return {
        "seed": seed,
        "iterations": args.iterations,
        "envs": args.envs,
        "unroll": args.unroll,
        "epochs": args.epochs,
        "hidden": list(args.hidden),
        "learning_rate": args.learning_rate,
        "discount": args.discount,
        "gae_lambda": args.gae_lambda,
        "clip": args.clip,
        "entropy": args.entropy,
        "value_weight": args.value_weight,
        "initial_std": args.initial_std,
        "checkpoint_every": args.checkpoint_every,
    }


def dispatch_one(
    args: argparse.Namespace,
    seed: int | None,
    *,
    bundle_path: Path,
    trainer: Path,
    interpreter: str,
    pins: dict[str, Any],
    jobs_root: Path,
    sweep: bool,
) -> tuple[int, dict[str, Any]]:
    """One training run, start to exit. Returns its exit code and its facts."""

    run_dir = make_run_dir(args.label, jobs_root, seed if sweep else None)

    try:
        staged_bundle, staged_model = stage(bundle_path, run_dir)
    except (FileNotFoundError, ValueError) as exc:
        return EXIT_USAGE, {"error": str(exc), "run_dir": str(run_dir)}

    hyperparameters = resolved_hyperparameters(args, seed)
    (run_dir / "hyperparameters.json").write_text(
        json.dumps(hyperparameters, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    out = run_dir / f"{args.label}.cxpolicy"
    command = [
        str(interpreter),
        # The shim only prepends an interpreter setting; the trainer, its
        # arguments and its digest are untouched. See tools/trainer_launch.py.
        *([] if args.child_gc else [str(TRAINER_LAUNCH)]),
        str(trainer), str(staged_bundle),
        "--out", str(out),
        "--label", args.label,
        "--progress", str(run_dir / "progress.json"),
        "--iterations", str(args.iterations),
        "--envs", str(args.envs),
        "--checkpoint-every", str(args.checkpoint_every),
        "--unroll", str(args.unroll),
        "--epochs", str(args.epochs),
        "--hidden", *[str(width) for width in args.hidden],
        "--learning-rate", repr(args.learning_rate),
        "--discount", repr(args.discount),
        "--gae-lambda", repr(args.gae_lambda),
        "--clip", repr(args.clip),
        "--entropy", repr(args.entropy),
        "--value-weight", repr(args.value_weight),
        "--initial-std", repr(args.initial_std),
    ]
    if seed is not None:
        command += ["--seed", str(seed)]

    env = dict(os.environ)
    if args.cpu:
        # No trainer-side guard exists; this is the whole of "run on CPU".
        env["JAX_PLATFORMS"] = "cpu"
        env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("MUJOCO_GL", None)

    # Neither of these changes an update rule, a draw, or a number the
    # trainer computes — they change whether the process survives long
    # enough to write one. Both are recorded anyway, because "the run that
    # crashed" and "the run that did not" have to be told apart later, and
    # the constants above say what each is for.
    if not args.xla_preallocate:
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = TRAINER_XLA_PREALLOCATE
    else:
        env.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)

    runtime = {
        "stack_mb": args.stack_mb,
        "stack_soft_before_mb": _stack_soft_mb(),
        "xla_preallocate": bool(args.xla_preallocate),
        "child_gc": bool(args.child_gc),
        "host": platform.node(),
    }

    log_path = run_dir / "train.log"
    facts: dict[str, Any] = {
        "run_dir": str(run_dir),
        "seed": seed,
        "bundle": str(staged_bundle),
        "bundle_sha256": sha256_file(staged_bundle),
        "model": str(staged_model),
        "model_sha256": sha256_file(staged_model),
        "out": str(out),
        "command": command,
        "device_requested": "cpu" if args.cpu else "gpu",
        "trainer_pins": pins,
        "hyperparameters": hyperparameters,
        "runtime": runtime,
        "log": str(log_path),
    }
    (run_dir / "runtime.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"run dir   {run_dir}")
    print(f"seed      {seed if seed is not None else '— (trainer default)'}")
    print(f"bundle    {staged_bundle.name}  {facts['bundle_sha256'][:12]}…")
    print(f"model     {staged_model.name}  {facts['model_sha256'][:12]}…")
    print(f"device    {facts['device_requested']}")
    inherited = runtime["stack_soft_before_mb"]
    print(f"runtime   stack {args.stack_mb} MiB for the trainer "
          f"(this process has {'unlimited' if inherited is None else f'{inherited:.0f} MiB'})"
          f", XLA preallocate "
          f"{'on — trainer default' if args.xla_preallocate else TRAINER_XLA_PREALLOCATE}"
          f", cyclic GC {'on — trainer default' if args.child_gc else 'off'}")
    print(f"command   {' '.join(command)}")
    print()

    started = time.monotonic()
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command, cwd=str(run_dir), env=env,
            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            preexec_fn=_raise_child_stack(args.stack_mb),
        )
        (run_dir / "train.pid").write_text(str(process.pid), encoding="utf-8")
        facts["pid"] = process.pid
        print(f"dispatched pid {process.pid}; log at {log_path}", flush=True)

        if args.supervise:
            # Give the trainer a moment to write its first progress file, so
            # the supervisor does not report "no progress.json" on a run that
            # is merely still importing jax.
            time.sleep(2.0)
            from harness.supervise import build_parser as supervise_parser
            from harness.supervise import watch

            supervise_args = supervise_parser().parse_args([
                "--run", str(run_dir),
                "--watch",
                "--poll", str(args.poll),
                "--patience", str(args.patience),
                "--min-iterations", str(args.min_iterations),
                "--require-device", args.require_device,
                "--timeout", str(args.timeout),
                "--sigma-floor", str(args.sigma_floor),
            ])
            watched = watch(run_dir, supervise_args)
            facts["watch"] = watched

            # ``supervise --timeout`` stops *watching*, by design — it does not
            # stop training. In a sweep that would make the per-seed wall cap
            # fictional and let one stuck seed eat the whole night, so the cap
            # is enforced here, where the process handle is.
            if any(event["event"] == "watch-timeout" for event in watched["events"]):
                print(f"  wall cap {args.timeout:g}s reached — terminating this "
                      "seed so the sweep can continue.", flush=True)
                process.terminate()
                facts["wall_capped"] = True

        returncode = process.wait()

    facts["returncode"] = returncode
    facts["wall_seconds"] = round(time.monotonic() - started, 2)
    print()
    print(f"trainer exited {returncode} after {facts['wall_seconds']:.1f} s "
          f"({facts['wall_seconds'] / 3600:.2f} h)", flush=True)

    salvage = post_mortem(run_dir, out, returncode, args.iterations)
    facts["post_mortem"] = salvage
    for line in salvage["report"]:
        print(line, flush=True)

    if returncode == 0:
        return EXIT_OK, facts
    return (EXIT_SALVAGEABLE if salvage["salvageable"] else EXIT_INFRASTRUCTURE), facts


def dispatch(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    """Resolve everything shared, then run each seed in turn.

    Everything that can fail for the whole sweep — the bundle, the trainer,
    the venv pins — is checked **once, before the first run directory is
    made**, so a typo costs nothing rather than one staged directory per seed.
    """

    bundle_path = Path(args.bundle).expanduser().resolve()
    if not bundle_path.is_file():
        return EXIT_USAGE, {"error": f"no bundle at {bundle_path}"}

    repo = Path(os.environ.get("CADEX_REPO", "/home/theo/cadex"))
    trainer = repo / "training" / "cadex_train.py"
    if not trainer.is_file():
        return EXIT_USAGE, {"error": f"no trainer at {trainer}"}

    try:
        interpreter = trainer_python()
        pins = check_pins()
    except TrainerVenvError as exc:
        return EXIT_INFRASTRUCTURE, {"error": str(exc)}

    # ADR-104 is "refuse to dispatch to a box running a different trainer",
    # and it lives in remote_train.sh — which local dispatch never touches.
    # So this path recorded trainer_sha256 in the manifest and checked
    # nothing, which is the hyperparameter confound's shape exactly: the fact
    # is captured, and captured facts do not stop anything. It matters most
    # off this box, where "same seed, same hyperparameters" is worthless if
    # the update rule differs.
    trainer_digest = sha256_file(trainer)
    if args.require_trainer and trainer_digest != args.require_trainer:
        return EXIT_USAGE, {
            "error": (
                f"trainer digest is {trainer_digest}, --require-trainer "
                f"expects {args.require_trainer}. This box's "
                f"{trainer} is not the one that produced the runs you are "
                f"comparing against; seeds from it are comparable to each "
                f"other and to nothing else."
            ),
            "trainer_sha256": trainer_digest,
        }

    jobs_root = Path(
        args.jobs_dir or os.environ.get("CDXRL_JOBS", str(REPO_ROOT / "jobs"))
    ).expanduser()

    sweep = args.seeds is not None
    seeds: list[int | None] = list(args.seeds) if sweep else [args.seed]

    shared = resolved_hyperparameters(args, None)
    shared.pop("seed")
    manifest: dict[str, Any] = {
        "schema": "cdxrl-sweep-v1",
        "label": args.label,
        "seeds": seeds,
        "source_bundle": str(bundle_path),
        "source_bundle_sha256": sha256_file(bundle_path),
        "hyperparameters": shared,
        "hyperparameter_defaults": "stand-task-20260802-200109",
        "trainer": str(trainer),
        "trainer_sha256": trainer_digest,
        "trainer_pins": pins,
        "device_requested": "cpu" if args.cpu else "gpu",
        "runtime": {
            "stack_mb": args.stack_mb,
            "xla_preallocate": bool(args.xla_preallocate),
            "child_gc": bool(args.child_gc),
            "host": platform.node(),
        },
        "runs": [],
    }
    sweep_dir = Path(args.sweep_dir).expanduser() if args.sweep_dir else None

    def save_manifest() -> None:
        if sweep_dir is not None:
            (sweep_dir / "sweep.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

    if sweep:
        print(f"sweep     {args.label}: {len(seeds)} seed(s) "
              f"{', '.join(str(value) for value in seeds)}, one at a time")
        print(f"trainer   {trainer_digest[:12]}…"
              + ("  (matches --require-trainer)" if args.require_trainer else
                 "  (unchecked — pass --require-trainer to pin it)"))
        print(f"          {args.iterations} iterations each, "
              f"envs {args.envs}, checkpoint every {args.checkpoint_every}")
        if sweep_dir is not None:
            print(f"manifest  {sweep_dir / 'sweep.json'}")
        print()
    save_manifest()

    worst = EXIT_OK
    for index, seed in enumerate(seeds):
        if sweep:
            print("=" * 72)
            print(f"seed {seed}   ({index + 1} of {len(seeds)})")
            print("=" * 72, flush=True)
        code, facts = dispatch_one(
            args, seed, bundle_path=bundle_path, trainer=trainer,
            interpreter=interpreter, pins=pins, jobs_root=jobs_root, sweep=sweep,
        )
        manifest["runs"].append(facts)
        save_manifest()
        if code != EXIT_OK:
            if SEVERITY.get(code, 99) > SEVERITY.get(worst, 0):
                worst = code
            print(f"  seed {seed} exited {code}"
                  + ("  (salvageable — checkpoints are on disk)"
                     if code == EXIT_SALVAGEABLE else "")
                  + "; continuing to the next seed.", flush=True)
        if not sweep:
            return code, facts

    print()
    print(f"sweep complete: {len(manifest['runs'])} run(s)")
    for facts in manifest["runs"]:
        salvage = facts.get("post_mortem") or {}
        note = ""
        if facts.get("returncode") not in (0, None):
            note = ("  SALVAGEABLE, %d checkpoint(s)" % salvage["checkpoints"]
                    if salvage.get("salvageable") else "  NOTHING USABLE")
        print(f"  seed {facts.get('seed')}  rc {facts.get('returncode')}  "
              f"{facts.get('wall_seconds', 0) / 3600:.2f} h  "
              f"{facts.get('run_dir')}{note}")
    return worst, manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tools/train.py",
        description="Dispatch cadex_train.py locally, under supervision.",
    )
    parser.add_argument("--bundle", required=True, help="A cadex-training-task-v1 bundle.")
    parser.add_argument("--label", required=True, help="Run label; names the run dir and the policy.")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Run these seeds sequentially, one run directory each. One run "
             "at a time on this card (cloud.md). --seed drives both the "
             "PRNGKey and the episode-variation draws, so each is a full "
             "independent replicate.",
    )

    tuning = parser.add_argument_group(
        "hyperparameters",
        "Every one defaults to what stand-task-20260802-200109 ran, NOT to "
        "cadex_train.py's default — six of them differ. See RUN_200109.",
    )
    tuning.add_argument("--envs", type=int, default=RUN_200109["envs"])
    tuning.add_argument("--unroll", type=int, default=RUN_200109["unroll"])
    tuning.add_argument("--epochs", type=int, default=RUN_200109["epochs"])
    tuning.add_argument("--hidden", type=int, nargs="+", default=list(RUN_200109["hidden"]))
    tuning.add_argument("--learning-rate", type=float, default=RUN_200109["learning_rate"])
    tuning.add_argument("--discount", type=float, default=RUN_200109["discount"])
    tuning.add_argument("--gae-lambda", type=float, default=RUN_200109["gae_lambda"])
    tuning.add_argument("--clip", type=float, default=RUN_200109["clip"])
    tuning.add_argument("--entropy", type=float, default=RUN_200109["entropy"])
    tuning.add_argument("--value-weight", type=float, default=RUN_200109["value_weight"])
    tuning.add_argument("--initial-std", type=float, default=RUN_200109["initial_std"])

    parser.add_argument(
        "--cpu", action="store_true",
        help="JAX_PLATFORMS=cpu. There is no trainer-side CPU guard, so this "
             "is all it takes — and it is what makes experiment 000 free.",
    )
    parser.add_argument("--jobs-dir", default=None, help="Overrides CDXRL_JOBS.")
    parser.add_argument(
        "--detach", action="store_true",
        help="Re-exec in a new session and return immediately. method.md "
             "step 8: 'Detached, always.'",
    )
    parser.add_argument(
        "--sweep-dir", default=None,
        help="Where sweep.json and the detached log live. Made automatically.",
    )
    parser.add_argument(
        "--detached-child", action="store_true",
        help=argparse.SUPPRESS,  # set by --detach on the re-exec; not for humans
    )
    parser.add_argument("--supervise", action="store_true", help="Attach supervise --watch.")
    parser.add_argument("--poll", type=float, default=5.0)
    parser.add_argument(
        "--patience", type=int, default=0,
        help="Iterations since best before stopping. 0 disables stopping "
             "(experiment 001 Phase A found reward patience stops working "
             "runs, so the default is off).",
    )
    parser.add_argument("--min-iterations", type=int, default=0)
    parser.add_argument(
        "--require-device", default="",
        help="Stop a run whose progress.json reports another device.",
    )
    parser.add_argument(
        "--timeout", type=float, default=0.0,
        help="Per-seed wall cap in seconds. Unlike supervise's own --timeout "
             "this one does stop the run, so that one stuck seed cannot eat "
             "a whole sweep.",
    )
    parser.add_argument(
        "--require-trainer", default="",
        help="Refuse to dispatch unless cadex_train.py has this sha256. "
             "ADR-104 on another box: same seed and same hyperparameters mean "
             "nothing if the update rule differs. Experiment 002 ran "
             "aacfa82318e4e2399f65cf2ffe234504a288b586a178fc1dbe32e539a1fe7b24.",
    )
    parser.add_argument(
        "--sigma-floor", type=float, default=0.02,
        help="action_std below this is σ collapse; the run is stopped. "
             "Default 0.02 — 200109 decayed 0.4002 → 0.3375 over 2500 "
             "iterations, so this sits far below anything healthy.",
    )
    parser.add_argument(
        "--stack-mb", type=int, default=TRAINER_STACK_MB, metavar="MiB",
        help=f"Stack limit given to the trainer process. Default "
             f"{TRAINER_STACK_MB}. MJX traces deeply enough to overflow the "
             f"8 MiB default at 2048 environments, which segfaults during "
             f"tracing with no traceback. Note that `ulimit -s unlimited` "
             f"does NOT fix this — glibc gives threads its own 8 MiB default "
             f"for RLIM_INFINITY, and JAX traces on threads.",
    )
    parser.add_argument(
        "--xla-preallocate", action="store_true",
        help="Let XLA preallocate 75%% of the card, as it does by default. "
             "cdx-rl turns this off because it delays sb9x's "
             "post-checkpoint SIGSEGV from the first checkpoint to the last. "
             "It is not a fix, and memory is not the constraint — the task "
             "peaks at 777 MiB of 12 282. See cloud.md §1.",
    )
    parser.add_argument(
        "--child-gc", action="store_true",
        help="Leave CPython's cyclic collector on in the trainer. cdx-rl "
             "turns it off: a collection landing mid-trace segfaults, and "
             "although the artefacts survive the exit code does not — which "
             "is the only thing a sweep reads to tell a finished seed from a "
             "broken one. See tools/trainer_launch.py.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the facts as JSON.")
    return parser


def relaunch_detached(argv: list[str], sweep_dir: Path) -> dict[str, Any]:
    """Re-exec this file in its own session; return the child's facts.

    ``start_new_session=True`` is the whole point: the sweep gets its own
    process group and session leader, so closing the terminal — or the ssh
    connection that opened it — does not deliver SIGHUP to nine hours of GPU
    time. Today's foreground ``process.wait()`` had no such protection.
    """

    keep = [item for item in argv if item != "--detach"]
    command = [
        sys.executable, str(Path(__file__).resolve()), *keep,
        "--sweep-dir", str(sweep_dir), "--detached-child",
    ]
    log_path = sweep_dir / "sweep.log"
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            command, cwd=str(REPO_ROOT),
            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    (sweep_dir / "sweep.pid").write_text(str(process.pid), encoding="utf-8")
    return {
        "detached": True,
        "pid": process.pid,
        "sweep_dir": str(sweep_dir),
        "log": str(log_path),
        "command": command,
    }


def main(argv: list[str] | None = None) -> int:
    load_env_file()
    raw = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw)

    # Before forking, and before any directory is made. The check exists for
    # cross-box work, and the cross-box failure mode is precisely: dispatch
    # detached, read "detached pid 1234", walk away, and come back to a sweep
    # that died in its first second inside a log nobody opened.
    if args.detach and not args.detached_child and args.require_trainer:
        repo = Path(os.environ.get("CADEX_REPO", "/home/theo/cadex"))
        trainer = repo / "training" / "cadex_train.py"
        digest = sha256_file(trainer) if trainer.is_file() else ""
        if digest != args.require_trainer:
            print(f"ERROR  trainer digest is {digest or '(no trainer found)'}, "
                  f"--require-trainer expects {args.require_trainer}. "
                  f"Refusing to detach.", file=sys.stderr)
            return EXIT_USAGE

    jobs_root = Path(
        args.jobs_dir or os.environ.get("CDXRL_JOBS", str(REPO_ROOT / "jobs"))
    ).expanduser()

    # A sweep, or anything detached, gets a directory of its own to hold the
    # manifest and the log. A plain foreground single run keeps its old shape.
    if args.sweep_dir:
        sweep_dir: Path | None = Path(args.sweep_dir).expanduser()
        sweep_dir.mkdir(parents=True, exist_ok=True)
    elif args.detach or args.seeds is not None:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        sweep_dir = _fresh(jobs_root / f"{args.label}-sweep-{stamp}")
        args.sweep_dir = str(sweep_dir)
    else:
        sweep_dir = None

    if args.detach and not args.detached_child:
        facts = relaunch_detached(raw, sweep_dir)  # type: ignore[arg-type]
        print(f"detached  pid {facts['pid']}, session leader of its own")
        print(f"sweep dir {facts['sweep_dir']}")
        print(f"log       {facts['log']}")
        print(f"manifest  {Path(facts['sweep_dir']) / 'sweep.json'}")
        print()
        print("This process now exits. The sweep does not.")
        if args.json:
            json.dump(facts, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
        return EXIT_OK

    code, facts = dispatch(args)
    if args.json:
        json.dump(facts, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    elif facts.get("error"):
        print(f"ERROR  {facts['error']}", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
