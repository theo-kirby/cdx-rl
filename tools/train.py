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
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from harness import EXIT_INFRASTRUCTURE, EXIT_OK, EXIT_USAGE  # noqa: E402
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


def make_run_dir(label: str, jobs_root: Path, seed: int | None = None) -> Path:
    """``<jobs>/<label>[-s<seed>]-<YYYYMMDD>-<HHMMSS>/``, created.

    UTC in the name. Two runs a second apart are distinguishable; two runs in
    the same second are not, and that has never happened because a dispatch
    takes longer than a second to set up.

    The ``-s<seed>`` infix appears only in a sweep. A single run keeps the
    plain ``<label>-<stamp>`` shape the eight directories in
    ``/home/theo/cadex-jobs`` use, so nothing downstream has to learn a second
    convention to read one.
    """

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    stem = label if seed is None else f"{label}-s{seed}"
    run_dir = jobs_root / f"{stem}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


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
        str(interpreter), str(trainer), str(staged_bundle),
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
        "log": str(log_path),
    }

    print(f"run dir   {run_dir}")
    print(f"seed      {seed if seed is not None else '— (trainer default)'}")
    print(f"bundle    {staged_bundle.name}  {facts['bundle_sha256'][:12]}…")
    print(f"model     {staged_model.name}  {facts['model_sha256'][:12]}…")
    print(f"device    {facts['device_requested']}")
    print(f"command   {' '.join(command)}")
    print()

    started = time.monotonic()
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command, cwd=str(run_dir), env=env,
            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
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
    return (EXIT_OK if returncode == 0 else EXIT_INFRASTRUCTURE), facts


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
        "trainer_sha256": sha256_file(trainer),
        "trainer_pins": pins,
        "device_requested": "cpu" if args.cpu else "gpu",
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
            worst = code
            print(f"  seed {seed} exited {code}; continuing to the next seed.",
                  flush=True)
        if not sweep:
            return code, facts

    print()
    print(f"sweep complete: {len(manifest['runs'])} run(s)")
    for facts in manifest["runs"]:
        print(f"  seed {facts.get('seed')}  rc {facts.get('returncode')}  "
              f"{facts.get('wall_seconds', 0) / 3600:.2f} h  "
              f"{facts.get('run_dir')}")
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
        "--sigma-floor", type=float, default=0.02,
        help="action_std below this is σ collapse; the run is stopped. "
             "Default 0.02 — 200109 decayed 0.4002 → 0.3375 over 2500 "
             "iterations, so this sits far below anything healthy.",
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
        sweep_dir = jobs_root / f"{args.label}-sweep-{stamp}"
        sweep_dir.mkdir(parents=True, exist_ok=False)
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
