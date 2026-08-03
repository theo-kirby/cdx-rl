#!/usr/bin/env python
"""Dispatch ``cadex_train.py`` into a run directory, under supervision.

```
uv run python tools/train.py --bundle PATH --label NAME
       [--iterations N] [--envs N] [--checkpoint-every N]
       [--cpu] [--supervise] [--patience N]
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


def make_run_dir(label: str, jobs_root: Path) -> Path:
    """``<jobs>/<label>-<YYYYMMDD>-<HHMMSS>/``, created.

    UTC in the name. Two runs a second apart are distinguishable; two runs in
    the same second are not, and that has never happened because a dispatch
    takes longer than a second to set up.
    """

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = jobs_root / f"{label}-{stamp}"
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


def dispatch(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
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
    run_dir = make_run_dir(args.label, jobs_root)

    try:
        staged_bundle, staged_model = stage(bundle_path, run_dir)
    except (FileNotFoundError, ValueError) as exc:
        return EXIT_USAGE, {"error": str(exc), "run_dir": str(run_dir)}

    out = run_dir / f"{args.label}.cxpolicy"
    command = [
        str(interpreter), str(trainer), str(staged_bundle),
        "--out", str(out),
        "--label", args.label,
        "--iterations", str(args.iterations),
        "--envs", str(args.envs),
        "--checkpoint-every", str(args.checkpoint_every),
        "--progress", str(run_dir / "progress.json"),
    ]
    if args.seed is not None:
        command += ["--seed", str(args.seed)]

    env = dict(os.environ)
    if args.cpu:
        # No trainer-side guard exists; this is the whole of "run on CPU".
        env["JAX_PLATFORMS"] = "cpu"
        env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("MUJOCO_GL", None)

    log_path = run_dir / "train.log"
    facts: dict[str, Any] = {
        "run_dir": str(run_dir),
        "bundle": str(staged_bundle),
        "bundle_sha256": sha256_file(staged_bundle),
        "model": str(staged_model),
        "model_sha256": sha256_file(staged_model),
        "out": str(out),
        "command": command,
        "device_requested": "cpu" if args.cpu else "gpu",
        "trainer_pins": pins,
        "log": str(log_path),
    }

    print(f"run dir   {run_dir}")
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
        print(f"dispatched pid {process.pid}; log at {log_path}")

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
            ])
            facts["watch"] = watch(run_dir, supervise_args)

        returncode = process.wait()

    facts["returncode"] = returncode
    facts["wall_seconds"] = round(time.monotonic() - started, 2)
    print()
    print(f"trainer exited {returncode} after {facts['wall_seconds']:.1f} s")
    return (EXIT_OK if returncode == 0 else EXIT_INFRASTRUCTURE), facts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tools/train.py",
        description="Dispatch cadex_train.py locally, under supervision.",
    )
    parser.add_argument("--bundle", required=True, help="A cadex-training-task-v1 bundle.")
    parser.add_argument("--label", required=True, help="Run label; names the run dir and the policy.")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--cpu", action="store_true",
        help="JAX_PLATFORMS=cpu. There is no trainer-side CPU guard, so this "
             "is all it takes — and it is what makes experiment 000 free.",
    )
    parser.add_argument("--jobs-dir", default=None, help="Overrides CDXRL_JOBS.")
    parser.add_argument("--supervise", action="store_true", help="Attach supervise --watch.")
    parser.add_argument("--poll", type=float, default=5.0)
    parser.add_argument(
        "--patience", type=int, default=0,
        help="Iterations since best before stopping. 0 disables stopping "
             "(experiment 001 Phase A found reward patience stops working "
             "runs, so the default is off).",
    )
    parser.add_argument("--min-iterations", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="Emit the facts as JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_env_file()
    args = build_parser().parse_args(argv)
    code, facts = dispatch(args)
    if args.json:
        json.dump(facts, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    elif facts.get("error"):
        print(f"ERROR  {facts['error']}", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
