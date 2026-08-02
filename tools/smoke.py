#!/usr/bin/env python
"""Prove cdx-rl can talk to everything it depends on, before it depends on it.

Five checks, in the order in which a failure is cheapest to diagnose:

1. **Engine** — ``CADEX_ENGINE_DEV_TREE`` (or ``CADEX_ENGINE_ROOT``) resolves
   to a real binary and a real module directory, **and its assembly domain
   actually has the MuJoCo surface**. That last clause is not decoration: the
   staged payload on this box predates the dynamics work and fails only later,
   from inside a script, in a way that reads like a modelling error.
2. **cadexd handshake** — a ``FreeCADCmd`` child spawns and announces
   ``ready`` over NDJSON.
3. **A round trip that builds something** — ``open_project`` →
   ``describe_api`` → ``write_script`` (a trivial parametric box) →
   ``rebuild`` → ``inspect scope="output"``, with the accepted attempt's
   ``outputs/`` directory resolved from ``script.json`` and **the digest
   compared across two rebuilds**. That last assertion is the one that
   matters: a digest that moves between two runs of one script means nothing
   downstream is reproducible, and every number this repository ever records
   is a number about a particular build.
4. **The trainer venv** — ``jax.default_backend() == "gpu"``, the four pinned
   versions, and a GPU that ``nvidia-smi`` agrees exists. Checked from
   *outside* the venv, by running its own interpreter, because that is how
   training is actually invoked.
5. **The Cadex checkout's HEAD** — ADR-104's guard. ``remote_train.sh``
   dispatches to a box that runs its *own* checkout of
   ``training/cadex_train.py``, so a trainer that predates a surface addition
   silently ignores the new fields while recording the new algorithm string
   in the policy header, and nothing fails loudly. It matters to cdx-rl for
   exactly that reason: the commit is part of what a result means.

    uv run python tools/smoke.py            # prose
    uv run python tools/smoke.py --json     # the envelope

Exit 0 if every check passed, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cadexd_client import (  # noqa: E402
    CadexdClient,
    CadexdError,
    Engine,
    EngineError,
    ScriptRefused,
    accepted_artifacts,
    accepted_attempt_dir,
    accepted_outputs_dir,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

#: A parametric box. Deliberately the smallest thing that is still a *real*
#: build: it declares a parameter, so the parameter cache is exercised, and it
#: publishes one output, so there is an artifact to resolve a path to.
SMOKE_SCRIPT = """
p = params(side=num(20.0, unit="mm", min=5.0, max=100.0, step=1.0))
cube = part.box(p.side, p.side, p.side)
result = {"cube": cube}
"""

#: What the trainer venv must be, exactly. These are not preferences: they
#: are the pins the recorded runs were produced under, and a mismatch means
#: a number in the graph was measured against a different simulator.
EXPECTED_TRAINER_PINS = {
    "python": "3.12.3",
    "mujoco": "3.10.0",
    "mujoco-mjx": "3.10.0",
    "jax": "0.7.2",
}


def load_env_file(path: Path) -> None:
    """Fold ``config/env`` into the environment, without overriding it.

    A value already exported wins, so a one-off ``CADEX_ENGINE_ROOT=... uv run
    ...`` does what it looks like it does.
    """

    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


class Checks:
    """A list of ``(name, ok, detail)``, and whether all of them passed."""

    def __init__(self) -> None:
        self.results: list[dict[str, Any]] = []

    def record(self, name: str, ok: bool, detail: Any = None) -> bool:
        self.results.append({"check": name, "ok": bool(ok), "detail": detail})
        return bool(ok)

    @property
    def ok(self) -> bool:
        return all(item["ok"] for item in self.results)


# --------------------------------------------------------------------------


def check_engine(checks: Checks, explicit: str | None = None) -> Engine | None:
    try:
        engine = Engine.resolve(explicit)
    except EngineError as exc:
        checks.record("engine_resolved", False, str(exc))
        return None
    checks.record("engine_resolved", True, engine.describe())
    return engine


def check_cadexd_round_trip(checks: Checks, engine: Engine) -> dict[str, Any]:
    """Handshake, build, resolve artifacts, rebuild, compare digests."""

    facts: dict[str, Any] = {}
    project_root = Path(tempfile.mkdtemp(prefix="cdxrl-smoke-", dir=REPO_ROOT / "projects"))
    facts["project_root"] = str(project_root)
    client = CadexdClient(engine)
    try:
        started = time.monotonic()
        try:
            client.start()
        except CadexdError as exc:
            checks.record("cadexd_handshake", False, str(exc))
            return facts
        facts["ready_seconds"] = round(time.monotonic() - started, 2)
        checks.record(
            "cadexd_handshake", True, {"ready_seconds": facts["ready_seconds"]}
        )

        client.open_project(project_root)
        checks.record("open_project", True, {"revision": client.revision})

        api = client.describe_api()
        domains = sorted(str(name) for name in (api.get("domains") or []))
        facts["domains"] = domains
        checks.record(
            "describe_api",
            bool(domains),
            {"domains": domains, "engine": api.get("engine")},
        )

        # The check that would have caught the stale payload on day one.
        try:
            exports = client.require_dynamics()
        except EngineError as exc:
            checks.record("dynamics_surface", False, str(exc))
        else:
            facts["assembly_exports"] = exports
            checks.record(
                "dynamics_surface", True, {"assembly_exports": len(exports)}
            )

        try:
            first = client.write_script(SMOKE_SCRIPT)
        except ScriptRefused as exc:
            # Exit code 3's shape, and the reason cdx-rl keeps the two
            # exception types apart: this is a modelling problem, not an
            # infrastructure one.
            checks.record(
                "write_script",
                False,
                {"failure_code": exc.failure_code, "error": str(exc)},
            )
            return facts
        digest_one = str(first.get("digest") or "")
        facts["digest"] = digest_one
        facts["outputs"] = [
            str(item.get("name") or "") for item in (first.get("outputs") or [])
        ]
        checks.record(
            "write_script",
            bool(digest_one),
            {"digest": digest_one, "outputs": facts["outputs"]},
        )

        # The artifact resolution the --json envelope does not do.
        try:
            staging = accepted_attempt_dir(project_root)
            outputs_dir = accepted_outputs_dir(project_root)
            artifacts = accepted_artifacts(project_root)
        except (ValueError, FileNotFoundError, OSError) as exc:
            checks.record("resolve_outputs_dir", False, str(exc))
            artifacts = []
            outputs_dir = None
        else:
            facts["attempt_dir"] = str(staging.relative_to(project_root))
            facts["artifacts"] = [
                {
                    "output": item.output,
                    "kind": item.kind,
                    "path": str(item.path),
                    "exists": item.exists(),
                }
                for item in artifacts
            ]
            checks.record(
                "resolve_outputs_dir",
                outputs_dir.is_dir() and all(item.exists() for item in artifacts),
                {
                    "outputs_dir": str(outputs_dir),
                    "artifact_count": len(artifacts),
                    "kinds": sorted({item.kind for item in artifacts}),
                },
            )

        inspected = client.inspect("output")
        facts["inspect_output"] = inspected.get("value")
        checks.record("inspect_output", inspected.get("ok") is True, inspected.get("value"))

        second = client.rebuild()
        digest_two = str(second.get("digest") or "")
        checks.record(
            "digest_stable",
            bool(digest_one) and digest_one == digest_two,
            {"first": digest_one, "second": digest_two},
        )
    finally:
        client.shutdown()
        shutil.rmtree(project_root, ignore_errors=True)
    return facts


TRAINER_PROBE = r"""
import json, sys
import importlib.metadata as md
import jax, mujoco, mujoco.mjx
out = {
    "python": ".".join(str(n) for n in sys.version_info[:3]),
    "backend": jax.default_backend(),
    "devices": [str(d) for d in jax.devices()],
    "mujoco": mujoco.__version__,
    "jax": jax.__version__,
}
for name in ("mujoco", "mujoco-mjx", "jax"):
    try:
        out[name + "_dist"] = md.version(name)
    except Exception as exc:
        out[name + "_dist"] = "missing: %s" % exc
print(json.dumps(out))
"""


def check_trainer(checks: Checks) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    venv = Path(os.environ.get("CADEX_TRAIN_VENV", "/home/theo/cadex-train-venv"))
    python = venv / "bin" / "python"
    if not python.is_file():
        checks.record("trainer_venv", False, f"No interpreter at {python}")
        return facts
    try:
        completed = subprocess.run(
            [str(python), "-c", TRAINER_PROBE],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        checks.record("trainer_venv", False, str(exc))
        return facts
    if completed.returncode != 0:
        checks.record(
            "trainer_venv", False, (completed.stderr or completed.stdout)[-2000:]
        )
        return facts
    try:
        probe = json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        checks.record("trainer_venv", False, completed.stdout[-2000:])
        return facts
    facts.update(probe)

    mismatches = {
        "python": (probe.get("python"), EXPECTED_TRAINER_PINS["python"]),
        "mujoco": (probe.get("mujoco_dist"), EXPECTED_TRAINER_PINS["mujoco"]),
        "mujoco-mjx": (probe.get("mujoco-mjx_dist"), EXPECTED_TRAINER_PINS["mujoco-mjx"]),
        "jax": (probe.get("jax_dist"), EXPECTED_TRAINER_PINS["jax"]),
    }
    wrong = {k: v for k, v in mismatches.items() if v[0] != v[1]}
    checks.record("trainer_pins", not wrong, wrong or EXPECTED_TRAINER_PINS)
    checks.record(
        "trainer_gpu",
        probe.get("backend") == "gpu",
        {"backend": probe.get("backend"), "devices": probe.get("devices")},
    )

    smi = shutil.which("nvidia-smi")
    if not smi:
        checks.record("nvidia_smi", False, "nvidia-smi not on PATH")
        return facts
    result = subprocess.run(
        [smi, "--query-gpu=name,memory.total,driver_version",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=60,
    )
    line = result.stdout.strip()
    facts["nvidia_smi"] = line
    checks.record("nvidia_smi", result.returncode == 0 and bool(line), line)
    return facts


def check_cadex_checkout(checks: Checks) -> dict[str, Any]:
    """ADR-104's guard: which trainer is on this box, exactly."""

    repo = Path(os.environ.get("CADEX_REPO", "/home/theo/cadex"))
    facts: dict[str, Any] = {"repo": str(repo)}
    if not (repo / ".git").is_dir():
        checks.record("cadex_checkout", False, f"{repo} is not a git checkout")
        return facts
    head = subprocess.run(
        ["git", "-C", str(repo), "log", "--oneline", "-1"],
        capture_output=True, text=True, timeout=60,
    )
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        capture_output=True, text=True, timeout=60,
    )
    facts["head"] = head.stdout.strip()
    facts["dirty"] = bool(status.stdout.strip())
    # A dirty checkout is reported, not failed: it is somebody's work in
    # progress and cdx-rl has no business having an opinion about it. What it
    # must not be is *unrecorded*, because a run dispatched against it cannot
    # be reproduced from a commit.
    checks.record(
        "cadex_checkout",
        head.returncode == 0 and bool(facts["head"]),
        {"head": facts["head"], "dirty": facts["dirty"]},
    )
    return facts


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="Emit the envelope.")
    parser.add_argument(
        "--engine", default=None,
        help="A staged engine payload root, overriding the environment.",
    )
    parser.add_argument(
        "--skip-engine", action="store_true",
        help="Skip the cadexd round trip. Rarely worth it: the ready banner "
             "arrives in ~0.04 s on a warm page cache, and the whole round "
             "trip is a few seconds.",
    )
    parser.add_argument(
        "--skip-trainer", action="store_true",
        help="Skip the trainer venv probe (it initialises CUDA).",
    )
    args = parser.parse_args(argv)

    load_env_file(REPO_ROOT / "config" / "env")
    (REPO_ROOT / "projects").mkdir(exist_ok=True)

    checks = Checks()
    facts: dict[str, Any] = {}

    if not args.skip_engine:
        engine = check_engine(checks, args.engine)
        if engine is not None:
            facts["cadexd"] = check_cadexd_round_trip(checks, engine)
            facts["engine"] = engine.describe()
    if not args.skip_trainer:
        facts["trainer"] = check_trainer(checks)
    facts["cadex_checkout"] = check_cadex_checkout(checks)

    envelope = {
        "schema": "cdxrl-smoke-v1",
        "ok": checks.ok,
        "checks": checks.results,
        "facts": facts,
    }

    if args.json:
        json.dump(envelope, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        for item in checks.results:
            mark = "ok  " if item["ok"] else "FAIL"
            detail = item["detail"]
            if isinstance(detail, (dict, list)):
                detail = json.dumps(detail, sort_keys=True)
            print(f"{mark}  {item['check']:<22} {detail}")
        print()
        print("PASS" if checks.ok else "FAIL")
    return 0 if checks.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
