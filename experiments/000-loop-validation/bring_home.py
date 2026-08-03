#!/usr/bin/env python
"""Bring a trained policy home: ``put_asset`` → ``assembly.policy`` → ``rollout``.

```
uv run python experiments/000-loop-validation/bring_home.py \\
    --project projects/000-loop-validation \\
    --script  experiments/000-loop-validation/rig.py \\
    --policy  jobs/pendulum-…/pendulum.cxpolicy
```

**This is the link nothing had tested.** Everything before it happens inside
one process or one file format; this is the round trip — a network trained by
a separate interpreter on a separate venv, copied into the project store as
an asset, declared in the script by digest, and re-verified by the engine.

What the engine re-checks when it accepts ``assembly.policy`` (ADR-084, and
the docstring on ``cadex_assembly_api.policy``):

1. the task bundle's own digest,
2. the model that bundle references,
3. the observation channels, **in order**,
4. the action table verbatim, at these indices, in these units and ranges,
5. the output map the bundle's derived action ranges imply, and
6. the witness — the observations the trainer recorded and the actions its
   own network produced for them — **re-evaluated in float64 by the engine's
   forward pass**, and refused past a measured tolerance.

``sha256`` is required and never inferred. VISION principle 3 says state that
cannot be rebuilt from the script is a bug, and hours of stochastic GPU
compute genuinely cannot be — so the script carries the one thing that *can*
be checked, which bytes it meant.

Two mechanics worth stating because getting either wrong is a refusal rather
than a warning:

* **``weights`` is the asset NAME**, not a path. ``put_asset`` copies the file
  into the project's ``assets/`` and returns the name it stored it under.
* **``frames_per_second`` must divide ``control_hz`` exactly.** The task runs
  at 50 Hz, so 25 is legal and 30 is not — a frame that lands between two
  actions is a frame of a trajectory that never happened.

Unlike ``mjcf``, ``task`` and ``policy``, a **rollout is baked** and falls
under ADR-077's one-simulation-per-script rule.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT))

from cadexd_client import (  # noqa: E402
    CadexdClient,
    CadexdError,
    Engine,
    EngineError,
    ScriptRefused,
    accepted_artifacts,
)

from harness import EXIT_INFRASTRUCTURE, EXIT_OK, EXIT_REFUSED, EXIT_USAGE  # noqa: E402
from harness.provenance import envelope, load_env_file, sha256_file  # noqa: E402

#: Appended to the rig script. ``result`` is already bound by the script's
#: last statement, so extending it is a one-liner rather than a rewrite —
#: which keeps the *authored* mechanism byte-identical between the revision
#: that trained and the revision that verifies.
EXTENSION = '''
policy = assembly.policy(task, weights={name!r}, sha256={digest!r},
                         label="pendulum-policy")
trace = assembly.rollout(policy, frames_per_second={fps}, seed={seed})
result = {{**result, "policy": policy, "rollout": trace}}
'''


def run(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    project = Path(args.project).expanduser().resolve()
    script = Path(args.script).expanduser().resolve()
    policy = Path(args.policy).expanduser().resolve()
    for path, what in ((script, "script"), (policy, "policy")):
        if not path.is_file():
            return EXIT_USAGE, {"error": f"no {what} at {path}"}

    digest = sha256_file(policy)
    payload: dict[str, Any] = {
        "project": str(project), "script": str(script),
        "policy": str(policy), "policy_sha256": digest,
        "policy_bytes": policy.stat().st_size,
        "frames_per_second": args.fps, "seed": args.seed,
    }

    try:
        engine = Engine.resolve()
    except EngineError as exc:
        return EXIT_INFRASTRUCTURE, {**payload, "error": str(exc)}

    client = CadexdClient(engine)
    try:
        client.start()
        client.open_project(project)
        client.require_dynamics()

        stored = client.put_asset(policy, name=args.name or policy.name)
        payload["asset"] = {
            key: stored.get(key) for key in ("name", "sha256", "bytes", "path")
        }
        name = str(stored.get("name") or policy.name)
        if stored.get("sha256") and str(stored["sha256"]) != digest:
            return EXIT_INFRASTRUCTURE, {
                **payload,
                "error": (
                    f"put_asset stored a file digesting to {stored['sha256']}; "
                    f"the file on disk digests to {digest}."
                ),
            }

        source = script.read_text(encoding="utf-8") + EXTENSION.format(
            name=name, digest=digest, fps=args.fps, seed=args.seed
        )
        payload["appended"] = EXTENSION.format(
            name=name, digest=digest, fps=args.fps, seed=args.seed
        )

        reply = client.write_script(source)
        payload["digest"] = str(reply.get("digest") or "")
        payload["outputs"] = [
            str(item.get("name") or "") for item in (reply.get("outputs") or [])
        ]
        payload["artifacts"] = [
            {"output": item.output, "kind": item.kind, "path": str(item.path),
             "bytes": item.path.stat().st_size if item.exists() else None,
             "sha256": sha256_file(item.path) if item.exists() else None}
            for item in accepted_artifacts(project)
        ]

    except ScriptRefused as exc:
        return EXIT_REFUSED, {
            **payload, "error": str(exc),
            "failure_code": exc.failure_code, "failure_stage": exc.failure_stage,
            **{key: exc.reply[key] for key in
               ("observed", "retry", "candidates", "correction", "reason")
               if key in exc.reply},
        }
    except (CadexdError, EngineError) as exc:
        return EXIT_INFRASTRUCTURE, {**payload, "error": str(exc)}
    except (OSError, ValueError) as exc:
        return EXIT_INFRASTRUCTURE, {**payload, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        client.shutdown()

    # The kinds the engine actually publishes, verified on this box rather
    # than guessed: a verified policy comes back as a *receipt*, and the
    # rollout as an ordinary simulation — which is the point of ADR-084 and
    # the rollout docstring, since nothing new reaches the viewport.
    payload["accepted_policy"] = any(
        item["kind"] == "assembly_policy_receipt_json"
        for item in payload["artifacts"]
    )
    payload["produced_rollout"] = any(
        item["kind"] == "assembly_simulation_json" for item in payload["artifacts"]
    )
    payload["receipts"] = [
        item for item in payload["artifacts"]
        if item["kind"] in ("assembly_policy_receipt_json", "assembly_simulation_json")
    ]
    ok = payload["accepted_policy"] and payload["produced_rollout"]
    return (EXIT_OK if ok else EXIT_INFRASTRUCTURE), payload


def report(payload: dict[str, Any], code: int) -> None:
    print(f"project   {payload['project']}")
    print(f"policy    {payload['policy']}")
    print(f"          sha256 {payload['policy_sha256']}  "
          f"({payload['policy_bytes']} bytes)")
    if payload.get("asset"):
        print(f"asset     {json.dumps(payload['asset'], sort_keys=True)}")
    if payload.get("error"):
        print(f"ERROR     {payload['error']}")
        for key in ("failure_code", "failure_stage", "observed", "retry",
                    "candidates", "correction", "reason"):
            if key in payload:
                value = payload[key]
                print(f"  {key:<14} "
                      + (json.dumps(value, sort_keys=True)
                         if isinstance(value, (dict, list)) else str(value)))
        return
    print(f"digest    {payload['digest']}")
    print(f"outputs   {', '.join(payload['outputs'])}")
    print()
    print(f"{'kind':<34} {'output':<14} {'bytes':>9}  sha256")
    for item in payload["artifacts"]:
        print(f"{item['kind']:<34} {item['output']:<14} "
              f"{(item['bytes'] if item['bytes'] is not None else '?'):>9}  "
              f"{str(item['sha256'] or '')[:12]}")
    print()
    print(f"engine accepted assembly.policy : {payload['accepted_policy']}")
    print(f"engine produced a rollout trace : {payload['produced_rollout']}")
    print()
    print("ok" if code == 0 else f"FAILED (exit {code})")


def main(argv: list[str] | None = None) -> int:
    load_env_file()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", required=True)
    parser.add_argument("--script", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--name", default=None, help="Asset name; defaults to the filename.")
    parser.add_argument(
        "--fps", type=int, default=25,
        help="Rollout frame rate. MUST divide the task's control_hz exactly.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    code, payload = run(args)
    if args.json:
        json.dump(envelope("bring_home", code == EXIT_OK, payload),
                  sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        report(payload, code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
