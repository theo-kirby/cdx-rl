#!/usr/bin/env python
"""Build the mg-legs rollout project and bake the best policy into a trace.

    cd ~/cdx-rl
    set -a; . ./config/env; set +a
    uv run python mechanisms/mg-legs/rollout/build.py

Produces ``projects/mg-legs-rollout.cadex`` — a project store carrying the
mechanism, the trained policy as an asset, and one ``assembly_simulation_json``
trace. That trace is what the **Shell** bakes: open the project on macOS and
the learned gait plays on the timeline, with the Policy Outputs panel beside
it. `cadex.md` §2: cdx-rl drives the engine and the trainer, and sends hero
results to the Shell for visual verification.

Nothing here trains, and nothing here renders. The engine replays the episode
the policy was scored on and writes the poses; the Shell draws them.

## What it plays

`stand10.001700.cxpolicy` — experiment 003, seed 2, iteration 1700. The best
policy this repository has produced: **18/24 on the conjunction** (stepped
>= 10 mm AND survived 300/300), 20/24 survival, mean episode 264.4 of 300.

Note the honest caveat, which the README repeats: McNemar cannot separate it
from seed 0's `001150` (p = 1.000) or seed 1's `001750` (p = 0.625). It leads
every point estimate and the continuous measure; it is not a proven winner.

**Evaluation seed 4**, chosen because it is the episode worth watching rather
than the average one: 10 steps, longest 121.8 mm, highest lift 36.5 mm, and it
survives the full 300 control steps. Six of the 24 seeds this policy is scored
on do not survive; this is not one of them, and a rollout of a failure would
be a different and also useful artifact (change ``seed=`` in ``script.py``).

## The traps this driver clears

* **The engine must be the checkout, not the staged payload.** The 2026-07-31
  payload predates the whole MuJoCo surface and fails with *"assembly.mjcf is
  not defined"*, which reads like a modelling bug. ``require_dynamics()``
  turns it into one loud failure instead.
* **``put_asset`` moves the project revision even though it is not a write**,
  so the next ``write_script`` is refused as ``STALE_PROGRAM_REVISION``.
  ``CadexdClient.put_asset`` calls ``refresh_revision()`` for exactly this.
  Order matters: the policy goes in *before* the script that names it.
* **The digest is never pasted.** ``assembly.policy(..., sha256=)`` is
  checked by the engine against the bytes in the store, and this script
  recomputes it from disk and refuses if ``script.py`` disagrees.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from cadexd_client import (  # noqa: E402
    CadexdClient,
    Engine,
    accepted_artifacts,
)

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "script.py"
ASSET = HERE / "assets" / "stand10.001700.cxpolicy"
PROJECT = REPO_ROOT / "projects" / "mg-legs-rollout.cadex"

#: The **live** ``assembly.policy(...)`` call. Anchored to the start of a line
#: because this project keeps every retired policy as a ``#``-prefixed record
#: — there are five of them above the live one, each with its own digest, and
#: an unanchored ``sha256="…"`` matches the oldest. Same rule, and the same
#: reason, as ``drivers/install_checkpoint.py::POLICY_CALL``.
POLICY_CALL = re.compile(
    r"^\w+\s*=\s*assembly\.policy\(.*?\)\s*$",
    re.MULTILINE | re.DOTALL,
)
POLICY_SHA = re.compile(r'sha256="([0-9a-f]{64})"')


def main() -> int:
    source = SCRIPT.read_text()

    # The digest in the script must be the digest of the bytes we are about to
    # install. The engine checks this too and refuses by name; checking here
    # says which of the two files is wrong.
    live = POLICY_CALL.search(source)
    if not live:
        print("no live assembly.policy(...) call in script.py", file=sys.stderr)
        return 2
    declared = POLICY_SHA.search(live.group(0))
    actual = hashlib.sha256(ASSET.read_bytes()).hexdigest()
    if not declared:
        print("no sha256= in script.py's live policy call", file=sys.stderr)
        return 2
    if declared.group(1) != actual:
        print(f"script.py declares {declared.group(1)[:16]}…\n"
              f"asset on disk is  {actual[:16]}…", file=sys.stderr)
        return 2
    print(f"policy    {ASSET.name}  {actual[:12]}…  "
          f"{ASSET.stat().st_size} bytes")

    engine = Engine.resolve()
    print(f"engine    {engine.describe().get('root')}  "
          f"({engine.describe().get('version')})")

    PROJECT.parent.mkdir(parents=True, exist_ok=True)
    with CadexdClient(engine) as client:
        client.open_project(PROJECT)
        client.require_dynamics()          # the stale-payload trap, loudly
        client.put_asset(ASSET)            # …and this moves the revision
        print(f"asset     installed into {PROJECT.name}/assets/")

        reply = client.write_script(source)
        digest = reply.get("digest") or ""
        outputs = [str(o.get("name") or "") for o in (reply.get("outputs") or [])]
        print(f"script    accepted, digest {digest[:12]}…, "
              f"{len(outputs)} outputs")

    artifacts = accepted_artifacts(PROJECT)
    kinds: dict[str, list[str]] = {}
    for art in artifacts:
        kinds.setdefault(str(art.artifact_kind), []).append(str(art.name))

    trace = [a for a in artifacts
             if str(a.artifact_kind) == "assembly_simulation_json"]
    print()
    for kind, names in sorted(kinds.items()):
        print(f"  {kind:34s} {', '.join(sorted(names))}")

    if not trace:
        print("\nNO SIMULATION TRACE — the Shell has nothing to bake.",
              file=sys.stderr)
        return 1

    path = Path(str(trace[0].path))
    frames = 0
    try:
        data = json.loads(path.read_text())
        frames = len(data.get("frames") or [])
    except Exception:  # noqa: BLE001 — reporting only
        pass
    print(f"\nTRACE     {path}")
    print(f"          {path.stat().st_size} bytes, {frames} frames")
    print(f"\nOpen {PROJECT} in the Shell on macOS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
