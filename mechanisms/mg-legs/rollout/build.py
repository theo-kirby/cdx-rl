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

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT))

from cadexd_client import (  # noqa: E402
    CadexdClient,
    Engine,
    accepted_artifacts,
)

from harness.replay import (  # noqa: E402
    ROLES as REPLAY_ROLES,
    ReplayError,
    find_set,
    read_set,
    script_drift,
    verify_set,
)

HERE = Path(__file__).resolve().parent


def _replay_set(arm: str) -> tuple[Path, dict] | None:
    """The imported replay set for ``arm``, verified, or ``None``.

    ``None`` is a state rather than an error: the two committed heroes still
    build from ``assets/`` on a checkout that has never imported anything, and
    that is the path every pre-ADR-134 invocation took.

    What is *not* silent is a set that fails its own manifest. That is refused
    here rather than passed on, because the alternative is ``decode_policy``
    complaining about a container three steps later, which reads like a corrupt
    file rather than a bad copy.
    """

    try:
        set_dir = find_set(arm)
    except ReplayError:
        return None
    manifest = read_set(set_dir)
    complaints = verify_set(set_dir, manifest)
    if complaints:
        raise SystemExit(
            f"The replay set at {set_dir} does not match its own manifest:\n  "
            + "\n  ".join(complaints)
            + "\nRe-import it, or delete it and export again."
        )
    drift = script_drift(manifest)
    if drift:
        print(f"NOTE      {drift}")
        print("          The build below is what can actually tell; this is "
              "only a heads-up.")
    return set_dir, manifest

#: The arms this directory can bake. Each is (script, asset, project); the
#: script names its policy by digest and ``main`` refuses if the two disagree.
#:
#: ``clamp25`` needs ADR-131's ``command_limits_degrees``, which merged on
#: 2026-08-05 (`theo-kirby/cadex#1`). Before that the clamped bundle could
#: only be produced by editing the derived task JSON by hand, which is why
#: this arm did not exist.
#:
#: **This table is the fallback, not the route.** Since ADR-134 both scripts
#: name a ``trained_task``, which is a file that arrives in a **replay set**
#: rather than one that lives in git — so ``--arm b8`` looks for
#: ``replay/`` first and only falls back to these paths when there is no set.
#: ``harness replay`` is what puts one there, on either machine.
ARMS = {
    "b8": (
        HERE / "script.py",
        HERE / "assets" / "stand10.001700.cxpolicy",
        REPO_ROOT / "projects" / "mg-legs-rollout.cadex",
    ),
    "clamp25": (
        HERE / "script-clamp25.py",
        HERE / "assets" / "stand13.001800.cxpolicy",
        REPO_ROOT / "projects" / "mg-legs-rollout-clamp25.cadex",
    ),
}
DEFAULT_ARM = "b8"

SCRIPT, ASSET, PROJECT = ARMS[DEFAULT_ARM]

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
    global SCRIPT, ASSET, PROJECT
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--arm", choices=sorted(ARMS), default=DEFAULT_ARM,
        help=(
            "b8: experiment 003 seed 2 iteration 1700, the unclamped hero. "
            "clamp25: experiment 004's stand13.001800 -- the same 18/24 on "
            "the conjunction with a quarter of the resting bracing."
        ),
    )
    parser.add_argument(
        "--project", type=Path, default=None,
        help=(
            "build here instead of the arm's usual project. A store built by "
            "one engine is refused by another at open_project ('the restore "
            "pass digest does not match'), which is correct and is why a "
            "second path is sometimes the shortest way forward."
        ),
    )
    args = parser.parse_args()
    arm = args.arm
    SCRIPT, ASSET, PROJECT = ARMS[arm]
    if args.project is not None:
        PROJECT = args.project

    # A replay set, if one has been imported for this arm, is the route
    # (ADR-134/135): it carries the ``.cxpolicy`` *and* the training bundle and
    # model that ``trained_task=`` binds the policy to. `harness replay` is the
    # only thing that assembles one, and `--import` is what checks its digests.
    replay_set = _replay_set(arm)
    if replay_set is None and "trained_task" in SCRIPT.read_text():
        print(f"{SCRIPT.name} names a trained_task, which arrives in a replay "
              f"set, and there is no set for arm {arm!r}.\n"
              f"On this box:   uv run python -m harness replay --export "
              f"--dir jobs/… --iteration … --task tasks/… --arm {arm} "
              f"--script {SCRIPT.relative_to(REPO_ROOT)} --label {arm}\n"
              f"From another:  uv run python -m harness replay --import "
              f"replay/{arm}\n"
              f"`harness replay --list` shows what there is.",
              file=sys.stderr)
        return 2
    source = SCRIPT.read_text()

    # The digest in the script must be the digest of the bytes we are about to
    # install. The engine checks this too and refuses by name; checking here
    # says which of the two files is wrong.
    live = POLICY_CALL.search(source)
    if not live:
        print("no live assembly.policy(...) call in script.py", file=sys.stderr)
        return 2
    declared = POLICY_SHA.search(live.group(0))
    if not declared:
        print("no sha256= in script.py's live policy call", file=sys.stderr)
        return 2

    # The set's policy wins over ``assets/`` when there is one: it is the copy
    # whose digest was checked on arrival, and it travelled with the bundle the
    # script's ``trained_task=`` names. ``assets/`` is the two committed heroes
    # and nothing else.
    installing = [ASSET]
    if replay_set is not None:
        set_dir, manifest = replay_set
        installing = [set_dir / manifest[role]["file"] for role in REPLAY_ROLES]
        ASSET = installing[0]
        print(f"set       {set_dir}  (arm {manifest.get('arm')!r}, "
              f"iteration {(manifest.get('run') or {}).get('iteration')})")

    actual = hashlib.sha256(ASSET.read_bytes()).hexdigest()
    if declared.group(1) != actual:
        print(f"{SCRIPT.name} declares {declared.group(1)[:16]}…\n"
              f"policy on disk is {actual[:16]}…  ({ASSET})", file=sys.stderr)
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
        for path in installing:            # …and each of these moves the revision
            client.put_asset(path)
            print(f"asset     {path.name}")

        reply = client.write_script(source)
        digest = reply.get("digest") or ""
        outputs = [str(o.get("name") or "") for o in (reply.get("outputs") or [])]
        print(f"script    accepted, digest {digest[:12]}…, "
              f"{len(outputs)} outputs")

    artifacts = accepted_artifacts(PROJECT)
    # ``Artifact`` is (output, kind, path, type, domain) — `kind`, not
    # `artifact_kind`, which is what the *reply* calls it. The two spellings
    # are one of this client's rougher edges.
    kinds: dict[str, list[str]] = {}
    for art in artifacts:
        kinds.setdefault(str(art.kind), []).append(str(art.output))

    trace = [a for a in artifacts if str(a.kind) == "assembly_simulation_json"]
    print()
    for kind, names in sorted(kinds.items()):
        shown = sorted(names)
        tail = "" if len(shown) <= 6 else f" …(+{len(shown)-6})"
        print(f"  {kind:34s} {', '.join(shown[:6])}{tail}")

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
