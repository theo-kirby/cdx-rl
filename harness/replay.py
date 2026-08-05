"""``replay`` — take a trained result to another machine and open it there.

The seventh driver, and the first whose output is a *project* rather than a
number or a picture. ``capture`` renders `mjVIS_INERTIA` boxes on this box in
two seconds; this puts the real CAD solids in front of you on the Mac, in the
Shell, where the mouse can shove the machine and the policy answers.

```
# on sb1x
uv run python -m harness replay --export --dir jobs/stand13-20260805-135926 \\
    --iteration 1800 --arm clamp25 --preflight
# → replay/stand13.001800/  +  a verdict, before anything is transferred

# on the Mac
uv run python -m harness replay --import replay/stand13.001800 --arm clamp25
uv run python mechanisms/mg-legs/rollout/build.py --arm clamp25
```

Exit codes are the package's: ``0`` fine, ``1`` infrastructure, ``2`` usage,
``3`` the engine refused.

## What a replay set is, and why it is not a project store

Five things, about 380 KB: the ``.cxpolicy``, the **training** task bundle, the
**training** MJCF, the repo-relative path of the script that authors the arm,
and a manifest of digests. Shipping a whole ``.cadex`` store would also work
and is what the fallback did — but it makes the far machine a pure consumer.
The rebuild *is* the point: it is what produces the BREP solids the Shell
renders, and a store full of somebody else's BREPs cannot be edited, varied or
re-derived.

The training bundle and the training MJCF travel because ADR-134 needs them.
They are what ``assembly.policy(..., trained_task=)`` binds the policy to,
whole-file and unweakened, before proving the locally built bundle equivalent.
The model is found by **digest**, not name — the manifest records it, and the
bundle records it too, so the filename is free to be anything.

## `--preflight` is the part that earns its keep

It answers *"will this replay on the Mac?"* **here**, by doing the thing the
Mac will do: install the set into a scratch project and build the arm's script
through `cadexd`. Accept or refuse, with the engine's own failure envelope.

That prediction is only sound because of ADR-133. Before the inertial snap, a
build on sb1x and a build on the Mac produced different MJCF bytes, so a local
accept said nothing about a remote one. After it, the two are byte-identical —
measured, on this mechanism, both boxes — which is what makes a local answer a
real answer rather than an encouraging one.

## Every set owes a Flywheel node

The same obligation, ledger and reason as ``capture``: cdx-rl's nodes are
MCP-owned, the CLI holds a different account and returns 403 against them
(`flywheel.md` §5), and a Python subprocess has no MCP. So the driver records
what it owes, prints the outstanding count on every run, and hands you the
exact ``prepare_artifact_uploads`` items with ``--pending``.

``--scp`` prints the one-liner instead, for when the graph is not wanted: it is
faster and it leaves no record, and that trade is the caller's to make out
loud.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from cadexd_client import (  # noqa: E402
    DEFAULT_WORKER_CPU_SECONDS,
    DEFAULT_WORKER_MEMORY_MB,
    CadexdClient,
    CadexdError,
    Engine,
    EngineError,
    ScriptRefused,
    accepted_artifacts,
)

from harness import (  # noqa: E402
    EXIT_INFRASTRUCTURE,
    EXIT_OK,
    EXIT_REFUSED,
    EXIT_USAGE,
)
from harness.episodes import BundleError, load_bundle  # noqa: E402
from harness.provenance import envelope, load_env_file, sha256_file, utc_now  # noqa: E402

#: Where exported and imported sets live. Gitignored: a set is 380 KB of
#: derived bytes whose originals are in ``jobs/`` and ``tasks/``, and the
#: record that matters is the Flywheel node.
SETS = REPO_ROOT / "replay"

SET_SCHEMA = "cdxrl-replay-set-v1"
LEDGER = SETS / "ledger.json"
LEDGER_SCHEMA = "cdxrl-replay-ledger-v1"

#: The three files that travel, by role. The manifest names each one's actual
#: filename, so a set is readable without guessing.
ROLES = ("policy", "task", "model")


class ReplayError(RuntimeError):
    """A set that could not be built, read or trusted."""


# ---------------------------------------------------------------------------
# The set: build one, read one, check one. All pure — no engine, no MuJoCo.
# ---------------------------------------------------------------------------


def checkpoint_for(run_dir: Path, iteration: int | None) -> Path:
    """The ``.cxpolicy`` tagged ``iteration``, or the last one written.

    **By the FILENAME TAG, which is what ``capture`` uses and what every table
    in this repository means by "iteration 1800".** There are two conventions
    on disk and they differ by one: ``series_checkpoints`` reads the tag, so
    ``stand13.001800.cxpolicy`` is 1800, while ``discover_policies`` reads the
    trainer's own index out of ``progress.json``, where the same file is 1799.

    Neither is wrong and both are needed — the trainer's index is what a reward
    curve is plotted against, and the tag is what a filename says. What would
    be wrong is two drivers disagreeing about what ``--iteration 1800`` means,
    so this one is pinned to ``capture``'s answer and
    ``test_replay.py::test_the_iteration_convention_matches_capture`` is what
    keeps it there. The first version of this driver used the other one and
    refused ``--iteration 1800`` on a run that has exactly that file.
    """

    from harness.episodes import series_checkpoints  # noqa: PLC0415

    available = series_checkpoints(run_dir, 0)
    if iteration is None:
        return available[-1][1]
    match = [path for tag, path in available if tag == iteration]
    if not match:
        raise ReplayError(
            f"{run_dir} has no checkpoint tagged {iteration}. It has: "
            + ", ".join(str(tag) for tag, _ in available)
        )
    return match[0]


def model_beside(bundle: dict[str, Any], search: list[Path]) -> Path:
    """The training MJCF, by the digest the bundle records.

    ``episodes.resolve_model`` does this for a run directory; this takes an
    explicit list because a training bundle in ``tasks/`` keeps its model
    beside itself and a run directory keeps a copy too, and the two are the
    same bytes or one of them is wrong.
    """

    declared = str((bundle.get("model") or {}).get("sha256") or "")
    if not declared:
        raise ReplayError(
            "The task bundle records no model.sha256, so the model it was "
            "built from cannot be identified. A bundle without one is not a "
            "bundle a policy can be checked against."
        )
    for path in search:
        if path.is_file() and sha256_file(path) == declared:
            return path
    looked = ", ".join(str(path) for path in search)
    raise ReplayError(
        f"No model digesting to {declared[:12]}… beside the bundle or the run. "
        f"Looked at: {looked}. Shipping a policy with a model it was not "
        "trained against would make the far machine's equivalence check "
        "compare the wrong two mechanisms."
    )


def manifest_for(
    *,
    label: str,
    arm: str,
    script: Path,
    policy: Path,
    task: Path,
    model: Path,
    run_dir: Path | None,
    iteration: int | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The manifest, with a digest for every travelling byte.

    ``script`` is recorded as a **repo-relative path and a digest**, and the
    file itself does not travel: it is committed, so the far machine has it
    already, and shipping a copy would let a set be built against a script
    that is not the one in git. The digest is what says which revision.
    """

    bundle, task_digest = load_bundle(task)
    return {
        "schema": SET_SCHEMA,
        "label": str(label),
        "arm": str(arm),
        "script": script.resolve().relative_to(REPO_ROOT).as_posix(),
        "script_sha256": sha256_file(script),
        "policy": _entry(policy),
        "task": _entry(task),
        "model": _entry(model),
        # Restated from the bundle so a reader can check the set is coherent
        # without parsing 30 KB of JSON. The far machine's engine finds the
        # model by this digest, not by filename.
        "task_model_sha256": str((bundle.get("model") or {}).get("sha256") or ""),
        "task_label": str(bundle.get("label") or ""),
        "run": {
            "dir": run_dir.resolve().as_posix() if run_dir else None,
            "iteration": iteration,
            "seed_trained": _trained_seed(run_dir),
        },
        **(extra or {}),
        "exported_utc": utc_now(),
    }


def _entry(path: Path) -> dict[str, Any]:
    return {
        "file": path.name,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _trained_seed(run_dir: Path | None) -> int | None:
    """The seed the POLICY was trained at, from the run's own hyperparameters.

    Never written as a bare ``seed``: one decides which policy you got and
    drives replication, the other decides which scenario you played it on and
    drives the *n* (`flywheel-conventions.md` §3).
    """

    if run_dir is None:
        return None
    path = run_dir / "hyperparameters.json"
    if not path.is_file():
        return None
    try:
        return int(json.loads(path.read_text(encoding="utf-8"))["seed"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def read_set(set_dir: Path) -> dict[str, Any]:
    """The manifest of a set on disk, with its schema checked."""

    path = set_dir / "manifest.json"
    if not path.is_file():
        raise ReplayError(f"{set_dir} has no manifest.json, so it is not a "
                          "replay set.")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReplayError(f"{path} is not readable JSON: {exc}") from exc
    if str(manifest.get("schema") or "") != SET_SCHEMA:
        raise ReplayError(
            f"{path} declares schema {manifest.get('schema')!r}; this driver "
            f"reads {SET_SCHEMA!r}."
        )
    return manifest


def verify_set(set_dir: Path, manifest: dict[str, Any]) -> list[str]:
    """Every travelling file present and digesting to what the manifest says.

    Returns the complaints rather than raising, so an import reports all three
    at once: "the policy is fine and the model is truncated" is a more useful
    sentence than the first half of it.

    The script is checked too, and a mismatch there is a **warning shape**
    rather than a missing file: the far machine's checkout may simply be at a
    different commit, which is worth saying and is not always wrong.
    """

    complaints: list[str] = []
    for role in ROLES:
        entry = manifest.get(role) or {}
        path = set_dir / str(entry.get("file") or "")
        if not path.is_file():
            complaints.append(f"{role}: {entry.get('file')!r} is missing")
            continue
        found = sha256_file(path)
        if found != str(entry.get("sha256") or ""):
            complaints.append(
                f"{role}: {path.name} digests to {found[:12]}…, manifest says "
                f"{str(entry.get('sha256'))[:12]}…"
            )
        elif int(entry.get("bytes") or -1) != path.stat().st_size:
            complaints.append(
                f"{role}: {path.name} is {path.stat().st_size} bytes, manifest "
                f"says {entry.get('bytes')}"
            )
    return complaints


def script_drift(manifest: dict[str, Any]) -> str | None:
    """Whether the committed script differs from the one the set was made with.

    Not an error. The script is committed, so both machines have *a* copy, and
    the interesting case — a set exported before a script edit — is a
    difference worth printing rather than a refusal. The build that follows is
    the real check, and it is the one that can actually tell.
    """

    path = REPO_ROOT / str(manifest.get("script") or "")
    if not path.is_file():
        return f"{manifest.get('script')} is not in this checkout"
    found = sha256_file(path)
    declared = str(manifest.get("script_sha256") or "")
    if declared and found != declared:
        return (f"{path.name} digests to {found[:12]}… here and "
                f"{declared[:12]}… where the set was exported")
    return None


def export_set(
    *,
    run_dir: Path | None,
    iteration: int | None,
    arm: str,
    script: Path,
    task: Path,
    policy: Path | None,
    label: str | None,
    destination: Path | None,
) -> tuple[Path, dict[str, Any]]:
    """Assemble one replay set on disk. Returns its directory and manifest."""

    if policy is None:
        if run_dir is None:
            raise ReplayError("--export needs either --dir or --policy.")
        policy = checkpoint_for(run_dir, iteration)
    bundle, _digest = load_bundle(task)
    model = model_beside(
        bundle,
        [task.parent / name for name in ("model-model.xml",)]
        + sorted(task.parent.glob("*model*.xml"))
        + ([run_dir / "model-model.xml"] if run_dir else [])
        + (sorted(run_dir.glob("*model*.xml")) if run_dir else []),
    )
    name = label or policy.name[: -len(".cxpolicy")]
    set_dir = destination or (SETS / name)
    set_dir.mkdir(parents=True, exist_ok=True)

    # The task and the model are renamed onto the **arm**, and the policy is
    # not. Three decisions, and the middle one is load-bearing:
    #
    # * the policy keeps the filename the run wrote, because that is how a
    #   checkpoint is identified everywhere else in this repository;
    # * the bundle and the model are named for the arm rather than for the
    #   checkpoint, because the script names the bundle **literally** in
    #   ``assembly.policy(..., trained_task=)`` and the script is committed.
    #   Every checkpoint of one arm trained against one bundle, so an
    #   arm-named file is stable and a label-named one would break the
    #   committed script every time the checkpoint moved;
    # * they are renamed at all because they arrive as ``stand-task.json`` and
    #   ``model-model.xml`` from whichever attempt produced them, and two sets
    #   side by side would be unreadable.
    stem = arm or name
    staged = {
        "policy": set_dir / policy.name,
        "task": set_dir / f"{stem}-task.json",
        "model": set_dir / f"{stem}-model.xml",
    }
    for source, target in ((policy, staged["policy"]), (task, staged["task"]),
                           (model, staged["model"])):
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)

    manifest = manifest_for(
        label=name, arm=arm, script=script,
        policy=staged["policy"], task=staged["task"], model=staged["model"],
        run_dir=run_dir, iteration=iteration,
        extra={"source": {
            "policy": str(policy.resolve()),
            "task": str(task.resolve()),
            "model": str(model.resolve()),
        }},
    )
    (set_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return set_dir, manifest


def import_set(source: Path, destination: Path | None = None) -> tuple[Path, dict]:
    """Copy a set into ``replay/<label>/``, checking every digest on arrival.

    A digest checked at the far end is the point: a transfer that silently
    truncated a ``.cxpolicy`` would otherwise surface as ``decode_policy``
    refusing a container three steps later, which reads like a corrupt file
    rather than a bad copy.
    """

    manifest = read_set(source)
    complaints = verify_set(source, manifest)
    if complaints:
        raise ReplayError(
            f"{source} does not match its own manifest:\n  "
            + "\n  ".join(complaints)
        )
    target = destination or (SETS / str(manifest["label"]))
    if source.resolve() != target.resolve():
        target.mkdir(parents=True, exist_ok=True)
        for role in ROLES:
            name = str((manifest.get(role) or {})["file"])
            shutil.copy2(source / name, target / name)
        shutil.copy2(source / "manifest.json", target / "manifest.json")
    arrived = verify_set(target, manifest)
    if arrived:
        raise ReplayError(
            f"The copy at {target} does not match the manifest:\n  "
            + "\n  ".join(arrived)
        )
    return target, manifest


def find_set(name: str) -> Path:
    """A set by label, by arm, or by path. In that order, and it says which."""

    direct = Path(name)
    if (direct / "manifest.json").is_file():
        return direct
    by_label = SETS / name
    if (by_label / "manifest.json").is_file():
        return by_label
    if SETS.is_dir():
        by_arm = [
            candidate
            for candidate in sorted(SETS.iterdir())
            if (candidate / "manifest.json").is_file()
            and str(read_set(candidate).get("arm") or "") == name
        ]
        if len(by_arm) == 1:
            return by_arm[0]
        if len(by_arm) > 1:
            raise ReplayError(
                f"{len(by_arm)} sets declare arm {name!r}: "
                + ", ".join(candidate.name for candidate in by_arm)
                + ". Name one by its label."
            )
    raise ReplayError(f"No replay set at {name!r}. `--list` shows what there is.")


def sets_on_disk() -> list[tuple[Path, dict[str, Any]]]:
    if not SETS.is_dir():
        return []
    found = []
    for candidate in sorted(SETS.iterdir()):
        if (candidate / "manifest.json").is_file():
            try:
                found.append((candidate, read_set(candidate)))
            except ReplayError:
                continue
    return found


# ---------------------------------------------------------------------------
# The build, which is also the preflight. One implementation, deliberately.
# ---------------------------------------------------------------------------


def install_and_build(
    set_dir: Path,
    manifest: dict[str, Any],
    project: Path,
    *,
    script: Path | None = None,
    cpu_seconds: float = DEFAULT_WORKER_CPU_SECONDS,
    memory_mb: int = DEFAULT_WORKER_MEMORY_MB,
    on_line: Any = print,
) -> dict[str, Any]:
    """Install a set's three assets into ``project`` and build its arm's script.

    **This is both the preflight and the real build**, and one implementation
    is the whole point: a preflight that re-implemented the check would be
    predicting its own behaviour rather than the engine's.

    Two orderings matter and both have cost time before:

    * ``put_asset`` moves the project revision even though it is not a write,
      so every asset goes in **before** the script that names them, or the
      write is refused as ``STALE_PROGRAM_REVISION``.
    * the **task bundle and the model** are assets here, not just the policy.
      ADR-134 finds the training model by digest among the project's assets, so
      a set that installed only the ``.cxpolicy`` would be refused for a
      missing model with a perfectly good one sitting in ``replay/``.
    """

    source = script or (REPO_ROOT / str(manifest["script"]))
    if not source.is_file():
        raise ReplayError(f"{source} is not in this checkout.")
    text = source.read_text(encoding="utf-8")

    engine = Engine.resolve()
    project.parent.mkdir(parents=True, exist_ok=True)
    with CadexdClient(engine) as client:
        client.open_project(project, cpu_seconds=cpu_seconds,
                            memory_mb=memory_mb)
        client.require_dynamics()
        for role in ROLES:
            name = str((manifest.get(role) or {})["file"])
            client.put_asset(set_dir / name)
            on_line(f"asset     {name}")
        reply = client.write_script(text)
    return {
        "engine": engine.describe(),
        "project": str(project),
        "digest": str(reply.get("digest") or ""),
        "outputs": [str(item.get("name") or "")
                    for item in (reply.get("outputs") or [])],
    }


def receipt_of(project: Path) -> dict[str, Any] | None:
    """The accepted attempt's policy receipt, which carries the equivalence.

    ADR-134 folds an ``equivalence`` block into the receipt when the policy
    travelled with its own bundle, and its **absence** is a record too: it says
    the policy was checked whole-file against the bundle the script built,
    which is the older and stricter claim.
    """

    for art in accepted_artifacts(project):
        if str(art.kind) == "assembly_policy_receipt_json":
            try:
                return json.loads(Path(str(art.path)).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
    return None


# ---------------------------------------------------------------------------
# The ledger: every set owes a Flywheel node
# ---------------------------------------------------------------------------


def read_ledger() -> dict[str, Any]:
    if not LEDGER.is_file():
        return {"schema": LEDGER_SCHEMA, "sets": {}}
    try:
        ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"schema": LEDGER_SCHEMA, "sets": {}}
    ledger.setdefault("sets", {})
    return ledger


def write_ledger(ledger: dict[str, Any]) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")


def upload_items(set_dir: Path, manifest: dict[str, Any],
                 note: str | None = None) -> list[dict[str, Any]]:
    """The ``prepare_artifact_uploads`` items for one set's four files.

    A ``.cxpolicy`` uploads as ``binary`` and a ``table`` must be JSON
    (`flywheel.md` §5), so the bundle and the manifest go as ``table`` and the
    MJCF — which is XML — goes as ``binary`` beside the weights.

    The notes say what the bytes *mean*, which is §5's rule. What no driver can
    know is why this checkpoint, and that is what ``--note`` is for.
    """

    label = str(manifest["label"])
    seed_trained = (manifest.get("run") or {}).get("seed_trained")
    iteration = (manifest.get("run") or {}).get("iteration")
    common = {
        "driver": "replay",
        "label": label,
        "arm": str(manifest.get("arm") or ""),
        "seed_trained": seed_trained,
        "iteration": iteration,
        "script": str(manifest.get("script") or ""),
        "script_sha256": str(manifest.get("script_sha256") or ""),
    }
    meaning = (
        f"Replay set {label}: everything needed to rebuild this policy's "
        f"mechanism on another machine and open it in the Shell on the real "
        f"CAD solids. The task bundle and the MJCF are the ones the policy was "
        f"TRAINED on, which is what assembly.policy(..., trained_task=) binds "
        f"it to whole-file before proving the locally built bundle equivalent "
        f"(ADR-134). The model is resolved by digest, so its filename does not "
        f"matter. The script is NOT here: it is committed, and "
        f"{manifest.get('script')} at {str(manifest.get('script_sha256'))[:12]}… "
        f"is the revision this set was exported against."
    )
    # One note for all four, ``--note`` first: the caller's sentence is why
    # this checkpoint, and it is the part that goes stale least. Every item
    # carries it, including the manifest -- a set is published as one batch and
    # a reader who opened the manifest and not the weights would otherwise get
    # the machine's half and none of the caller's.
    body = f"{note}\n\n{meaning}" if note else meaning
    items = []
    for role, artifact_type, media in (
        ("policy", "binary", "application/octet-stream"),
        ("task", "table", "application/json"),
        ("model", "binary", "application/xml"),
    ):
        entry = manifest[role]
        items.append({
            "artifact_type": artifact_type,
            "filename": str(entry["file"]),
            "media_type": media,
            "title": f"{label} — {role}",
            "note": body,
            "metadata": {**common, "role": role,
                         "sha256": str(entry["sha256"]),
                         "bytes": int(entry["bytes"])},
        })
    items.append({
        "artifact_type": "table",
        "filename": "manifest.json",
        "media_type": "application/json",
        "title": f"{label} — replay manifest",
        "note": body,
        "metadata": {**common, "role": "manifest",
                     "sha256": sha256_file(set_dir / "manifest.json"),
                     "bytes": (set_dir / "manifest.json").stat().st_size},
    })
    return items


def record(set_dir: Path, manifest: dict[str, Any],
           items: list[dict[str, Any]]) -> dict[str, Any]:
    """Enter one set in the ledger as owing a node."""

    ledger = read_ledger()
    label = str(manifest["label"])
    existing = ledger["sets"].get(label, {})
    fingerprint = [item["metadata"]["sha256"] for item in items]
    ledger["sets"][label] = {
        "set_dir": str(set_dir),
        "exported_utc": str(manifest.get("exported_utc") or ""),
        "upload_items": items,
        # A re-export under the same label is a NEW set of bytes and owes a new
        # artifact, even if the old one was published: keeping the node id
        # would claim the graph holds these bytes when it holds others. Same
        # rule ``capture``'s ledger keeps, and for the same reason.
        "flywheel": (
            existing.get("flywheel", {})
            if [item["metadata"]["sha256"]
                for item in existing.get("upload_items", [])] == fingerprint
            else {"node_id": None, "published_utc": None}
        ),
    }
    write_ledger(ledger)
    return ledger


def pending(ledger: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Sets with no node yet — and still on disk to upload."""

    ledger = ledger if ledger is not None else read_ledger()
    return [entry for entry in ledger["sets"].values()
            if not (entry.get("flywheel") or {}).get("node_id")
            and Path(entry["set_dir"]).is_dir()]


def print_pending() -> int:
    ledger = read_ledger()
    outstanding = pending(ledger)
    gone = [entry for entry in ledger["sets"].values()
            if not (entry.get("flywheel") or {}).get("node_id")
            and not Path(entry["set_dir"]).is_dir()]
    if gone:
        print(f"{len(gone)} unpublished set(s) no longer on disk — deleted "
              f"before they reached the graph:")
        for entry in gone:
            print(f"  {entry['set_dir']}  (exported {entry['exported_utc']})")
        print()
    if not outstanding:
        print("Every recorded replay set still on disk is on a Flywheel node.")
        for label, entry in ledger["sets"].items():
            node = (entry.get("flywheel") or {}).get("node_id")
            if node and Path(entry["set_dir"]).is_dir():
                print(f"  {label:<28} {node}")
        return EXIT_OK
    print(f"{len(outstanding)} replay set(s) recorded and NOT YET on the graph:")
    for entry in outstanding:
        total = sum(int(item["metadata"]["bytes"])
                    for item in entry["upload_items"])
        print(f"  {entry['set_dir']}  {len(entry['upload_items'])} files, "
              f"{total / 1e3:.0f} kB")
    print()
    print("items[] for flywheel_prepare_artifact_uploads:")
    print(json.dumps(
        [item for entry in outstanding for item in entry["upload_items"]],
        indent=2,
    ))
    print()
    print("Then PUT each file's RAW BYTES to its upload_url (expect 202), "
          "finalize the batch,\nand record it:")
    print("  uv run python -m harness replay --mark-published <node_id>")
    print("\nWrite with the identity that OWNS the node — cdx-rl's nodes are "
          "MCP-owned and\nthe CLI returns 403 against them "
          "(flywheel.md §5). That is why this driver cannot\ndo it for you.")
    return EXIT_OK


def mark_published(node_id: str) -> int:
    ledger = read_ledger()
    targets = [label for label, entry in ledger["sets"].items()
               if entry in pending(ledger)]
    if not targets:
        print("Nothing pending.")
        return EXIT_OK
    for label in targets:
        ledger["sets"][label]["flywheel"] = {"node_id": node_id,
                                             "published_utc": utc_now()}
    write_ledger(ledger)
    print(f"{len(targets)} replay set(s) recorded as artifacts of {node_id}.")
    return EXIT_OK


def scp_line(set_dir: Path, host: str) -> str:
    """The transport with no record, printed rather than run.

    Printed because it is the caller's trade to make: it is faster and it
    leaves nothing on the graph, and a driver that ran it quietly would be
    choosing "no record" on the caller's behalf.
    """

    return (f"scp -r {set_dir} {host}:~/cdx-rl/replay/ && "
            f"ssh {host} 'cd ~/cdx-rl && "
            f"uv run python -m harness replay --import replay/{set_dir.name}'")


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harness replay",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--export", action="store_true",
                      help="assemble a replay set from a finished run")
    mode.add_argument("--import", dest="import_from", metavar="SRC",
                      help="bring a set in, checking every digest on arrival")
    mode.add_argument("--preflight", metavar="SET", nargs="?", const="",
                      help="build a set's arm here and report the verdict the "
                           "far machine will reach")
    mode.add_argument("--list", action="store_true",
                      help="the sets on disk, and their Flywheel nodes")
    mode.add_argument("--pending", action="store_true",
                      help="sets recorded and not yet on the graph")
    mode.add_argument("--mark-published", metavar="NODE_ID",
                      help="record every pending set as an artifact of NODE_ID")

    parser.add_argument("--dir", type=Path,
                        help="the training run directory (--export)")
    parser.add_argument("--iteration", type=int,
                        help="which checkpoint; default is the last written")
    parser.add_argument("--policy", type=Path,
                        help="an explicit .cxpolicy, instead of --dir/--iteration")
    parser.add_argument("--task", type=Path,
                        help="the TRAINING task bundle the policy was trained on")
    parser.add_argument("--arm", default="",
                        help="the arm this set replays, e.g. clamp25")
    parser.add_argument("--script", type=Path,
                        help="the script that authors the arm (--export)")
    parser.add_argument("--label", help="the set's name; default is the "
                                        "checkpoint's")
    parser.add_argument("--set", dest="set_dir", type=Path,
                        help="write the set here instead of replay/<label>/")
    parser.add_argument("--project", type=Path,
                        help="the project a preflight builds into")
    parser.add_argument("--scp", metavar="HOST",
                        help="print the scp one-liner for HOST and stop")
    parser.add_argument("--note", help="why this checkpoint — the part no "
                                       "driver can know")
    parser.add_argument("--worker-cpu-seconds", type=float,
                        default=DEFAULT_WORKER_CPU_SECONDS)
    parser.add_argument("--worker-memory-mb", type=int,
                        default=DEFAULT_WORKER_MEMORY_MB)
    parser.add_argument("--json", action="store_true",
                        help="the envelope on stdout, and nothing else")
    return parser


def _print_set(set_dir: Path, manifest: dict[str, Any]) -> None:
    print(f"set       {set_dir}")
    print(f"label     {manifest['label']}   arm {manifest.get('arm') or '—'}")
    for role in ROLES:
        entry = manifest[role]
        print(f"{role:9s} {entry['file']:<34} {str(entry['sha256'])[:12]}…  "
              f"{entry['bytes']} bytes")
    print(f"script    {manifest.get('script')}  "
          f"{str(manifest.get('script_sha256'))[:12]}…")
    run = manifest.get("run") or {}
    print(f"run       iteration {run.get('iteration')}, "
          f"seed_trained {run.get('seed_trained')}")


def main(argv: list[str] | None = None) -> int:
    load_env_file()
    args = build_parser().parse_args(argv)

    try:
        if args.list:
            found = sets_on_disk()
            if not found:
                print(f"No replay sets in {SETS}.")
                return EXIT_OK
            ledger = read_ledger()
            for set_dir, manifest in found:
                node = ((ledger["sets"].get(str(manifest["label"])) or {})
                        .get("flywheel") or {}).get("node_id")
                print(f"{manifest['label']:<28} {manifest.get('arm') or '—':<10} "
                      f"{node or 'NOT ON THE GRAPH'}")
            return EXIT_OK

        if args.pending:
            return print_pending()

        if args.mark_published:
            return mark_published(args.mark_published)

        if args.export:
            if not args.task:
                print("--export needs --task: the bundle the policy was "
                      "trained on, which is the thing that travels.",
                      file=sys.stderr)
                return EXIT_USAGE
            script = args.script
            if script is None:
                print("--export needs --script: the file that authors this "
                      "arm. It does not travel; its digest does.",
                      file=sys.stderr)
                return EXIT_USAGE
            set_dir, manifest = export_set(
                run_dir=args.dir, iteration=args.iteration, arm=args.arm,
                script=script, task=args.task, policy=args.policy,
                label=args.label, destination=args.set_dir,
            )
            items = upload_items(set_dir, manifest, args.note)
            record(set_dir, manifest, items)
            if not args.json:
                _print_set(set_dir, manifest)
                print()
                if args.scp:
                    print(scp_line(set_dir, args.scp))
                else:
                    print(f"To ship it:  --scp <host>   or publish it: "
                          f"{len(items)} artifacts owed "
                          f"(--pending prints them)")
            outstanding = len(pending())
            if outstanding:
                print(f"\n{outstanding} replay set(s) not yet on the graph — "
                      f"`--pending` prints the payload.",
                      file=sys.stderr if args.json else sys.stdout)
            if args.json:
                print(json.dumps(envelope("replay", manifest), indent=2))
            return EXIT_OK

        if args.import_from:
            target, manifest = import_set(Path(args.import_from), args.set_dir)
            if not args.json:
                _print_set(target, manifest)
                drift = script_drift(manifest)
                print()
                if drift:
                    print(f"NOTE      {drift}")
                    print("          The build below is what can actually "
                          "tell; this is only a heads-up.")
                else:
                    print("script    matches this checkout")
                print(f"\nNext:  uv run python -m harness replay --preflight "
                      f"{manifest['label']}")
            else:
                print(json.dumps(envelope("replay", manifest), indent=2))
            return EXIT_OK

        # --preflight
        name = args.preflight or args.arm
        if not name:
            print("--preflight needs a set: its label, its arm, or its path. "
                  "`--list` shows what there is.", file=sys.stderr)
            return EXIT_USAGE
        set_dir = find_set(name)
        manifest = read_set(set_dir)
        complaints = verify_set(set_dir, manifest)
        if complaints:
            print("This set does not match its own manifest:\n  "
                  + "\n  ".join(complaints), file=sys.stderr)
            return EXIT_INFRASTRUCTURE
        project = args.project or (
            REPO_ROOT / "projects" / f"replay-{manifest['label']}.cadex"
        )
        drift = script_drift(manifest)
        if drift and not args.json:
            print(f"NOTE      {drift}")
        result = install_and_build(
            set_dir, manifest, project,
            script=args.script,
            cpu_seconds=args.worker_cpu_seconds,
            memory_mb=args.worker_memory_mb,
            on_line=(lambda line: None) if args.json else print,
        )
        receipt = receipt_of(project)
        payload = {**result, "set": str(set_dir), "manifest": manifest,
                   "receipt": receipt}
        if args.json:
            print(json.dumps(envelope("replay", payload), indent=2))
            return EXIT_OK
        print(f"engine    {result['engine'].get('root')}")
        print(f"script    accepted, digest {result['digest'][:12]}…, "
              f"{len(result['outputs'])} outputs")
        equivalence = (receipt or {}).get("equivalence")
        if equivalence:
            print()
            print("EQUIVALENCE PROVED (ADR-134)")
            print(f"  trained on      {equivalence['trained_task']}  "
                  f"{equivalence['trained_task_sha256'][:12]}…")
            print(f"  script built    "
                  f"{equivalence['script_task_sha256'][:12]}…")
            print(f"  same task       "
                  f"{equivalence['task_semantic_sha256'][:12]}…  "
                  f"(semantic digest)")
            print(f"  models          {equivalence['trained_model']} vs the "
                  f"script's, within {equivalence['model_field_tolerance']:g}")
        elif receipt:
            print("\nThe policy verified against the bundle this script built, "
                  "whole-file.\nNo trained_task= in the script, so ADR-134 was "
                  "not used — which is the\nstricter claim and needs the two "
                  "bundles to be byte-identical.")
        print(f"\nWILL REPLAY. Project at {project}")
        return EXIT_OK

    except (ReplayError, BundleError) as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE
    except ScriptRefused as exc:
        reply = getattr(exc, "reply", {}) or {}
        print("REFUSED — and this is the verdict the far machine would reach "
              "too.\n", file=sys.stderr)
        print(json.dumps(reply, indent=2), file=sys.stderr)
        return EXIT_REFUSED
    except (CadexdError, EngineError, OSError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_INFRASTRUCTURE


if __name__ == "__main__":
    raise SystemExit(main())
