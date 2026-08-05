"""``capture`` — watch the policy execute, as an MP4 with the load on it.

Every result this repository has published is a number. ``harness steps``
says 18/24, ``hazard15.py`` says 12.6 % duty, and until this driver nobody
had ever *watched* the machine.

    uv run python -m harness capture --dir jobs/stand13-20260805-135926 \\
        --iteration 1800 --seeds 4 --tile
    # → video/stand13.001800.tile.mp4  +  video/stand13.001800.tile.json

Exit codes are the package's: ``0`` fine, ``1`` infrastructure, ``2`` usage.

## It is the same episode the table scored

The frames come out of ``CadexDynamics.evaluate_episode``'s ``sample`` hook —
the loop ``harness steps`` and ``hazard15.py`` already play — with the run's
own bundle and the run's own seeds. ``--seed 3`` is the episode ``steps``
scored at index 3, and ``--quiet`` drops the disturbance schedule exactly as
``hazard15`` does, so the clip is the episode that produced the row rather
than a re-enactment of it. Nothing here re-implements physics; it adds
sampling, which is all ``rollout_policy`` added to the same loop.

## What you see is mass, not CAD, and that is deliberate

`stand-b8`'s MJCF has 24 bodies and **5 geoms** — the ground and four
foot/toe pads — so a default render is a floor and a sky. The capture turns
on ``mjVIS_INERTIA``, a **viewer flag**, and draws each body's equivalent
inertia box. Adding visual geoms instead would move the MJCF digest, the task
digest with it, and produce a video of a different machine from the one the
run trained against. ``harness/_capture.py``'s docstring has the measured
numbers.

## The torque strip is what makes it an instrument

Along the bottom of every pane: per-motor ``|data.actuator_force|`` over
``model.actuator_forcerange``, red at 90 % — hazard 15's own threshold. Same
source as ``hazard15.py`` and never ``step["action"]``, which under a
position action space is a joint angle in degrees that
``_episodes._torque_columns`` still labels N·mm. You are watching the bracing
that experiment 004 reports as one number.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from harness import EXIT_INFRASTRUCTURE, EXIT_OK, EXIT_USAGE
from harness._capture import (
    CaptureError,
    JOB_SCHEMA,
    even,
    steps_per_frame,
    tile_layout,
)
from harness.episodes import (
    BundleError,
    load_bundle,
    resolve_model,
    series_checkpoints,
)
from harness.provenance import envelope, load_env_file, sha256_file
from harness.trainer_venv import (
    TrainerVenvError,
    cadex_module_dir,
    check_pins,
    trainer_python,
    unpinned_drift,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).resolve().parent / "_capture.py"

#: The last resort in the ffmpeg search. It is the PR clone's pixi
#: environment, which carries a libx264 build; it is last because it is a
#: path on one box and the three before it are not.
PIXI_FFMPEG = Path("/home/theo/cadex-prs/.pixi/envs/default/bin/ffmpeg")

#: MuJoCo's default offscreen framebuffer, and therefore the default pane.
#: A model can declare a larger one with ``<visual><global offwidth=…>``;
#: this repository's models do not, and the capture will not edit one to get
#: a bigger picture.
DEFAULT_PANE = (640, 480)

#: **Every MP4 this driver records must end up as the artifact of a Flywheel
#: node, and this file is what makes that checkable rather than promised.**
#:
#: The driver cannot publish for itself, and the reason is measured rather
#: than a limitation of effort: nodes cdx-rl creates are owned by the **MCP
#: identity** (`flywheel.md` §5), the CLI holds a different account and
#: returns 403 against them, and a Python subprocess has no MCP. So the
#: obligation is recorded instead — one entry per video, carrying the exact
#: ``prepare_artifact_uploads`` item needed to discharge it — and the driver
#: refuses to let it go quiet: every run prints the outstanding count, and
#: ``--pending`` prints the payload.
#:
#: It lives under ``video/``, which is gitignored, because it is state about
#: files on this box. Once a video is published the *node* is the record;
#: the ledger only remembers which node, so that a second run does not
#: publish it twice.
LEDGER = REPO_ROOT / "video" / "ledger.json"
LEDGER_SCHEMA = "cdxrl-capture-ledger-v1"


def resolve_ffmpeg(explicit: str | None = None) -> tuple[Path, str]:
    """An ffmpeg binary and how it was found. Order is fixed and recorded.

    ``$CDX_FFMPEG`` → ``imageio_ffmpeg`` → ``PATH`` → the pixi build. The
    order runs from *what this invocation asked for* to *what happens to be
    on this box*, and the answer goes into the sidecar because an encoder is
    part of how a file came to look the way it does.

    There is no ffmpeg on this box's ``PATH``, measured 2026-08-05, which is
    why ``imageio-ffmpeg`` is a dependency: it ships a static binary and
    keeps cdx-rl self-contained rather than borrowing the PR clone's.
    """

    if explicit:
        candidate = Path(explicit).expanduser()
        if not candidate.is_file():
            raise CaptureError(f"CDX_FFMPEG names {candidate}, which is not a file")
        return candidate, "CDX_FFMPEG"
    try:
        import imageio_ffmpeg  # noqa: PLC0415 — optional, and probed not assumed

        found = Path(imageio_ffmpeg.get_ffmpeg_exe())
        if found.is_file():
            return found, "imageio_ffmpeg"
    except Exception:  # noqa: BLE001 — absent or unable to unpack its binary
        pass
    on_path = shutil.which("ffmpeg")
    if on_path:
        return Path(on_path), "PATH"
    if PIXI_FFMPEG.is_file():
        return PIXI_FFMPEG, "pixi"
    raise CaptureError(
        "No ffmpeg. Tried $CDX_FFMPEG, imageio_ffmpeg, PATH and "
        f"{PIXI_FFMPEG}. `uv sync` installs imageio-ffmpeg, which ships one."
    )


def ffmpeg_version(binary: Path) -> str:
    """The first line of ``ffmpeg -version``, or why it could not be read."""

    try:
        completed = subprocess.run([str(binary), "-version"], capture_output=True,
                                   text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        return f"unreadable: {exc}"
    return (completed.stdout or completed.stderr).splitlines()[0].strip()


def find_task(run_dir: Path) -> Path:
    """The bundle a run carries. Every ``jobs/*/`` has exactly one.

    Globbed rather than named: experiment 000's pendulum writes
    ``task-task.json`` and the biped writes ``stand-task.json``, and a driver
    that hardcoded either would be mechanism-specific for no reason.
    """

    found = sorted(p for p in run_dir.glob("*task*.json")
                   if p.name not in ("progress.json", "hyperparameters.json"))
    if len(found) != 1:
        raise BundleError(
            f"{run_dir} carries {len(found)} task bundles ({[p.name for p in found]}); "
            "pass --task to say which. A run directory is normally "
            "self-contained — one bundle, one model XML — which is what makes "
            "--dir enough on its own."
        )
    return found[0]


def select_policies(run_dir: Path | None, explicit: list[str],
                    iteration: int | None) -> list[tuple[int | None, Path]]:
    """``[(iteration, path), …]``, from ``--policy`` and/or ``--dir``.

    ``series_checkpoints(…, stride=0)`` is the *whole* periodic list, and it
    is shared with ``hazard15 --series`` rather than globbed a third time.
    """

    chosen: list[tuple[int | None, Path]] = [
        (None, Path(p).expanduser().resolve()) for p in explicit
    ]
    if run_dir is not None:
        available = series_checkpoints(run_dir, 0)
        if iteration is None:
            chosen.append(available[-1])
        else:
            match = [(it, p) for it, p in available if it == iteration]
            if not match:
                raise BundleError(
                    f"{run_dir} has no checkpoint at iteration {iteration}. "
                    f"It has: {', '.join(str(it) for it, _ in available)}."
                )
            chosen.extend(match)
    for _, path in chosen:
        if not path.is_file():
            raise BundleError(f"No policy at {path}")
    return chosen


def default_output(policies: list[tuple[int | None, Path]], tile: bool,
                   quiet: bool) -> Path:
    """``video/<stem>[.quiet][.tile].mp4``. ``video/`` is gitignored."""

    stem = policies[0][1].name.removesuffix(".cxpolicy")
    if len(policies) > 1:
        stem = f"{stem}+{len(policies) - 1}"
    suffix = ("".join([".quiet" if quiet else "", ".tile" if tile else ""]))
    return REPO_ROOT / "video" / f"{stem}{suffix}.mp4"


def trainers_in_flight() -> list[str]:
    """Run directories whose ``train.pid`` still names a live process.

    Not a refusal. EGL rendering uses the card's *graphics* engine for about
    half a second an episode, which will not meaningfully contend with a
    training run — but the sidecar records the host, so the driver says what
    was sharing the box rather than leaving a reader to wonder.
    """

    from harness.runlog import process_gone  # noqa: PLC0415 — cheap, local

    live: list[str] = []
    for pid_file in sorted((REPO_ROOT / "jobs").glob("*/train.pid")):
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        if not process_gone(pid):
            live.append(pid_file.parent.name)
    return live


# ---------------------------------------------------------------------------
# The ledger: every MP4 owes a Flywheel artifact
# ---------------------------------------------------------------------------


def read_ledger() -> dict[str, Any]:
    if not LEDGER.is_file():
        return {"schema": LEDGER_SCHEMA, "videos": {}}
    try:
        ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"schema": LEDGER_SCHEMA, "videos": {}}
    ledger.setdefault("videos", {})
    return ledger


def write_ledger(ledger: dict[str, Any]) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")


def training_seed(run_dir: Path | None) -> int | None:
    """The seed the POLICY was trained at, from the run's own hyperparameters.

    Distinct from the evaluation seeds this driver plays, and the convention
    (`flywheel-conventions.md` §3) is that artifact metadata never says bare
    ``seed``: one decides which policy you got and drives replication, the
    other decides which scenario you played it on and drives the *n*.
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


def upload_item(payload: dict[str, Any], destination: Path,
                seed_trained: int | None, title: str | None,
                note: str | None) -> dict[str, Any]:
    """The ``prepare_artifact_uploads`` item for one MP4, ready to send.

    ``binary``, because Flywheel has no video type and the type is validated
    against the bytes at the PUT. The note states **what the numbers mean**
    rather than what the file is, which is §5's rule, and the part that no
    driver can know — *why this policy* — is what ``--note`` is for.
    """

    rows = payload["episodes"]
    survived = sum(1 for row in rows if row["survived"])
    duty = " / ".join(f"{row['frac_above_90pct_worst'] * 100:.1f}"
                      for row in rows)
    names = sorted({row["policy"] for row in rows})
    measured = (
        f"{', '.join(names)} driving its own episode at "
        f"{len(payload['seeds'])} evaluation seed(s), "
        f"{payload['tile'][0]}x{payload['tile'][1]}, "
        f"{payload['fps']} fps"
        + (" with the disturbance schedule dropped (hazard 15's conditions)"
           if payload["variant"] else "")
        + f". {survived} of {len(rows)} survived the horizon. "
        f"The bars along the bottom of each pane are per-motor "
        f"|actuator_force| over actuator_forcerange, red at 90 % — hazard "
        f"15's own duty threshold; worst-motor duty above 90 % is {duty} % "
        f"of frames, per seed in order. The shapes are mjVIS_INERTIA "
        f"equivalent-inertia boxes, NOT the CAD solids: this MJCF has 5 "
        f"geoms and renders empty without the flag, and adding visual geoms "
        f"would move the digest the policy and the task are bound by."
    )
    return {
        "artifact_type": "binary",
        "filename": destination.name,
        "media_type": "video/mp4",
        "title": title or destination.name,
        "note": measured if not note else f"{note}\n\n{measured}",
        "metadata": {
            "driver": "capture",
            "sha256": sha256_file(destination),
            "bytes": destination.stat().st_size,
            "frames": payload["frames"],
            "fps": payload["fps"],
            "encoded_fps": payload["encoded_fps"],
            "seeds_eval": payload["seeds"],
            "seed_trained": seed_trained,
            "policies": payload["policies"],
            "task_sha256": payload["task_sha256"],
            "model_sha256": payload["model_sha256"],
            "viewer_flags": payload["viewer_flags"],
            "bracing_fraction": payload["bracing_fraction"],
            "mujoco": payload["mujoco"],
            "host": payload["provenance"]["host"]["hostname"],
            "head_commit_sha": payload["provenance"]["cdxrl"].get("commit"),
            "cdxrl_dirty": payload["provenance"]["cdxrl"].get("dirty"),
        },
    }


def record(destination: Path, payload: dict[str, Any],
           item: dict[str, Any]) -> dict[str, Any]:
    """Enter one video in the ledger as owing a node. Returns the ledger."""

    ledger = read_ledger()
    existing = ledger["videos"].get(destination.name, {})
    ledger["videos"][destination.name] = {
        "video": str(destination),
        "sidecar": str(destination.with_suffix(".json")),
        "recorded_utc": payload["provenance"]["timestamp_utc"],
        "upload_item": item,
        # A re-render of the same name is a NEW set of bytes, so it owes a
        # new artifact even if the old one was published. Keeping the old
        # node id would claim the graph holds these bytes when it holds
        # different ones.
        "flywheel": (existing.get("flywheel", {})
                     if existing.get("upload_item", {})
                     .get("metadata", {}).get("sha256")
                     == item["metadata"]["sha256"]
                     else {"node_id": None, "published_utc": None}),
    }
    write_ledger(ledger)
    return ledger


def pending(ledger: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Videos with no node yet — and still on disk to upload."""

    ledger = ledger if ledger is not None else read_ledger()
    return [entry for entry in ledger["videos"].values()
            if not (entry.get("flywheel") or {}).get("node_id")
            and Path(entry["video"]).is_file()]


def print_pending() -> int:
    """The outstanding obligation, as the payload that discharges it."""

    ledger = read_ledger()
    outstanding = pending(ledger)
    # An unpublished video that was deleted is reported rather than dropped.
    # It is no longer an obligation — there are no bytes to upload — but a
    # ledger that quietly forgot it would make "every MP4 is on the graph"
    # true by attrition, which is the wrong way to make a claim true.
    gone = [entry for entry in ledger["videos"].values()
            if not (entry.get("flywheel") or {}).get("node_id")
            and not Path(entry["video"]).is_file()]
    if gone:
        print(f"{len(gone)} unpublished video(s) no longer on disk — deleted "
              f"before they reached the graph:")
        for entry in gone:
            print(f"  {entry['video']}  (recorded {entry['recorded_utc']})")
        print()
    if not outstanding:
        print("Every recorded MP4 still on disk is the artifact of a "
              "Flywheel node.")
        for entry in ledger["videos"].values():
            node = (entry.get("flywheel") or {}).get("node_id")
            if node and Path(entry["video"]).is_file():
                print(f"  {entry['upload_item']['filename']:<34} {node}")
        return EXIT_OK
    print(f"{len(outstanding)} video(s) recorded and NOT YET on the graph:")
    for entry in outstanding:
        item = entry["upload_item"]
        print(f"  {entry['video']}  "
              f"{item['metadata']['bytes'] / 1e6:.2f} MB  "
              f"{item['metadata']['sha256'][:12]}…")
    print()
    print("items[] for flywheel_prepare_artifact_uploads:")
    print(json.dumps([entry["upload_item"] for entry in outstanding], indent=2))
    print()
    print("Then PUT each file's RAW BYTES to its upload_url (expect 202), "
          "finalize the batch,\nand record it:")
    print("  uv run python -m harness capture --mark-published <node_id>")
    print("\nWrite with the identity that OWNS the node — cdx-rl's nodes are "
          "MCP-owned and\nthe CLI returns 403 against them "
          "(flywheel.md §5). That is why this driver cannot\ndo it for you.")
    return EXIT_OK


def mark_published(node_id: str) -> int:
    """Record that every pending video is now an artifact of ``node_id``.

    All of them, not a subset: a batch is one ``finalize`` and one revision
    bump, so "what was just published" and "what was pending" are the same
    list. Publishing to two nodes is two passes, marking after each.
    """

    ledger = read_ledger()
    targets = [entry["upload_item"]["filename"] for entry in pending(ledger)]
    if not targets:
        print("Nothing pending.")
        return EXIT_OK
    from harness.provenance import utc_now  # noqa: PLC0415

    for name in targets:
        ledger["videos"][name]["flywheel"] = {"node_id": node_id,
                                              "published_utc": utc_now()}
    write_ledger(ledger)
    print(f"{len(targets)} video(s) recorded as artifacts of {node_id}.")
    return EXIT_OK


def encode(binary: Path, size: tuple[int, int], fps: int, slowmo: int,
           destination: Path) -> subprocess.Popen[bytes]:
    """ffmpeg, reading raw RGB24 on stdin.

    ``--slowmo N`` divides the *encoded* rate and drops nothing: every frame
    the renderer produced is in the file, played N times slower. Expressed as
    a rational so 50/3 stays exact.

    **``-threads 1`` and ``+bitexact``, and they were paid for.** The first
    five clips this driver published went onto the graph with an ``sha256`` in
    their artifact metadata, and re-rendering them produced *different bytes*
    — 190213 and 190524 for the same episode, back to back. The measurements
    were identical (154 control steps, `collapsed`, the same ten per-motor
    peaks, 30.323 % duty): the **episode** is deterministic and the **encode**
    was not. libx264's frame threading lets rate control depend on thread
    scheduling, and ``+bitexact`` stops the muxer stamping its own version
    into the file, which would move the digest on an ffmpeg upgrade.

    A digest that changes when nothing changed is worse than no digest, and
    the whole point of publishing one is that a reader can check the bytes
    against the claim. Single-threaded costs about a second on a 300-frame
    clip, which is nothing against the ~2 s the whole capture takes.
    """

    width, height = size
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(binary), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{width}x{height}", "-framerate", f"{fps}/{slowmo}",
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        "-preset", "medium", "-threads", "1",
        "-flags:v", "+bitexact", "-fflags", "+bitexact",
        "-movflags", "+faststart",
        str(destination),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def main(argv: list[str]) -> int:
    load_env_file()
    parser = argparse.ArgumentParser(prog="harness capture", description=__doc__)
    parser.add_argument("--dir", help="a run directory; supplies task, model "
                                      "and (with --iteration) the checkpoint")
    parser.add_argument("--policy", nargs="*", default=[],
                        help="explicit .cxpolicy files, played side by side")
    parser.add_argument("--iteration", type=int,
                        help="which periodic checkpoint of --dir; default the last")
    parser.add_argument("--task", help="override the run's own bundle")
    parser.add_argument("--model", help="override the model XML")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--seed-list", help="explicit comma-separated seeds")
    parser.add_argument("--fps", type=int, default=50,
                        help="must divide the task's control_hz (default 50)")
    parser.add_argument("--slowmo", type=int, default=1,
                        help="encode at fps/N, dropping no frames")
    parser.add_argument("--tile", action="store_true",
                        help="lay every episode out in one grid")
    parser.add_argument("--quiet", action="store_true",
                        help="drop the disturbance schedule — hazard 15's "
                             "conditions, nothing pushing")
    parser.add_argument("--track", nargs="?", const="pelvis", default=None,
                        metavar="BODY",
                        help="follow a body instead of holding a world camera")
    parser.add_argument("--azimuth", type=float, default=125.0)
    parser.add_argument("--elevation", type=float, default=-12.0)
    parser.add_argument("--distance", type=float,
                        help="metres; default --distance-scale x the model's "
                             "own extent")
    parser.add_argument("--distance-scale", type=float, default=2.2,
                        help="camera distance as a multiple of model.stat.extent")
    parser.add_argument("--width", type=int, default=DEFAULT_PANE[0])
    parser.add_argument("--height", type=int, default=DEFAULT_PANE[1])
    parser.add_argument("--out", help="destination .mp4")
    parser.add_argument("--title", help="artifact title for the Flywheel node")
    parser.add_argument("--note", help="what this clip is FOR — prepended to "
                                       "the measured note in the ledger entry")
    parser.add_argument("--pending", action="store_true",
                        help="list every recorded MP4 that is not yet the "
                             "artifact of a Flywheel node, with its payload")
    parser.add_argument("--mark-published", metavar="NODE_ID",
                        help="record that the pending videos are now "
                             "artifacts of this node")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    # The ledger modes answer without touching a renderer, a bundle or the
    # trainer venv, so they are handled before anything is resolved.
    if args.pending:
        return print_pending()
    if args.mark_published:
        return mark_published(args.mark_published)

    run_dir = Path(args.dir).expanduser().resolve() if args.dir else None
    if run_dir is None and not args.policy:
        print("capture: nothing to play. Pass --dir RUN or --policy FILE…",
              file=sys.stderr)
        return EXIT_USAGE

    try:
        task_path = (Path(args.task).expanduser().resolve() if args.task
                     else find_task(run_dir))
        bundle, task_sha = load_bundle(task_path)
        policies = select_policies(run_dir, args.policy, args.iteration)
        search = [task_path.parent, *(p.parent for _, p in policies)]
        if run_dir is not None:
            search.insert(0, run_dir)
        candidates: list[Path] = []
        for directory in search:
            for xml in sorted(directory.glob("*model*.xml")):
                if xml not in candidates:
                    candidates.append(xml)
        model = (Path(args.model).expanduser().resolve() if args.model
                 else resolve_model(bundle, candidates))
        stride = steps_per_frame(bundle["episode"]["control_hz"], args.fps)
    except (BundleError, CaptureError) as exc:
        print(f"capture: {exc}", file=sys.stderr)
        return EXIT_USAGE

    if args.slowmo < 1:
        print("capture: --slowmo is a divisor of the frame rate; it is at "
              "least 1.", file=sys.stderr)
        return EXIT_USAGE

    seeds = ([int(s) for s in args.seed_list.split(",")] if args.seed_list
             else list(range(args.seeds)))
    episodes = [{"policy": str(path), "seed": seed,
                 "iteration": iteration}
                for iteration, path in policies for seed in seeds]
    if not episodes:
        print("capture: no episodes — --seeds 0 renders nothing.",
              file=sys.stderr)
        return EXIT_USAGE

    pane = (even(args.width), even(args.height))
    columns, rows = tile_layout(len(episodes)) if args.tile else (1, 1)
    if not args.tile and len(episodes) > 1:
        print(f"capture: {len(episodes)} episodes and no --tile. Pass --tile "
              "to lay them out together, or one seed at a time.",
              file=sys.stderr)
        return EXIT_USAGE
    size = (pane[0] * columns, pane[1] * rows)

    destination = (Path(args.out).expanduser().resolve() if args.out
                   else default_output(policies, args.tile, args.quiet))

    try:
        check_pins()
        interpreter = trainer_python()
        module_dir = cadex_module_dir()
        ffmpeg, how = resolve_ffmpeg(os.environ.get("CDX_FFMPEG"))
    except (TrainerVenvError, CaptureError) as exc:
        print(f"capture: {exc}", file=sys.stderr)
        return EXIT_INFRASTRUCTURE

    in_flight = trainers_in_flight()
    if in_flight:
        print(f"note: {', '.join(in_flight)} still has a live trainer. EGL "
              f"rendering uses the card's graphics engine for ~0.5 s an "
              f"episode; this is recorded, not refused.", file=sys.stderr)

    with tempfile.TemporaryDirectory(prefix="cdxrl-capture-") as scratch:
        summary_path = Path(scratch) / "summary.json"
        spec = {
            "schema": JOB_SCHEMA,
            "module_dir": str(module_dir),
            "model_xml": str(model),
            "task_path": str(task_path),
            "summary_path": str(summary_path),
            "steps_per_frame": stride,
            "pane": list(pane),
            "tile": [columns, rows],
            "variant": {"disturbance": False} if args.quiet else {},
            "camera": {
                "track": args.track,
                "azimuth": args.azimuth,
                "elevation": args.elevation,
                "distance": args.distance,
                "distance_scale": args.distance_scale,
            },
            "episodes": episodes,
        }

        print(f"capture  {len(episodes)} episode(s) at {args.fps} fps "
              f"({stride} control step(s) a frame), {size[0]}x{size[1]}")
        print(f"         {destination}")
        sys.stdout.flush()   # the worker narrates on stderr; keep the order

        # MUJOCO_GL AT THE CALL SITE, never in the environment. It is unset
        # box-wide and there is no precedent for setting it: episode
        # evaluation is headless CPU MuJoCo and opens no renderer, so an
        # exported MUJOCO_GL would change what `compare`, `capability` and
        # `steps` do. This is the one process in the repository that wants a
        # GL context, and it is the only one that gets one.
        environment = dict(os.environ)
        environment["MUJOCO_GL"] = "egl"

        try:
            worker = subprocess.Popen(
                [str(interpreter), str(WORKER)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                env=environment,
            )
        except OSError as exc:
            print(f"capture: could not spawn {WORKER}: {exc}", file=sys.stderr)
            return EXIT_INFRASTRUCTURE

        try:
            encoder = encode(ffmpeg, size, args.fps, args.slowmo, destination)
        except OSError as exc:
            worker.kill()
            print(f"capture: could not spawn {ffmpeg}: {exc}", file=sys.stderr)
            return EXIT_INFRASTRUCTURE

        assert worker.stdin is not None and worker.stdout is not None
        worker.stdin.write((json.dumps(spec) + "\n").encode("utf-8"))
        worker.stdin.flush()
        worker.stdin.close()

        # Pump rather than wiring the two pipes together directly, so that
        # the byte count is checked against the geometry that was declared to
        # ffmpeg. A short frame silently becomes a diagonal smear otherwise.
        written = 0
        assert encoder.stdin is not None
        try:
            while True:
                block = worker.stdout.read(1 << 20)
                if not block:
                    break
                encoder.stdin.write(block)
                written += len(block)
        except BrokenPipeError:
            pass
        finally:
            worker.stdout.close()
            try:
                encoder.stdin.close()
            except BrokenPipeError:
                pass

        worker_rc = worker.wait()
        encoder_rc = encoder.wait()

        if worker_rc != 0:
            print(f"capture: the renderer exited {worker_rc}.", file=sys.stderr)
            return EXIT_USAGE if worker_rc == 2 else EXIT_INFRASTRUCTURE
        if encoder_rc != 0:
            print(f"capture: ffmpeg exited {encoder_rc}.", file=sys.stderr)
            return EXIT_INFRASTRUCTURE

        frame_bytes = size[0] * size[1] * 3
        if written % frame_bytes:
            print(f"capture: the renderer produced {written} bytes, which is "
                  f"not a whole number of {size[0]}x{size[1]} frames.",
                  file=sys.stderr)
            return EXIT_INFRASTRUCTURE

        rendered = json.loads(summary_path.read_text(encoding="utf-8"))

    payload = envelope("capture", True, {
        "video": str(destination),
        "bytes": destination.stat().st_size,
        "frames": written // frame_bytes,
        "fps": args.fps,
        "slowmo": args.slowmo,
        "encoded_fps": round(args.fps / args.slowmo, 4),
        "task": str(task_path),
        "task_sha256": task_sha,
        "model": str(model),
        "model_sha256": sha256_file(model),
        "policies": [
            {"name": path.name, "path": str(path), "iteration": iteration,
             "sha256": sha256_file(path)}
            for iteration, path in policies
        ],
        "seeds": seeds,
        "variant": spec["variant"],
        "camera": spec["camera"],
        "ffmpeg": {"path": str(ffmpeg), "found_by": how,
                   "version": ffmpeg_version(ffmpeg)},
        "trainers_in_flight": in_flight,
        # What the video actually SHOWS. `viewer_flags` is the honest label
        # on the picture: mjVIS_INERTIA means these are equivalent inertia
        # boxes, not the CAD solids, and the model was not touched to make
        # them appear.
        **rendered,
        "unpinned_drift": dict(unpinned_drift),
    })

    sidecar = destination.with_suffix(".json")
    sidecar.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    item = upload_item(payload, destination, training_seed(run_dir),
                       args.title, args.note)
    ledger = record(destination, payload, item)
    outstanding = pending(ledger)

    if args.json:
        # On stderr, because stdout is the envelope. The obligation is
        # printed either way — a machine-readable mode is not a reason for a
        # debt to go unmentioned.
        print(f"FLYWHEEL: {len(outstanding)} recorded MP4(s) not yet the "
              f"artifact of a node; see --pending.", file=sys.stderr)
        print(json.dumps(payload, indent=2))
        return EXIT_OK

    print()
    # The checkpoint column is only printed when there is more than one, and
    # then it is mandatory: two policies at seed 0 are two rows reading `0`,
    # and the reader cannot tell which is which.
    wide = len(policies) > 1
    head = f"{'checkpoint':<30} " if wide else ""
    print(f"{head}{'seed':>5} {'steps':>10} {'reward':>10} {'peak load':>10} "
          f"{'>=90%':>7}  ended")
    for row in rendered["episodes"]:
        name = f"{row['policy']:<30} " if wide else ""
        print(f"{name}{row['seed']:>5} "
              f"{row['step_count']:>4}/{row['max_steps']:<5} "
              f"{row['total_reward']:>10.2f} "
              f"{row['peak_frac_of_forcerange'] * 100:>9.0f}% "
              f"{row['frac_above_90pct_worst'] * 100:>6.1f}%  "
              f"{'survived the horizon' if row['survived'] else row['termination']}")
    print()
    print(f"{written // frame_bytes} frames at {args.fps / args.slowmo:g} fps "
          f"→ {destination}")
    print(f"sidecar {sidecar}")
    print(f"viewer flags {', '.join(rendered['viewer_flags'])} — the shapes "
          f"are equivalent INERTIA BOXES,\n  not the CAD solids. The MJCF was "
          f"not edited: {Path(model).name} is the run's own, "
          f"digest {sha256_file(model)[:12]}…")

    # Never a `--verbose` gate on this. Every MP4 owes a Flywheel node, and a
    # debt that only shows when asked about is a debt nobody pays.
    print()
    print(f"FLYWHEEL: {len(outstanding)} recorded MP4(s) not yet the artifact "
          f"of a node.")
    print("  uv run python -m harness capture --pending    "
          "# the payload that discharges it")
    return EXIT_OK
