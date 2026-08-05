"""``rebuild`` — get a project to a known state, and prove the state is a state.

The thinnest driver, and the one every other one stands on. Almost all of the
work is already in ``tools/cadexd_client.py``; what this adds is a command
line, an envelope, and the assertion in ``--verify``.

```
rebuild --project DIR [--script FILE] [--set k=v]… [--verify] [--json]
```

**Why ``--verify`` is the floor.** Everything this repository records is a
number about a particular build. If two rebuilds of one script produce two
digests, then "the build" is not a thing and no measurement taken against it
means anything. The check costs one extra rebuild.

Compare **digests**, never files: a STEP export writes a wall-clock timestamp
into its ``FILE_NAME`` header, so two exports of an identical model differ
byte for byte across a second boundary while being the same model. The
engine's digest is computed over the script and the collected specs, which is
the thing we actually want to be stable.

On refusal the failure envelope is emitted **verbatim** — ``failure_code``,
``failure_stage``, ``observed``, ``retry``, ``candidates`` — and the exit code
is 3. Those fields are what the next attempt gets written from, and a driver
that reduced them to a message would throw away the diagnosis to keep the
complaint.
"""

from __future__ import annotations

import argparse
import json
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
    accepted_attempt_dir,
)

from harness import EXIT_INFRASTRUCTURE, EXIT_OK, EXIT_REFUSED, EXIT_USAGE  # noqa: E402
from harness.provenance import envelope, load_env_file, sha256_file  # noqa: E402


def parse_set(values: list[str]) -> dict[str, Any]:
    """``k=v`` pairs, with numbers read as numbers.

    A parameter declared ``num()`` and set to the string ``"200"`` is a type
    error the engine reports three steps later, so the guess is made here
    where it is cheap and visible.
    """

    parsed: dict[str, Any] = {}
    for item in values:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"--set expects k=v, got {item!r}")
        text = raw.strip()
        try:
            parsed[key.strip()] = json.loads(text)
        except ValueError:
            parsed[key.strip()] = text
    return parsed


def describe_artifacts(project_root: Path) -> list[dict[str, Any]]:
    """Every artifact of the accepted revision, with digest and size.

    The digest is per *file* here, not the project digest: it is what a graph
    node's artifact record needs, and it is how a run directory's copy of a
    bundle is later shown to be the bundle that was built.
    """

    described = []
    for artifact in accepted_artifacts(project_root):
        record: dict[str, Any] = {
            "output": artifact.output,
            "kind": artifact.kind,
            "path": str(artifact.path),
            "exists": artifact.exists(),
        }
        if artifact.exists():
            record["sha256"] = sha256_file(artifact.path)
            record["bytes"] = artifact.path.stat().st_size
        described.append(record)
    return described


def run(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    project_root = Path(args.project).expanduser().resolve()
    payload: dict[str, Any] = {"project": str(project_root)}

    try:
        engine = Engine.resolve(args.engine)
    except EngineError as exc:
        return EXIT_INFRASTRUCTURE, {**payload, "error": str(exc)}

    client = CadexdClient(engine)
    try:
        client.start()
        opened = client.open_project(
            project_root,
            cpu_seconds=args.worker_cpu_seconds,
            memory_mb=args.worker_memory_mb,
        )
        payload["worker_budgets"] = dict(opened.get("budgets") or {})
        payload["opened_revision"] = client.revision
        # One round trip that turns a stale engine into a loud failure
        # instead of "assembly.mjcf is not defined" from inside a script.
        client.require_dynamics()

        if args.script:
            source = Path(args.script).expanduser().read_text(encoding="utf-8")
            reply = client.write_script(source, replace=args.replace)
            payload["script"] = str(Path(args.script).expanduser().resolve())
        elif args.set:
            reply = client.set_params(parse_set(args.set))
        else:
            reply = client.rebuild()
            if not opened.get("script"):
                return EXIT_USAGE, {
                    **payload,
                    "error": (
                        f"{project_root} has no stored script and none was "
                        "given. Pass --script."
                    ),
                }

        if args.set and args.script:
            reply = client.set_params(parse_set(args.set))

        digest = str(reply.get("digest") or "")
        payload.update({
            "digest": digest,
            "accepted_revision": str(reply.get("revision") or client.revision),
            "outputs": [
                str(item.get("name") or "") for item in (reply.get("outputs") or [])
            ],
        })

        if args.verify:
            second = client.rebuild()
            again = str(second.get("digest") or "")
            payload["verify"] = {
                "first": digest,
                "second": again,
                "stable": bool(digest) and digest == again,
            }

        payload["attempt_dir"] = str(
            accepted_attempt_dir(project_root).relative_to(project_root)
        )
        payload["artifacts"] = describe_artifacts(project_root)

    except ScriptRefused as exc:
        # Verbatim. These fields are the next attempt's input.
        return EXIT_REFUSED, {
            **payload,
            "error": str(exc),
            "failure_code": exc.failure_code,
            "failure_stage": exc.failure_stage,
            **{
                key: exc.reply[key]
                for key in ("observed", "retry", "candidates", "correction", "reason")
                if key in exc.reply
            },
        }
    except (CadexdError, EngineError) as exc:
        return EXIT_INFRASTRUCTURE, {**payload, "error": str(exc)}
    except (OSError, ValueError) as exc:
        return EXIT_INFRASTRUCTURE, {**payload, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        client.shutdown()

    verified = payload.get("verify")
    if verified is not None and not verified["stable"]:
        return EXIT_INFRASTRUCTURE, payload
    if not all(item.get("exists") for item in payload["artifacts"]):
        return EXIT_INFRASTRUCTURE, payload
    return EXIT_OK, payload


def report(payload: dict[str, Any], code: int) -> None:
    """Prose. Every computed value appears; nothing is behind a flag."""

    print(f"project   {payload.get('project')}")
    if payload.get("script"):
        print(f"script    {payload['script']}")
    if payload.get("error"):
        print(f"ERROR     {payload['error']}")
        for key in ("failure_code", "failure_stage", "observed", "retry",
                    "candidates", "correction", "reason"):
            if key in payload:
                value = payload[key]
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, sort_keys=True)
                print(f"  {key:<14} {value}")
        return

    print(f"digest    {payload.get('digest')}")
    print(f"revision  {payload.get('accepted_revision')}")
    print(f"attempt   {payload.get('attempt_dir')}")
    verified = payload.get("verify")
    if verified is not None:
        mark = "STABLE" if verified["stable"] else "UNSTABLE"
        print(f"verify    {mark}  {verified['first']} / {verified['second']}")
    print()
    print(f"{'kind':<30} {'output':<14} {'bytes':>9}  sha256    path")
    for item in payload.get("artifacts") or []:
        digest = str(item.get("sha256") or "")[:8] or "--------"
        size = item.get("bytes")
        print(
            f"{item['kind']:<30} {item['output']:<14} "
            f"{(size if size is not None else '?'):>9}  {digest}  {item['path']}"
            + ("" if item.get("exists") else "   MISSING")
        )
    print()
    print("ok" if code == 0 else f"FAILED (exit {code})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harness rebuild",
        description="Rebuild a Cadex project and resolve its artifacts.",
    )
    parser.add_argument("--project", required=True, help="Project root (created if absent).")
    parser.add_argument("--script", help="A script file to write before rebuilding.")
    parser.add_argument(
        "--set", action="append", default=[], metavar="K=V",
        help="Set a declared parameter. Repeatable.",
    )
    parser.add_argument(
        "--replace", action="store_true",
        help="Permit a script that drops an output the accepted revision "
             "declares (ADR-045 refuses this without it).",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Rebuild twice and assert the digest is identical.",
    )
    parser.add_argument("--engine", help="A staged engine payload root, overriding the environment.")
    parser.add_argument(
        "--worker-cpu-seconds",
        type=float,
        default=DEFAULT_WORKER_CPU_SECONDS,
        help=(
            "RLIMIT_CPU for the engine's isolated domain worker, in CPU "
            "seconds (user+sys). The engine's own default is 300, which "
            "mg-legs does not fit in. Overrun surfaces as "
            "DOMAIN_WORKER_NO_RESULT with returncode -24 (SIGXCPU). "
            f"Default {DEFAULT_WORKER_CPU_SECONDS:g}."
        ),
    )
    parser.add_argument(
        "--worker-memory-mb",
        type=int,
        default=DEFAULT_WORKER_MEMORY_MB,
        help=(
            "RLIMIT_AS for that worker, in MB. The engine's own default is "
            "6144. Sent together with --worker-cpu-seconds because the "
            "engine accepts the pair only when both are positive. "
            f"Default {DEFAULT_WORKER_MEMORY_MB}."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit the envelope.")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_env_file()
    args = build_parser().parse_args(argv)
    try:
        code, payload = run(args)
    except ValueError as exc:
        print(f"usage: {exc}", file=sys.stderr)
        return EXIT_USAGE

    if args.json:
        json.dump(
            envelope("rebuild", code == EXIT_OK, payload),
            sys.stdout, indent=2, sort_keys=True,
        )
        sys.stdout.write("\n")
    else:
        report(payload, code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
