"""A standalone cadexd client, and the artifact resolver the CLI does not have.

This is cdx-rl's spine. Every driver in ``harness/`` sits on it.

**Why it exists rather than importing Cadex's own.** ``cli/cadex_cli/client.py``
in the Cadex checkout does the same job, but it is LGPL-2.1-or-later and it
is *inside a repository cdx-rl is read-only toward*. Importing it would make
this repo's tooling depend on a path we have promised not to touch and on a
licence we have not chosen. The protocol is documented — ``cadex-cadexd-v1``,
``docs/INTEGRATION.md`` §"cadexd protocol" — so this is written from the
specification. Where the two disagree, the engine is right.

**What the protocol is.** Newline-delimited JSON over stdio, 8 MB frame cap,
one ``cadexd`` child (a ``FreeCADCmd`` process) per open project. Requests are
``{schema, id, op, args}``; responses are ``{id, ok, ...payload}``; progress
arrives as ``{id, event}`` frames interleaved with them. A ``ready`` banner
event is emitted on startup. FreeCAD's own chatter goes to stderr once
cadexd hijacks the fds, but *before* that it can land on stdout — so a
non-JSON line before the banner is expected and skipped, not an error.

**The part that is not in the protocol.** The ``--json`` envelope names
outputs and their kinds and does not say where the files are, and there is no
CLI op that answers "where did the accepted rebuild put its MJCF". The engine
knows: ``script.json`` carries ``accepted_attempt.staging``, a project-root
relative path pinned against garbage collection, and that directory holds
``result.json`` (the worker report, with a per-output ``artifact_path``) and
an ``outputs/`` subdirectory with the files themselves.
:func:`accepted_attempt_dir` and :func:`accepted_artifacts` are that lookup.
It is the smallest thing cdx-rl had to add to be able to work at all, and it
is item one on ``cadex-wishlist.md`` — still **open**, and now a candidate
PR rather than a permanent workaround. It is not the first one, because it
costs a resolver cdx-rl already has rather than GPU hours; #12 and #11 are
ahead of it.

Usage::

    from cadexd_client import Engine, CadexdClient, accepted_artifacts

    with CadexdClient(Engine.resolve()) as cadexd:
        cadexd.open_project(project_root)
        reply = cadexd.write_script(source)
        artifacts = accepted_artifacts(project_root)
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROTOCOL_SCHEMA = "cadex-cadexd-v1"
ENGINE_MANIFEST_NAME = "cadex-engine.json"
ENGINE_MANIFEST_SCHEMA = "cadex-engine-v1"

#: 8 MB, the protocol's own frame cap. A line longer than this is a bug on
#: one side or the other; refusing it is better than buffering it.
MAX_FRAME_BYTES = 8 * 1024 * 1024

#: FreeCADCmd loads OCCT and every kept workbench before cadexd announces
#: itself. On a cold page cache that is tens of seconds.
READY_TIMEOUT_SECONDS = 180.0

#: A modelling op can honestly take minutes. This cap is here to stop a
#: wedged engine hanging a sweep forever, not to bound real work.
DEFAULT_TIMEOUT_SECONDS = 900.0

#: The **worker** budget, which is a different thing from the cap above and
#: bit us on 2026-08-05.
#:
#: ``DEFAULT_TIMEOUT_SECONDS`` bounds how long *this client* waits for a
#: reply. These two bound what the engine's isolated domain worker is allowed
#: to spend, and they are applied as real ``setrlimit`` calls inside it
#: (``cadex_domain_worker.py:_resource_limits``) — ``RLIMIT_CPU`` and
#: ``RLIMIT_AS``.
#:
#: The engine's own defaults are 300 s and 6144 MB
#: (``CadexEngineSettings.DEFAULT_SCRIPTED_TIMEOUT_SECONDS``). 300 s is
#: generous for the toy assemblies in ``smoke.py`` and **not enough for
#: ``mechanisms/mg-legs/script.py``**, which is a real machine: the first
#: attempt to build it on sb1x burned 287 s and died on ``SIGXCPU``. What
#: comes back is ``DOMAIN_WORKER_NO_RESULT`` with ``returncode: -24`` buried
#: in a multi-kilobyte ``stdout`` of OCCT progress bars, which does not read
#: like "you ran out of CPU seconds" at all. Signal 24 is ``SIGXCPU``; if a
#: rebuild ever dies at a suspiciously round number of seconds, look here
#: first.
#:
#: ``open_project`` takes these over the protocol
#: (``cadexd.py`` → ``CadexEngineSettings.resolve_budgets``), so raising them
#: is a cdx-rl-side call and needs no engine change. **Both must be positive
#: or the engine silently ignores the pair and falls back to its
#: preferences** — that is ``resolve_budgets``'s contract, not a bug.
#:
#: **Both numbers are raised, and the memory one is load-bearing.**
#:
#: The first version of this constant kept memory at the engine's own 6144 MB,
#: reasoning that the run peaked at 228 MB so memory was never the constraint.
#: That was right about memory and wrong about the limit: ``RLIMIT_AS`` caps
#: *address space*, not resident memory, and MuJoCo's allocator and its plugin
#: ``dlopen``s reserve far more virtual address space than they ever touch.
#: Measured on sb1x, same script and a fresh project, changing only this:
#:
#:     6144 MB  -> SIGXCPU, never finishes (1787 s of CPU, ~80 % of it *system*
#:                 time, RSS 218 MB, the engine's own memory_exceeded false)
#:    32768 MB  -> succeeds in 8.2 s
#:
#: Roughly **500x**, and not a tuning preference: at 6144 the mg-legs build
#: does not complete at all. Cadex's own suite shows it too -- two
#: ``test_dynamics_collision`` tests fail on ``main`` with *"libelasticity.so:
#: failed to map segment from shared object"* and pass when their
#: ``open_project`` is given a larger budget. ``cadex-wishlist.md`` #14 is the
#: upstream fix; this is cdx-rl not paying for it in the meantime.
#:
#: 1800 s is then ~200x the measured build rather than ~6x, which is fine: it
#: is a runaway guard, not a schedule.
DEFAULT_WORKER_CPU_SECONDS = 1800.0
DEFAULT_WORKER_MEMORY_MB = 32768

#: The assembly-domain exports cdx-rl cannot work without. An engine missing
#: any of them is not a Cadex that can do reinforcement learning, and finding
#: that out at startup is worth one round trip.
REQUIRED_ASSEMBLY_EXPORTS = (
    "body",
    "collision",
    "joint_dynamics",
    "actuator",
    "observation",
    "reward",
    "termination",
    "reset_variation",
    "disturbance",
    "dynamics",
    "mjcf",
    "task",
    "policy",
    "rollout",
)

EventCallback = Callable[[dict[str, Any]], None]


class CadexdError(RuntimeError):
    """The engine could not be spoken to: spawn, EOF, timeout, or bad shape."""


class EngineError(RuntimeError):
    """No usable engine payload, or a manifest that does not check out."""


class ScriptRefused(RuntimeError):
    """The engine ran the script and refused the result.

    The counterpart of the CLI's exit code 3, and separate from
    :class:`CadexdError` for the same reason: a refused script is a modelling
    problem to feed back into the next attempt, an unreachable engine is an
    infrastructure problem to retry or abort on. A driver that collapses the
    two will retry a script that will never build.
    """

    def __init__(self, reply: Mapping[str, Any]) -> None:
        self.reply = dict(reply)
        self.failure_code = str(reply.get("failure_code") or "")
        self.failure_stage = str(reply.get("failure_stage") or "")
        message = str(reply.get("error") or self.failure_code or "refused")
        super().__init__(
            f"{message}"
            + (f"  [{self.failure_code}" if self.failure_code else "")
            + (f" @ {self.failure_stage}]" if self.failure_stage else
               ("]" if self.failure_code else ""))
        )


# --------------------------------------------------------------------------
# Engine resolution
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Engine:
    """A resolved engine payload: what to run, and what to put on its path."""

    root: Path
    freecadcmd: Path
    module_dir: Path
    version: str
    protocol: str

    @classmethod
    def resolve(cls, explicit: str | os.PathLike[str] | None = None) -> Engine:
        """``explicit``, else ``CADEX_ENGINE_ROOT``, else ``CADEX_ENGINE_DEV_TREE``.

        Both environment routes are explicit choices. There is no *silent*
        fallback from one to the other, because the two are not the same
        engine and finding that out from a missing API function three steps
        into a build is expensive — see :meth:`dev_tree` for the case where
        that is not hypothetical.
        """

        source = explicit or os.environ.get("CADEX_ENGINE_ROOT", "").strip()
        if not source:
            checkout = os.environ.get("CADEX_ENGINE_DEV_TREE", "").strip()
            if checkout:
                return cls.dev_tree(checkout)
            raise EngineError(
                "No engine. Set CADEX_ENGINE_DEV_TREE to a built Cadex checkout, "
                "or CADEX_ENGINE_ROOT to a staged payload directory (the one "
                f"holding {ENGINE_MANIFEST_NAME}), or pass --engine. "
                "See config/env.example."
            )
        root = Path(str(source)).expanduser()
        if not root.is_dir():
            raise EngineError(f"{root} is not a directory.")
        root = root.resolve()

        manifest_path = root / ENGINE_MANIFEST_NAME
        if not manifest_path.is_file():
            raise EngineError(
                f"{root} has no {ENGINE_MANIFEST_NAME}. The manifest is the "
                "payload's discovery contract (ADR-020); a directory without "
                "one is not a payload."
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise EngineError(f"{manifest_path} is not readable JSON: {exc}") from exc
        if manifest.get("schema") != ENGINE_MANIFEST_SCHEMA:
            raise EngineError(
                f"{manifest_path} declares schema {manifest.get('schema')!r}; "
                f"expected {ENGINE_MANIFEST_SCHEMA!r}."
            )

        try:
            binary = root.joinpath(*str(manifest["freecadcmd"]).split("/"))
            module_dir = root.joinpath(*str(manifest["module_dir"]).split("/"))
        except KeyError as exc:
            raise EngineError(f"{manifest_path} is missing {exc}.") from exc
        if not binary.is_file():
            raise EngineError(f"{manifest_path} names a missing binary: {binary}")
        if not module_dir.is_dir():
            raise EngineError(f"{manifest_path} names a missing module dir: {module_dir}")

        declared = str(manifest.get("protocol") or "")
        if declared and declared != PROTOCOL_SCHEMA:
            raise EngineError(
                f"{manifest_path} speaks {declared!r}; this client speaks "
                f"{PROTOCOL_SCHEMA!r}."
            )
        return cls(
            root=root,
            freecadcmd=binary,
            module_dir=module_dir,
            version=str(manifest.get("version") or ""),
            protocol=declared or PROTOCOL_SCHEMA,
        )

    #: Where a dev-tree engine looks for a binary, in the CLI's own order:
    #: the installed pixi prefix first, then the build tree. The installed
    #: binary loads everything from one prefix, so type registration is sound.
    DEV_BINARY_CANDIDATES = (
        Path(".pixi/envs/default/bin/FreeCADCmd"),
        Path("build/release/bin/FreeCADCmd"),
    )

    @classmethod
    def dev_tree(cls, checkout: str | os.PathLike[str]) -> Engine:
        """A built Cadex checkout driven directly: its binary, its ``src/``.

        **This is the engine cdx-rl uses on sb1x, and the reason is not a
        preference.** The staged payload at
        ``build/engine/cadex-engine-0.0.0-linux-x64`` was assembled on
        2026-07-31 and its ``Mod/cadex`` predates the whole MuJoCo surface:
        no ``CadexDynamics.py``, and an ``assembly`` domain whose exports stop
        at ``exploded_view``. Asked for its API it answers::

            ['assembly', 'component', 'connector', 'joint', 'solve',
             'motion', 'simulation', 'exploded_view']

        while the checkout's ``src/Mod/cadex`` answers with those plus
        ``dynamics``, ``mjcf``, ``task``, ``policy``, ``rollout``, ``body``,
        ``collision``, ``joint_dynamics``, ``actuator``, ``observation``,
        ``reward``, ``termination``, ``randomise``, ``reset_variation`` and
        ``disturbance``. Every one of those is load-bearing for this
        repository. A driver pointed at the payload does not get a clear
        error; it gets ``assembly.mjcf`` is not defined, from inside a script,
        which reads like a script bug.

        :func:`cdxrl_requires_dynamics` is the assertion that turns that into
        one loud failure at startup, and every driver should call it.

        Re-staging the payload would fix this. Since 2026-08-05 cdx-rl has
        its own clone at ``/home/theo/cadex-prs`` and *could* run
        ``pixi run stage-engine`` there — the old objection, that it writes
        into a checkout this repository is read-only toward, is gone. It
        still has not, deliberately: the dev-tree route below is the one
        every driver uses, and a second engine route to keep current is a
        liability rather than a convenience. ``cadex-wishlist.md`` #2.
        """

        root = Path(str(checkout)).expanduser().resolve()
        module_dir = root / "src" / "Mod" / "cadex"
        if not module_dir.is_dir():
            raise EngineError(f"{root} is not a Cadex checkout: no {module_dir}.")
        binary = next(
            (root / candidate for candidate in cls.DEV_BINARY_CANDIDATES
             if (root / candidate).is_file()),
            None,
        )
        if binary is None:
            raise EngineError(
                f"{root} has no built FreeCADCmd. Looked for: "
                + ", ".join(str(root / c) for c in cls.DEV_BINARY_CANDIDATES)
            )
        return cls(
            root=root,
            freecadcmd=binary,
            module_dir=module_dir,
            version="dev-tree",
            protocol=PROTOCOL_SCHEMA,
        )

    def describe(self) -> dict[str, str]:
        """The engine identity a run record carries.

        Two runs against two engines have to be tellable apart in a log, and
        a graph node claiming a measurement without one is a claim about
        nothing in particular.
        """

        return {
            "root": str(self.root),
            "freecadcmd": str(self.freecadcmd),
            "module_dir": str(self.module_dir),
            "version": self.version,
            "protocol": self.protocol,
        }


# --------------------------------------------------------------------------
# The client
# --------------------------------------------------------------------------


class CadexdClient:
    """Spawn one ``cadexd`` and talk to it. One request at a time."""

    def __init__(
        self,
        engine: Engine,
        *,
        on_event: EventCallback | None = None,
        stderr: int | None = subprocess.DEVNULL,
    ) -> None:
        self.engine = engine
        self.on_event = on_event
        self._stderr = stderr
        self._process: subprocess.Popen[bytes] | None = None
        self._sequence = 0
        self._pending: dict[str, dict[str, Any]] = {}
        self._write_lock = threading.Lock()
        self.events: list[dict[str, Any]] = []
        self.project_root: Path | None = None
        #: The revision the next write must declare it expects. Tracked here
        #: rather than asked of the caller because there is only ever one
        #: writer: a driver has one project and runs one op at a time, so a
        #: stale value could only come from this client forgetting to update
        #: it. ``STALE_PROGRAM_REVISION`` is the guard if it ever does.
        self.revision: str = ""

    # -- lifecycle ---------------------------------------------------------

    def start(self, timeout: float = READY_TIMEOUT_SECONDS) -> None:
        """Spawn the engine and block until it announces ``ready``."""

        if self._process is not None:
            raise CadexdError("This client already has an engine running.")
        command = [
            str(self.engine.freecadcmd),
            "-c",
            (
                f"import sys; sys.path.insert(0, {str(self.engine.module_dir)!r}); "
                "import cadexd; raise SystemExit(cadexd.main())"
            ),
        ]
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr,
                # PYTHONHASHSEED pins set/dict iteration inside the worker.
                # The digest is supposed to be a function of the script, and
                # this is one of the few places a machine could disagree with
                # itself between two runs of the same thing.
                env={**os.environ, "PYTHONHASHSEED": "0"},
            )
        except OSError as exc:
            raise CadexdError(
                f"Could not start the engine ({self.engine.freecadcmd}): {exc}"
            ) from exc

        frame = self._read_frame(timeout)
        event = frame.get("event")
        if not isinstance(event, dict) or event.get("event") != "ready":
            raise CadexdError(
                f"The engine's first frame was not a ready banner: {frame!r}"
            )
        self.ready_banner = event

    def close(self) -> None:
        """Stop the engine, by force. Safe to call twice."""

        process, self._process = self._process, None
        if process is None:
            return
        if process.poll() is None:
            process.kill()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        for stream in (process.stdin, process.stdout):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass

    def shutdown(self, timeout: float = 60.0) -> None:
        """Ask the engine to exit, then make sure it did."""

        if self._process is None or self._process.poll() is not None:
            self.close()
            return
        try:
            self.request("shutdown", timeout=timeout)
            self._process.wait(timeout=timeout)
        except (CadexdError, subprocess.TimeoutExpired, OSError):
            pass
        finally:
            self.close()

    def __enter__(self) -> CadexdClient:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.shutdown()

    # -- the wire ----------------------------------------------------------

    def _read_frame(self, timeout: float) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdout is None:
            raise CadexdError("No engine is running.")
        deadline = time.monotonic() + timeout
        while time.monotonic() <= deadline:
            line = process.stdout.readline()
            if not line:
                raise CadexdError(
                    "The engine closed its protocol stream "
                    f"(exit status {process.poll()!r}). Re-run with "
                    "--engine-stderr to see what it said on the way out."
                )
            if len(line) > MAX_FRAME_BYTES:
                raise CadexdError(
                    f"A frame of {len(line)} bytes exceeds the protocol's "
                    f"{MAX_FRAME_BYTES} byte cap."
                )
            line = line.strip()
            if not line:
                continue
            try:
                frame = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                # FreeCADCmd chatter, printed before cadexd took the fds.
                continue
            if isinstance(frame, dict):
                return frame
        raise CadexdError(f"No frame from the engine within {timeout:g}s.")

    def _send(self, op: str, args: dict[str, Any] | None, request_id: str) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise CadexdError("No engine is running.")
        frame: dict[str, Any] = {"schema": PROTOCOL_SCHEMA, "id": request_id, "op": op}
        if args is not None:
            frame["args"] = args
        payload = (json.dumps(frame, separators=(",", ":")) + "\n").encode("utf-8")
        if len(payload) > MAX_FRAME_BYTES:
            raise CadexdError(
                f"A {op} request of {len(payload)} bytes exceeds the protocol's "
                f"{MAX_FRAME_BYTES} byte cap. Binary belongs on disk: put_asset "
                "takes a path, not bytes."
            )
        try:
            with self._write_lock:
                process.stdin.write(payload)
                process.stdin.flush()
        except OSError as exc:
            raise CadexdError(f"Could not write to the engine: {exc}") from exc

    def request(
        self,
        op: str,
        args: dict[str, Any] | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        """Send one request, absorb its progress events, return the reply.

        The reply is returned whatever its ``ok`` — a refusal is a result, and
        the envelope carries the diagnostics that say why. :meth:`checked` is
        the wrapper that turns one into an exception.
        """

        self._sequence += 1
        request_id = f"cdxrl-{self._sequence}"
        self._send(op, args, request_id)

        stashed = self._pending.pop(request_id, None)
        if stashed is not None:
            return stashed
        deadline = time.monotonic() + timeout
        while True:
            frame = self._read_frame(max(0.1, deadline - time.monotonic()))
            if "event" in frame:
                self.events.append(frame)
                if self.on_event is not None:
                    self.on_event(frame)
                continue
            if frame.get("id") == request_id:
                return frame
            # A reply to something else — a cancel ack overtaking the request
            # it cancelled. Keep it; somebody asked for it.
            self._pending[str(frame.get("id"))] = frame

    def checked(
        self,
        op: str,
        args: dict[str, Any] | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        """:meth:`request`, raising on a refusal.

        Server-level failures (``CADEXD_*``) are infrastructure and raise
        :class:`CadexdError`; everything else the engine refuses raises
        :class:`ScriptRefused`. That split is the exit-1/exit-3 distinction,
        available to a driver rather than only to a shell.
        """

        reply = self.request(op, args, timeout=timeout)
        if reply.get("ok") is True:
            return reply
        code = str(reply.get("failure_code") or "")
        if code.startswith("CADEXD_"):
            raise CadexdError(
                f"{op} failed at the server: {reply.get('error') or code}"
            )
        raise ScriptRefused(reply)

    def cancel(self, request_id: str | None = None) -> None:
        """Ask the engine to abandon its in-flight modelling run."""

        self._sequence += 1
        args = {"request_id": request_id} if request_id else None
        try:
            self._send("cancel", args, f"cdxrl-cancel-{self._sequence}")
        except CadexdError:
            pass

    # -- the ops we actually use ------------------------------------------

    def open_project(
        self,
        project_root: str | Path,
        *,
        restore: bool = True,
        cpu_seconds: float = DEFAULT_WORKER_CPU_SECONDS,
        memory_mb: int = DEFAULT_WORKER_MEMORY_MB,
    ) -> dict[str, Any]:
        """Open (or create) a project root.

        cadexd does the ``mkdir(parents=True, exist_ok=True)`` itself, so this
        may name a directory that does not exist yet.

        ``cpu_seconds`` and ``memory_mb`` are the *worker* budgets, resolved
        once here and applied as ``setrlimit`` inside every isolated domain
        worker for the life of the session. They default above the engine's
        own 300 s because ``mg-legs`` does not fit in 300 s — see
        :data:`DEFAULT_WORKER_CPU_SECONDS` for the measurement and for why
        overrunning is so hard to read in the reply.

        Both are sent together on purpose: ``resolve_budgets`` accepts the
        caller's pair only when *both* are positive, and otherwise discards
        both and uses the engine preferences. Sending one is the same as
        sending neither, silently.
        """

        root = Path(str(project_root)).expanduser().resolve()
        args: dict[str, Any] = {"project_root": str(root), "restore": bool(restore)}
        if cpu_seconds > 0 and memory_mb > 0:
            args["budgets"] = {
                "timeout_seconds": float(cpu_seconds),
                "memory_limit_mb": int(memory_mb),
            }
        reply = self.checked("open_project", args)
        self.project_root = root
        script = reply.get("script")
        if isinstance(script, Mapping):
            self.revision = str(
                script.get("working_revision") or script.get("revision") or ""
            )
        if not self.revision:
            # ``open_project``'s reply does not always carry the working
            # revision — measured on this box, it comes back empty for a
            # project that has an accepted script. An empty expectation is
            # accepted by the engine only until something else in the session
            # observes the script, after which the next write is refused with
            # STALE_PROGRAM_REVISION. Ask for it explicitly instead.
            self.refresh_revision()
        return reply

    def refresh_revision(self) -> str:
        """Re-read the working revision from the engine. Returns it.

        **The guard exists because a mutation that is not a write still moves
        the revision.** ``put_asset`` copies a file into the project store,
        and the very next ``write_script`` is refused with
        ``STALE_PROGRAM_REVISION`` — *"The project script changed after
        inspection"* — even though the script has not changed at all. That is
        exactly the shape of experiment 000's step 6: put the trained policy
        in the store, then declare it in the script. Call this after any op
        that touches the project and is not itself a script write.
        """

        value = self.inspect("script").get("value")
        if isinstance(value, Mapping):
            revisions = value.get("revisions")
            if isinstance(revisions, Mapping):
                self.revision = str(
                    revisions.get("working_revision")
                    or revisions.get("accepted_revision")
                    or self.revision
                )
        return self.revision

    def _remember_revision(self, reply: Mapping[str, Any]) -> dict[str, Any]:
        """Carry the new revision forward for the next write's guard."""

        self.revision = str(reply.get("revision") or self.revision)
        return dict(reply)

    def describe_api(self) -> dict[str, Any]:
        """The authoring surface this engine offers, verbatim.

        Worth calling rather than trusting a document: the domain list and the
        per-domain function signatures are the ground truth for what a script
        may say, and they move.
        """

        return self.checked("describe_api")

    def write_script(self, source: str, *, replace: bool = False, **kwargs: Any) -> dict[str, Any]:
        """Replace the whole project script and rebuild.

        ``replace=True`` is you saying you mean to drop an output the accepted
        revision declares. Without it such a script is refused (ADR-045).
        """

        args: dict[str, Any] = {
            "source": source,
            "expected_revision": kwargs.pop("expected_revision", self.revision),
            **kwargs,
        }
        if replace:
            args["replace"] = True
        return self._remember_revision(self.checked("write_script", args))

    def set_params(self, values: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Set declared parameters and rebuild. No model in the loop."""

        args: dict[str, Any] = {
            "values": dict(values),
            "expected_revision": kwargs.pop("expected_revision", self.revision),
            **kwargs,
        }
        return self._remember_revision(self.checked("set_params", args))

    def rebuild(self, **kwargs: Any) -> dict[str, Any]:
        """Deterministically re-run the stored script.

        The digest of two rebuilds of one script is the reproducibility check
        the whole repository rests on. Compare digests, never files: STEP
        writes a wall-clock timestamp into ``FILE_NAME`` and two exports of an
        identical model differ byte for byte across a second boundary.
        """

        return self._remember_revision(self.checked("rebuild", dict(kwargs) or None))

    def inspect(self, scope: str, **kwargs: Any) -> dict[str, Any]:
        """A bounded read of project or document state.

        Bounded is the point of the op, not a limitation of it: replies cap at
        32 KiB, containers page, and any single value over 1 KiB is replaced
        by a marker naming the JSON Pointer that reaches it. ``value`` is a
        view, never a promise of the whole.
        """

        return self.checked("inspect", {"scope": scope, **kwargs})

    def assembly_exports(self) -> list[str]:
        """The assembly domain's function names, from the engine itself."""

        domains = self.describe_api().get("domains") or {}
        assembly = domains.get("assembly") if isinstance(domains, Mapping) else None
        if not isinstance(assembly, Mapping):
            return []
        return [
            str(item.get("name") or "")
            for item in (assembly.get("exports") or [])
            if isinstance(item, Mapping)
        ]

    def require_dynamics(self) -> list[str]:
        """Fail now if this engine has no MuJoCo surface. Returns the exports.

        Call it once, immediately after :meth:`open_project`. The failure it
        prevents is the expensive kind: an engine without the dynamics domain
        refuses the *script* rather than the connection, so every diagnostic
        points at the model and none of them point at the engine.
        """

        exports = self.assembly_exports()
        missing = [name for name in REQUIRED_ASSEMBLY_EXPORTS if name not in exports]
        if missing:
            raise EngineError(
                f"This engine's assembly domain is missing {missing}. It offers "
                f"{exports}.\n"
                f"  engine: {self.engine.root} ({self.engine.version})\n"
                "The staged payload on sb1x predates the MuJoCo work. Point "
                "CADEX_ENGINE_DEV_TREE at the Cadex checkout and leave "
                "CADEX_ENGINE_ROOT unset (see config/env.example)."
            )
        return exports

    def put_asset(self, source_path: str | Path, *, name: str | None = None) -> dict[str, Any]:
        """Copy one file into the project's ``assets/``.

        Takes a path, not bytes — the asset budget is 128 MB against an 8 MB
        frame cap. Accepts meshes (``.stl``/``.obj``/``.ply``) and, since
        ADR-084, ``.cxpolicy``: this is how a trained policy comes home.
        """

        args: dict[str, Any] = {"source_path": str(Path(str(source_path)).resolve())}
        if name:
            args["name"] = name
        reply = self.checked("put_asset", args)
        # Storing an asset counts as touching the project: without this the
        # next write_script is refused as stale. See :meth:`refresh_revision`.
        self.refresh_revision()
        return reply


# --------------------------------------------------------------------------
# Artifact resolution — the part the CLI does not offer
# --------------------------------------------------------------------------


def read_script_state(project_root: str | Path) -> dict[str, Any]:
    """A project's ``script.json``: revisions, digest, params, accepted attempt."""

    path = Path(str(project_root)).expanduser().resolve() / "script.json"
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FileNotFoundError(f"No project state at {path}: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"{path} is not readable JSON: {exc}") from exc
    if not isinstance(state, dict):
        raise ValueError(f"{path} is not a JSON object.")
    return state


def accepted_attempt_dir(project_root: str | Path) -> Path:
    """The staging directory of the accepted revision's attempt.

    ``script_artifacts/<revision>/attempt-<id>/``. Pinned by ``script.json``
    against the pruner that keeps only the last few attempts, so this is the
    one attempt directory that is safe to reference from a graph node.

    The containment check is the engine's own: ``staging`` is a relative path
    and a value that escaped the project root would be a store somebody else
    wrote.

    **Do not construct this path from ``accepted_revision``.** The two
    disagree, by design and silently. ``CadexScriptedRuntime`` names the
    staging directory with a *pre-run* revision computed over the stored spec
    cache, and then ``validate_project_result`` recomputes the revision with
    the specs the worker actually collected and records *that* as the durable
    one. Measured on this box: an accepted revision of ``104826f0…`` whose
    artifacts sat under ``script_artifacts/93a118d4…/``. A resolver that
    joins ``script_artifacts/<accepted_revision>/`` gets a missing directory
    for a script whose parameters the worker had anything to say about —
    which is every script that declares one.
    """

    root = Path(str(project_root)).expanduser().resolve()
    attempt = read_script_state(root).get("accepted_attempt")
    if not isinstance(attempt, Mapping) or not str(attempt.get("staging") or ""):
        raise ValueError(
            f"{root} has no accepted attempt. Rebuild it once: nothing is "
            "pinned until a revision is accepted."
        )
    staging = (root / str(attempt["staging"])).resolve()
    if root not in staging.parents:
        raise ValueError(f"{staging} is not under {root}.")
    if not staging.is_dir():
        raise ValueError(f"The accepted attempt directory is missing: {staging}")
    return staging


def accepted_outputs_dir(project_root: str | Path) -> Path:
    """``<accepted attempt>/outputs/`` — where the engine writes files.

    The MJCF lands here as ``<output>-model.xml``, the training bundle as
    ``<output>-task.json``, a policy receipt as ``<output>-policy.json`` and a
    simulation or rollout trace as ``assembly-simulation-trace.json``.
    """

    return accepted_attempt_dir(project_root) / "outputs"


def worker_report(project_root: str | Path) -> dict[str, Any]:
    """The accepted attempt's ``result.json``, refusing one that failed."""

    path = accepted_attempt_dir(project_root) / "result.json"
    if not path.is_file():
        raise ValueError(f"The accepted attempt has no worker report at {path}.")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("ok") is not True:
        raise ValueError(f"{path} is not a usable worker report.")
    return report


@dataclass(frozen=True)
class Artifact:
    """One file the accepted rebuild produced."""

    output: str
    kind: str
    path: Path
    type: str = ""
    domain: str = ""

    def exists(self) -> bool:
        return self.path.is_file()

    def sha256(self) -> str:
        import hashlib

        digest = hashlib.sha256()
        with self.path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()


def accepted_artifacts(project_root: str | Path) -> list[Artifact]:
    """Every artifact of the accepted revision, as absolute paths.

    This is the lookup the ``--json`` envelope does not perform. It names
    outputs and their kinds; it does not say where the files are, and for
    ``assembly_mjcf_xml`` and ``assembly_training_task_json`` — the two that
    matter to a training pipeline — the CLI has no export path at all
    (``cadex export`` writes STEP/STL/BREP, which are BREP-domain formats).

    ``artifact_kind`` is an open set. Select on the kinds you know and ignore
    one you have never heard of; do not fail on it.
    """

    staging = accepted_attempt_dir(project_root)
    report = worker_report(project_root)
    artifacts: list[Artifact] = []
    for item in report.get("outputs") or []:
        if not isinstance(item, Mapping):
            continue
        relative = str(item.get("artifact_path") or "")
        if not relative:
            continue
        artifacts.append(
            Artifact(
                output=str(item.get("name") or ""),
                kind=str(item.get("artifact_kind") or ""),
                path=(staging / relative).resolve(),
                type=str(item.get("type") or ""),
                domain=str(item.get("domain") or ""),
            )
        )
    return artifacts


def artifacts_of_kind(project_root: str | Path, kind: str) -> list[Artifact]:
    """:func:`accepted_artifacts`, filtered. The selecting-not-failing rule."""

    return [item for item in accepted_artifacts(project_root) if item.kind == kind]
