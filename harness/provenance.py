"""The block every envelope carries, and the small things that build it.

A number without provenance is a number about nothing in particular. Two
runs against two engine commits have to be tellable apart from the record
alone, months later, by somebody who was not here — and the cheapest moment
to make that possible is the moment the number is produced.

What a block carries, and why each field is in it:

``timestamp_utc``
    When. UTC, ISO-8601, seconds resolution. Local time in a record that
    outlives a timezone is a small trap for no benefit.
``cadex``
    The Cadex checkout's HEAD and whether it was dirty. ADR-104's guard:
    the trainer and the engine both come out of that tree, so a result is a
    result *of that commit*. A dirty tree is recorded, never failed on — it
    is somebody's work in progress and cdx-rl has no opinion about it. What
    it must not be is unrecorded.
``cdxrl``
    This repository's HEAD and dirty flag, for the same reason applied to
    the instrument rather than the subject. `compare` measuring the same
    checkpoint differently before and after a fix is a thing that happens,
    and the row has to say which `compare` it was.
``engine``
    ``Engine.describe()`` where a driver drove one. Which binary, which
    module directory, which protocol.
``host``
    Hostname and platform. sb1x is not the only box this could run on.
``python``
    The interpreter that produced the envelope. Not the trainer's — that is
    reported separately by whatever spawned it, because it is a different
    interpreter and conflating the two is how a pin goes unnoticed.
"""

from __future__ import annotations

import hashlib
import os
import platform
import socket
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The schema every driver's ``--json`` envelope declares. One version for
#: the whole set: they are released together and read together.
ENVELOPE_SCHEMA = "cdxrl-envelope-v1"


def utc_now() -> str:
    """Now, as the only string a record should hold."""

    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | os.PathLike[str]) -> str:
    """One file's digest, read in blocks so a 500 KB policy and a 60 MB
    trace cost the same memory."""

    digest = hashlib.sha256()
    with Path(str(path)).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state(repo: str | os.PathLike[str]) -> dict[str, Any]:
    """``{commit, subject, dirty}`` for a checkout, or ``{error}``.

    Read-only: ``log`` and ``status``, nothing that writes. This is the one
    place cdx-rl touches the Cadex checkout's git at all, and it touches it
    the way a bystander does.
    """

    root = Path(str(repo)).expanduser()
    if not (root / ".git").exists():
        return {"error": f"{root} is not a git checkout"}
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "log", "-1", "--format=%H%x00%s"],
            capture_output=True, text=True, timeout=60, check=False,
        )
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=60, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        return {"error": str(exc)}
    if head.returncode != 0:
        return {"error": (head.stderr or "git log failed").strip()}
    commit, _, subject = head.stdout.strip().partition("\0")
    return {
        "commit": commit,
        "short": commit[:8],
        "subject": subject,
        "dirty": bool(status.stdout.strip()),
    }


def provenance(engine: Any = None, **extra: Any) -> dict[str, Any]:
    """The block. ``engine`` is anything with a ``describe()``, or ``None``.

    Extra keyword arguments are folded in verbatim — a driver that knows
    something else about where its numbers came from (a bundle digest, a run
    directory) puts it here rather than inventing a second block.
    """

    block: dict[str, Any] = {
        "timestamp_utc": utc_now(),
        "cadex": git_state(os.environ.get("CADEX_REPO", "/home/theo/cadex")),
        "cdxrl": git_state(REPO_ROOT),
        "host": {"hostname": socket.gethostname(), "platform": platform.platform()},
        "python": ".".join(str(part) for part in sys.version_info[:3]),
    }
    if engine is not None and hasattr(engine, "describe"):
        block["engine"] = engine.describe()
    block.update(extra)
    return block


def envelope(
    driver: str,
    ok: bool,
    payload: Mapping[str, Any],
    *,
    engine: Any = None,
    **extra: Any,
) -> dict[str, Any]:
    """One driver's ``--json`` output: schema, driver, ok, provenance, payload.

    The shape is fixed across drivers on purpose. A Flywheel artifact is made
    out of one of these, and a reader who has learned to read one has learned
    to read all of them.
    """

    return {
        "schema": ENVELOPE_SCHEMA,
        "driver": driver,
        "ok": bool(ok),
        "provenance": provenance(engine, **extra),
        **dict(payload),
    }


def load_env_file(path: Path | None = None) -> None:
    """Fold ``config/env`` into the environment without overriding it.

    Same rule as ``tools/smoke.py``: a value already exported wins, so a
    one-off ``CADEX_ENGINE_ROOT=… uv run …`` does what it looks like it does.
    Every driver calls this before reading a single environment variable, so
    that ``uv run python -m harness …`` works without a preceding ``set -a``.
    """

    target = path or (REPO_ROOT / "config" / "env")
    if not target.is_file():
        return
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())
