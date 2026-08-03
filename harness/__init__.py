"""cdx-rl's drivers — the instruments ``MUJOCO.md`` §7 assumes exist.

Four of the six specified in :doc:`DESIGN.md` are built:

``rebuild``
    get a project to a known state, and assert its digest is stable.
``supervise``
    watch a training run, or read a finished one, and report.
``compare``
    play every checkpoint in a run directory and choose by measured survival.
``capability``
    sweep the declared disturbance band and say what the task was asking.

``measure`` and ``feasibility`` are not built. Both earn their keep only
before a *new* dispatch, and nothing here dispatches anything but a CPU
pendulum.

Run them through the package's own entry point::

    uv run python -m harness rebuild --project projects/000-loop-validation …
    uv run python -m harness supervise --run /home/theo/cadex-jobs/… --report-only
    uv run python -m harness compare --dir … --task …
    uv run python -m harness capability --policy … --task …

**Exit codes mirror Cadex's**, so a shell can act on them without reading
prose: ``0`` fine, ``1`` infrastructure, ``2`` usage, ``3`` the engine refused
the script. Collapsing 1 and 3 makes a driver retry a script that will never
build, which is why :class:`~cadexd_client.ScriptRefused` and
:class:`~cadexd_client.CadexdError` are separate types upstream of here.
"""

from __future__ import annotations

#: Exit codes, named. Every driver returns one of these from ``main``.
EXIT_OK = 0
EXIT_INFRASTRUCTURE = 1
EXIT_USAGE = 2
EXIT_REFUSED = 3

#: The trainer died, and the run is still usable.
#:
#: Only ``tools/train.py`` returns this, and only when the process exited
#: non-zero *and* left complete checkpoints behind — sb9x's open ``SIGSEGV``
#: at ``train()``'s return (``cloud.md`` §1) does exactly that. It is its own
#: code because the two obvious alternatives are both wrong: reporting ``0``
#: would hide a real crash and let a sweep treat a truncated run as finished,
#: and reporting ``1`` puts four hours of witness-checked checkpoints in the
#: same bucket as a run that died importing jax. A shell can now branch on
#: "worth running ``compare`` against" without reading prose, which is the
#: whole point of this list.
EXIT_SALVAGEABLE = 4

__all__ = ["EXIT_OK", "EXIT_INFRASTRUCTURE", "EXIT_USAGE", "EXIT_REFUSED",
           "EXIT_SALVAGEABLE"]
