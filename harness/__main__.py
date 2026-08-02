"""``uv run python -m harness <driver> …`` — one entry point for the four.

A single dispatcher rather than four scripts, for one reason: the drivers
share an envelope, an exit-code convention and a provenance block, and a
reader who has learned one has learned all of them. Four loose scripts drift.

    rebuild     get a project to a known state, assert digest stability
    supervise   watch a training run, or read a finished one
    compare     play every checkpoint in a run directory
    capability  sweep the declared disturbance band

``measure`` and ``feasibility`` are specified in ``DESIGN.md`` and not built;
asking for one says so rather than failing with an argparse error, because
"that driver does not exist yet" and "you typed it wrong" are different
problems.
"""

from __future__ import annotations

import sys

from harness import EXIT_USAGE

DRIVERS = ("rebuild", "supervise", "compare", "capability")
DEFERRED = {
    "measure": "specified in harness/DESIGN.md §2; deferred — it earns its "
               "keep before a *new* dispatch, and nothing has dispatched yet.",
    "feasibility": "specified in harness/DESIGN.md §3; deferred for the same "
                   "reason as measure.",
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        print("drivers:", " ".join(DRIVERS))
        return 0 if argv else EXIT_USAGE

    name, rest = argv[0], argv[1:]
    if name in DEFERRED:
        print(f"{name}: not built. {DEFERRED[name]}", file=sys.stderr)
        return EXIT_USAGE
    if name not in DRIVERS:
        print(f"unknown driver {name!r}; expected one of {', '.join(DRIVERS)}",
              file=sys.stderr)
        return EXIT_USAGE

    module = __import__(f"harness.{name}", fromlist=["main"])
    return int(module.main(rest))


if __name__ == "__main__":
    raise SystemExit(main())
