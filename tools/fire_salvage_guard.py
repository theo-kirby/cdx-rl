#!/usr/bin/env python
"""Make ``train.py``'s salvage classifier decide, on purpose, in both directions.

``harness/DESIGN.md`` §6: *"every check must be able to fail, and must have
been made to fail once on purpose."* This one decides an **exit code**, which
is what a sweep and any calling shell branch on, so both of its mistakes are
expensive and neither is loud:

* calling a lost run salvageable promotes a directory with nothing in it, and
  the next step — ``compare`` — fails far away from the cause; while
* calling a salvageable run lost throws away a witnessed, complete set of
  checkpoints because the process that wrote them died on the way out. On
  sb9x that is four GPU-hours a seed (``cloud.md`` §1).

The classifier reads a run directory, so the cases here **are** run
directories, built to match the two failures actually observed on this box.
No mocking: this calls ``train.post_mortem``.

```
uv run python tools/fire_salvage_guard.py
```

Exit 0 if every case classifies as expected.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from harness import EXIT_INFRASTRUCTURE, EXIT_OK, EXIT_SALVAGEABLE  # noqa: E402

_spec = importlib.util.spec_from_file_location("train", REPO_ROOT / "tools" / "train.py")
train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train)

LABEL = "synthetic"

#: (name, files to create, returncode, expect_salvageable, why)
CASES: list[tuple[str, list[str], int, bool, str]] = [
    (
        "segfault after checkpoints",
        [f"{LABEL}.000100.cxpolicy", f"{LABEL}.000200.cxpolicy",
         f"{LABEL}.best.cxpolicy"],
        -11,
        True,
        "sb9x's observed fault: train() returned, everything but the final "
        "policy is on disk",
    ),
    (
        "segfault during tracing",
        [],
        -11,
        False,
        "died before iteration 0 — the run directory holds no policy at all",
    ),
    (
        "clean exit",
        [f"{LABEL}.000100.cxpolicy", f"{LABEL}.cxpolicy"],
        0,
        True,
        "nothing to salvage because nothing was lost",
    ),
    (
        "killed with one checkpoint",
        [f"{LABEL}.000100.cxpolicy"],
        -15,
        True,
        "SIGTERM mid-run: one checkpoint is still a checkpoint, and this is "
        "the shape experiment 002's stopped seed 2 has",
    ),
]


def fire(name: str, files: list[str], returncode: int, expect: bool,
         why: str, root: Path) -> dict[str, Any]:
    run_dir = root / name.replace(" ", "-")
    run_dir.mkdir(parents=True, exist_ok=True)
    for filename in files:
        (run_dir / filename).write_bytes(b"not a real policy, only its name")
    (run_dir / "stand-task.json").write_text("{}", encoding="utf-8")

    out = run_dir / f"{LABEL}.cxpolicy"
    result = train.post_mortem(run_dir, out, returncode, 1500)

    code = (EXIT_OK if returncode == 0
            else EXIT_SALVAGEABLE if result["salvageable"]
            else EXIT_INFRASTRUCTURE)
    passed = result["salvageable"] == expect

    print(f"--- {name} " + "-" * max(0, 52 - len(name)))
    print(f"  why          {why}")
    print(f"  files        {files or '(none)'}")
    print(f"  rc {returncode:<4}      salvageable {result['salvageable']}  "
          f"(expected {expect})   -> exit {code}")
    for line in result["report"]:
        print(f"  {line.strip()}")
    print(f"  RESULT       {'PASS' if passed else 'FAIL'}")
    print()
    return {"case": name, "returncode": returncode, "exit_code": code,
            "salvageable": result["salvageable"], "expected": expect,
            "checkpoints": result["checkpoints"], "passed": passed}


def severity_order() -> dict[str, Any]:
    """A salvageable seed must never mask a seed that produced nothing."""

    def worst_of(codes: list[int]) -> int:
        worst = EXIT_OK
        for code in codes:
            if train.SEVERITY.get(code, 99) > train.SEVERITY.get(worst, 0):
                worst = code
        return worst

    checks = [
        ([EXIT_SALVAGEABLE, EXIT_INFRASTRUCTURE], EXIT_INFRASTRUCTURE),
        ([EXIT_INFRASTRUCTURE, EXIT_SALVAGEABLE], EXIT_INFRASTRUCTURE),
        ([EXIT_OK, EXIT_SALVAGEABLE], EXIT_SALVAGEABLE),
        ([EXIT_OK, EXIT_OK], EXIT_OK),
    ]
    print("--- sweep verdict ordering " + "-" * 31)
    print("  EXIT_SALVAGEABLE is 4 and EXIT_INFRASTRUCTURE is 1, so max() "
          "over raw codes")
    print("  would report a sweep as salvageable when a seed produced "
          "nothing. It must not.")
    ok = True
    for codes, want in checks:
        got = worst_of(codes)
        good = got == want
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {codes} -> {got} (want {want})")
    print()
    return {"passed": ok}


def main() -> int:
    parser = argparse.ArgumentParser(prog="tools/fire_salvage_guard.py")
    parser.add_argument("--json", action="store_true")
    options = parser.parse_args()

    print("Firing train.py's salvage classifier on purpose (DESIGN.md §6).\n")

    with tempfile.TemporaryDirectory(prefix="fire-salvage-") as raw:
        results = [fire(*case, Path(raw)) for case in CASES]
    ordering = severity_order()

    ok = all(item["passed"] for item in results) and ordering["passed"]
    print("=" * 64)
    for item in results:
        print(f"  {item['case']:<28} exit {item['exit_code']}  "
              f"{'PASS' if item['passed'] else 'FAIL'}")
    print(f"  {'sweep verdict ordering':<28}        "
          f"{'PASS' if ordering['passed'] else 'FAIL'}")
    print()
    print("PASS — a lost run and a salvageable one are told apart"
          if ok else "FAIL")

    if options.json:
        json.dump({"results": results, "ordering": ordering, "passed": ok},
                  sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
