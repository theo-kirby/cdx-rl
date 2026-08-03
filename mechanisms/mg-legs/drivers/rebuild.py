#!/usr/bin/env python
"""Rebuild mg-legs.cadex from script.py, headlessly, and say where the MJCF is.

    pixi run python ~/cadex-legs/rebuild.py [--script script.py]

Drives `cadexd` over NDJSON the way the shell does -- `open_project` then
`write_script` -- so what this proves is what the product would do. Prints
the accepted attempt's `outputs/` directory, which is what
`feasibility.py` reads and what a training run is dispatched from.

Lives beside the project rather than in the repository: the biped is not a
test (ADR-075 §6), and neither is its driver.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCHEMA = "cadex-cadexd-v1"
REPO = Path(os.environ.get("CADEX_REPO", Path.home() / "cadex"))
CADEX_ROOT = REPO / "src" / "Mod" / "cadex"
FREECADCMD = REPO / "build" / "release" / "bin" / "FreeCADCmd"
PROJECT = Path(__file__).resolve().parent / "mg-legs.cadex"


class Cadexd:
    def __init__(self) -> None:
        self.process = subprocess.Popen(
            [str(FREECADCMD), "-c",
             f"import sys; sys.path.insert(0, {str(CADEX_ROOT)!r}); "
             "import cadexd; raise SystemExit(cadexd.main())"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env={**os.environ, "PYTHONHASHSEED": "0"},
        )
        self.n = 0
        self.pending: dict[str, dict] = {}
        assert self.frame(120.0).get("event", {}).get("event") == "ready"

    def frame(self, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() <= deadline:
            line = self.process.stdout.readline()
            if not line:
                raise EOFError("cadexd closed its stream")
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                continue           # anything printed before the fd hijack
            if isinstance(value, dict):
                return value
        raise TimeoutError("no cadexd frame within the timeout")

    def request(self, op: str, args: dict | None = None,
                timeout: float = 900.0) -> dict:
        self.n += 1
        rid = f"r{self.n}"
        frame = {"schema": SCHEMA, "id": rid, "op": op}
        if args is not None:
            frame["args"] = args
        self.process.stdin.write(json.dumps(frame).encode() + b"\n")
        self.process.stdin.flush()
        if rid in self.pending:
            return self.pending.pop(rid)
        deadline = time.monotonic() + timeout
        while True:
            value = self.frame(max(0.1, deadline - time.monotonic()))
            if "event" in value:
                continue
            if value.get("id") == rid:
                return value
            self.pending[str(value.get("id"))] = value

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=30)


def main(argv: list[str]) -> int:
    source = Path(argv[1] if len(argv) > 1 else PROJECT / "script.py")
    if not FREECADCMD.exists():
        print(f"FAIL: {FREECADCMD} is not built. Run: pixi run build-engine")
        return 1
    text = source.read_text(encoding="utf-8")

    client = Cadexd()
    try:
        opened = client.request("open_project", {"project_root": str(PROJECT)})
        if not opened.get("ok"):
            # The restore pass re-runs the accepted script and compares
            # digests, so a project last built by a different engine build
            # refuses to open (MUJOCO.md hazard 3). `restore=false` is the
            # documented escape hatch and re-accepting records the new
            # digest; it is taken loudly rather than by default, because on
            # any script we were NOT about to replace it would be a silent
            # substitution of one physics for another.
            if opened.get("failure_code") != "CADEXD_RESTORE_FAILED":
                print("FAIL: open_project\n"
                      + json.dumps(opened, indent=1)[:3000])
                return 1
            print("==> the stored digest does not match a restore pass:")
            print("    " + json.dumps(opened.get("observed") or {}))
            print("==> reopening with restore=false; this write re-accepts.")
            opened = client.request(
                "open_project", {"project_root": str(PROJECT),
                                 "restore": False})
            if not opened.get("ok"):
                print("FAIL: open_project\n"
                      + json.dumps(opened, indent=1)[:3000])
                return 1

        # The store guards every write with the revision the caller believed
        # it was editing. `open_project` reports it, nested differently
        # depending on whether the project had an accepted script.
        def find(node, key):
            if isinstance(node, dict):
                if key in node and isinstance(node[key], str) and node[key]:
                    return node[key]
                for value in node.values():
                    found = find(value, key)
                    if found:
                        return found
            elif isinstance(node, list):
                for value in node:
                    found = find(value, key)
                    if found:
                        return found
            return ""

        # The store's own record first. `find` walks the response looking for
        # any key called `revision`, and once a write has been REFUSED the
        # response carries a `latest_candidate` whose revision is the
        # rejected one -- so after any failed rebuild the next one guessed a
        # revision the store had never accepted and was refused for that
        # instead, which reads exactly like the original failure and is not.
        stored = json.loads((PROJECT / "script.json").read_text())
        expected = (str(stored.get("accepted_revision") or "")
                    or find(opened, "next_write_expected_revision")
                    or find(opened, "revision"))

        written = client.request(
            "write_script",
            {"source": text, "replace": True, "expected_revision": expected})
        # ...and if the guess was wrong, the refusal SAYS what the revision
        # is, so take it and go again. Both ways of guessing have been wrong:
        # the response carries a rejected candidate's revision after a failed
        # build, and `script.json` on disk is stale after a `restore=false`
        # reopen has re-accepted under a new one. One retry against the
        # store's own answer beats a third theory about where to read it.
        if (not written.get("ok")
                and written.get("failure_code") == "STALE_PROGRAM_REVISION"):
            current = str((written.get("observed") or {})
                          .get("current_revision") or "")
            if current and current != expected:
                print(f"==> the store is at {current[:12]}, not "
                      f"{expected[:12]}; writing against that")
                written = client.request(
                    "write_script",
                    {"source": text, "replace": True,
                     "expected_revision": current})
        if not written.get("ok"):
            print("FAIL: write_script\n" + json.dumps(written, indent=1)[:6000])
            return 1

        result = written.get("result", written)
        revision = result.get("revision") or result.get("accepted_revision")
        print(f"==> accepted revision {revision}")
        for output in result.get("outputs") or ():
            print(f"    {output.get('domain','?'):9s} "
                  f"{output.get('name','?'):14s} {output.get('type','?')}")

        # THE SLIDERS ARE THE RESET POSE, so the driver puts them back to
        # what the script says rather than trusting them. A `num(...)` in the
        # source is only a default: the project stores its own accepted
        # values, and this one had a 49-degree hip and an 80-degree knee left
        # over from posing it by hand. Training against that would bake a
        # contorted stance into every episode, silently, because nothing
        # about a stored parameter is visible in the script.
        #
        # IT IS THE SCRIPT'S DEFAULT AND NOT ZERO, and B8 is why that
        # distinction had to be made. Through B7 the nominal pose WAS all
        # zeros, so "zero it" and "restore the default" were the same
        # instruction and this block wrote 0.0. B8's nominal pose is a
        # crouch declared in those defaults, so a driver that zeroed them
        # would silently straighten the machine's legs back out and train a
        # different robot than the one every measurement was taken on --
        # which is hazard 9 arriving through the back door.
        state = json.loads((PROJECT / "script.json").read_text())
        default = {str(spec["name"]): float(spec["default"])
                   for spec in (state.get("param_specs") or ())
                   if spec.get("default") is not None}
        pose = {name: default[name]
                for name in sorted(state.get("param_values") or ())
                if name.startswith(("hip_pitch", "knee_pitch", "ankle_pitch",
                                    "hip_roll", "ankle_roll"))
                and name in default}
        stale = {name: value
                 for name, value in (state.get("param_values") or {}).items()
                 if name in pose and abs(float(value) - pose[name]) > 1.0e-9}
        if stale:
            print("==> the stored pose is not the script's; restoring it")
            for name in sorted(stale):
                print(f"    {name:16s} {stale[name]:+8.3f} -> {pose[name]:+.3f}")
            zeroed = client.request(
                "set_params", {"values": pose, "expected_revision": revision})
            if not zeroed.get("ok"):
                print("FAIL: set_params\n"
                      + json.dumps(zeroed, indent=1)[:6000])
                return 1

        # The staging directory of the attempt that was just accepted.
        state = json.loads((PROJECT / "script.json").read_text())
        staging = PROJECT / state["accepted_attempt"]["staging"]
        outputs = staging / "outputs"
        print(f"==> outputs {outputs}")
        for path in sorted(outputs.glob("*.xml")) + sorted(outputs.glob("*.json")):
            print(f"    {path.name}")
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
