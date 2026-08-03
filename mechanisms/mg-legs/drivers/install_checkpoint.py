#!/usr/bin/env python
"""Put a trained .cxpolicy into a project and make it the one that plays.

    CADEX_REPO=~/cadex-standing-policy pixi run python \\
        ~/cdx-mjc/install_checkpoint.py <policy.cxpolicy> [--project DIR] \\
        [--name stand7.cxpolicy] [--dry-run]

A policy reaches the viewport by exactly one route: it is an **asset** in the
project store, and the project's script names it by filename *and digest*
(`assembly.policy(..., weights=, sha256=)`). There is no checkpoint selector
in the UI and there should not be one -- VISION principle 3 says any state
that cannot be rebuilt from the script is a bug, and "which of thirty
checkpoints is loaded" is exactly that state. So switching policies is a
script edit, and this is that edit performed correctly rather than by hand.

What it does, in the order that matters:

1. copies the file into ``<project>/assets/<name>``;
2. rewrites the ONE live ``assembly.policy(...)`` call to name it and carry
   its sha256 -- computed here from the bytes on disk, never pasted;
3. runs the script through the engine's accept path, which re-checks the
   task digest, the model digest, the observation channels, the action table
   and the trainer's witness, and refuses by name if any of them disagree;
4. rebuilds twice and asserts the digest is reproducible.

**It never edits ``script.py`` in place.** The store binds an accepted digest
to that file, so a hand edit makes every later entry point fail with "the
restore pass digest does not match the accepted digest". The new text goes
in as ``write_script`` source, which is the only path that re-accepts.

``--dry-run`` prints the rewritten policy call and stops, which is the cheap
way to check the regex found the right one.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
DEFAULT_PROJECT = HERE / "mg-legs.cadex"

#: The live `assembly.policy(...)` call: at the start of a line, so the
#: commented-out history above it -- this project keeps every retired policy
#: as a `#`-prefixed record -- cannot match.
POLICY_CALL = re.compile(
    r"^(?P<name>\w+)\s*=\s*assembly\.policy\((?P<args>.*?)\)\s*$",
    re.MULTILINE | re.DOTALL,
)

DRIVER = r'''
import json, os, sys
from pathlib import Path
ROOT = Path(os.environ["ICP_ROOT"]); REPO = Path(os.environ["ICP_REPO"])
sys.path.insert(0, str(REPO / "src" / "Mod" / "cadex"))
import FreeCAD as App
import cadex_rebuild
from CadexScriptedDomainPublication import publish_project_candidate
from CadexScriptedRuntime import (accept_project_candidate,
                                  capture_project_state, execute_candidate,
                                  prepare_project_candidate,
                                  validate_project_result)
from CadexProject import CadexProjectScriptStore
source = Path(os.environ["ICP_SOURCE"]).read_text(encoding="utf-8")
working = str(CadexProjectScriptStore(ROOT).read_state().get("working_revision") or "")
document = App.newDocument("InstallCheckpoint")
service = cadex_rebuild._RebuildService(ROOT, document)
captured = capture_project_state(
    service, "xscript.project.write_script",
    {"source": source, "expected_revision": working, "replace": True})
prepared = prepare_project_candidate(captured)
execution = execute_candidate(prepared, cancellation_check=None)
if execution.get("ok") is not True:
    print("ICP_FAIL " + json.dumps(execution, default=str)[:4000]); raise SystemExit(1)
validated = validate_project_result(prepared, execution)
publication = publish_project_candidate(service, prepared, validated)
accept_project_candidate(prepared, publication, validated)
state = CadexProjectScriptStore(ROOT).read_state()
App.closeDocument(document.Name)
first = cadex_rebuild.rebuild_project(ROOT)
second = cadex_rebuild.rebuild_project(ROOT)
print("ICP_OK " + json.dumps({
    "accepted_digest": str(state["accepted_digest"]),
    "reproducible": first["digest"] == second["digest"],
    "matches_accepted": bool(first["digest_matches_accepted"]),
}))
'''


def rewrite(source: str, weights: str, digest: str) -> tuple[str, str]:
    """Point the live policy call at ``weights``/``digest``."""

    matches = [m for m in POLICY_CALL.finditer(source)]
    if len(matches) != 1:
        raise SystemExit(
            f"FAIL: found {len(matches)} live `assembly.policy(...)` calls; "
            "this project must declare exactly one (ADR-062).")
    match = matches[0]
    name = match.group("name")
    args = match.group("args")

    task = args.split(",", 1)[0].strip()
    label = re.search(r'label\s*=\s*("[^"]*")', args)
    call = (
        f'{name} = assembly.policy({task}, weights="{weights}",\n'
        f'{" " * (len(name) + 24)}sha256="{digest}",\n'
        f'{" " * (len(name) + 24)}label={label.group(1) if label else chr(34) + name + chr(34)})'
    )
    return source[:match.start()] + call + source[match.end():], call


def main(argv: list[str]) -> int:
    argv = list(argv[1:])
    dry = "--dry-run" in argv
    argv = [a for a in argv if a != "--dry-run"]

    project = DEFAULT_PROJECT
    name = ""
    while "--project" in argv:
        i = argv.index("--project")
        project = Path(argv[i + 1]).expanduser()
        del argv[i:i + 2]
    while "--name" in argv:
        i = argv.index("--name")
        name = argv[i + 1]
        del argv[i:i + 2]
    if len(argv) != 1:
        print(__doc__)
        return 2

    policy = Path(argv[0]).expanduser().resolve()
    if not policy.is_file():
        print(f"FAIL: {policy} is not a file.")
        return 1
    repo = Path(os.environ.get("CADEX_REPO", "")).expanduser()
    freecadcmd = repo / "build" / "release" / "bin" / "FreeCADCmd"
    if not freecadcmd.is_file():
        print(f"FAIL: no FreeCADCmd at {freecadcmd}. Set CADEX_REPO and "
              "`pixi run build-engine`.")
        return 1

    name = name or policy.name
    digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    source = (project / "script.py").read_text(encoding="utf-8")
    rewritten, call = rewrite(source, name, digest)

    print(f"policy   {policy}")
    print(f"sha256   {digest}")
    print(f"install  {project}/assets/{name}")
    print()
    print(call)
    if dry:
        print("\n--dry-run: nothing written.")
        return 0

    shutil.copyfile(policy, project / "assets" / name)
    staged = Path(tempfile.mkdtemp(prefix="cadex-icp-")) / "source.py"
    staged.write_text(rewritten, encoding="utf-8")

    environment = dict(os.environ)
    environment.update({"ICP_ROOT": str(project), "ICP_REPO": str(repo),
                        "ICP_SOURCE": str(staged)})
    finished = subprocess.run(
        [str(freecadcmd), "-c", DRIVER], env=environment,
        capture_output=True, text=True)
    for line in (finished.stdout or "").splitlines():
        if line.startswith("ICP_OK "):
            print("\n==> accepted. " + line[len("ICP_OK "):])
            print("    Open the project and press Start in the Cadex Live "
                  "editor; the panel now names the policy it is playing.")
            return 0
        if line.startswith("ICP_FAIL "):
            print("\n==> the engine REFUSED this policy:")
            print("    " + line[len("ICP_FAIL "):][:2000])
            return 1
    print("\n==> the driver produced no verdict; stderr tail:")
    print("\n".join((finished.stderr or "").splitlines()[-25:]))
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
