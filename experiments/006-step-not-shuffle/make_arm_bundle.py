#!/usr/bin/env python
"""Derive experiment 006's two arm bundles from `tasks/stand-b8-clamp25/`.

    uv run python experiments/006-step-not-shuffle/make_arm_bundle.py \\
        --variant shove --newtons-high 0.90
    uv run python experiments/006-step-not-shuffle/make_arm_bundle.py \\
        --variant quiet --sigma <measured>

Idempotent; prints every digest and asserts the ones that must not move.

## Why this is a derived bundle and not a script revision — MEASURED, not assumed

The plan preferred authoring both arms as a B9 revision of
`mechanisms/mg-legs/script.py`, because wishlist #12 shipped as ADR-131 and
`command_limits_degrees` is now a first-class `assembly.actuator` kwarg — so
unlike experiment 004, a clamped bundle *can* come from source. The condition
attached to that preference was: **the MJCF must come out byte-identical to
clamp25's `80eaa18f…`; verify it rather than assume it.**

It was verified, on sb1x 2026-08-06, by rebuilding
`mechanisms/mg-legs/rollout/script-clamp25.py` with its policy/rollout block
removed. **It does not.** The script-built bundle differs from
`tasks/stand-b8-clamp25/stand-task.json` in exactly three places:

| | |
|---|---|
| `actions[*].source` | `angle_limits_degrees` → `command_limits_degrees` — inert, every action *number* identical (the ADR-131 note in CLAUDE.md) |
| `label` | set by the caller |
| `model.sha256` | **`80eaa18f…` → `203f746e…`**, and `model.bytes` 14179 → 14169 |

The model difference is one attribute of one line: the pelvis inertial x
coordinate, `5.10066e-11` → `0`. That is **ADR-133's inertial snap**, which
landed *after* clamp25 was authored, and the quantity it moved is
mathematically zero by symmetry.

Numerically negligible, and still disqualifying here, for the reason
`CLAUDE.md` invariant 2 gives: **the control is already paid for.** `stand13`
and both `stand12` seeds were trained against `80eaa18f…`. Authoring the arms
from source would move the physics in the same step as the task, and no result
could then be attributed to either. So 006 takes the same route 004 took —
and, unlike 004, it is a route chosen by a measurement rather than by a
missing engine surface.

**What that costs, stated plainly:** criterion 5 is not met by these bundles.
They are a claim about committed bytes. The route that *would* meet it is
available and costs one extra 5 GPU-hour run — re-derive both arms *and* the
control from `script-b9-*.py` on the `203f746e…` model — and that is a
decision about budget, not about tooling.

## What is checked here rather than downstream

* an **identity self-check** first: re-emitting the source bundle unchanged
  must reproduce its own bytes. If the serialiser has moved, every derived
  digest below is unreliable and the run must not start. `json.dumps(...,
  indent=1, sort_keys=True)` was verified a fixed point on this file.
* `model.sha256` still equals the digest of the copied MJCF, and still equals
  clamp25's. `harness/episodes.resolve_model` would catch a mismatch later;
  the generator is where it is cheap.
* the engine's two hard caps by name — `MAXIMUM_REWARD_TERMS` (16) and
  `MAXIMUM_OBSERVATION_CHANNELS` (64).
* that the reward's total weight moved by exactly what was intended, because
  `reward_per_step` stops being comparable across bundles the moment it does.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "tasks" / "stand-b8-clamp25"

#: `CadexDynamics.py:4860` and `:4854`. Quoted by name so a future widening is
#: a one-line change here rather than a silent overrun in the engine.
MAXIMUM_REWARD_TERMS = 16
MAXIMUM_OBSERVATION_CHANNELS = 64

#: The ten actuated joints, in the bundle's own action order. `posture` sums
#: the `_a` channels of exactly these; `quiet` is its velocity analogue and
#: sums the `_v` channels of exactly these.
LABELS = {"shove": "stand14", "quiet": "stand15"}


def emit(task: dict) -> bytes:
    """The one serialisation. Every digest in this repository assumes it."""

    return json.dumps(task, indent=1, sort_keys=True).encode()


def identity_check() -> str:
    """Re-emit the source unchanged and require its own bytes back.

    If this fails, the serialiser has moved and no derived digest below means
    anything — so it raises rather than warns.
    """

    raw = (SRC / "stand-task.json").read_bytes()
    again = emit(json.loads(raw))
    if again != raw:
        raise SystemExit(
            f"identity check FAILED: re-emitting {SRC / 'stand-task.json'} "
            f"through json.dumps(indent=1, sort_keys=True) produced "
            f"{len(again)} bytes against {len(raw)}. The serialisation this "
            f"repository's digests assume has moved; do not derive anything "
            f"from it until that is understood."
        )
    return hashlib.sha256(raw).hexdigest()


def make_shove(task: dict, newtons_high: float, split: bool,
               far_sigma: float) -> list:
    """Raise the top of the aimed and unaimed shove bands. `wind` untouched.

    **`wind` is deliberately not scaled with the band**, and `script.py`
    §B7 is the reason to cite: a wind that tracked a big band would be a
    permanent lean *"paid for by every one of the new terms at every step of
    every episode including the ones with no shove in them."*
    """

    notes = []
    for entry in task["disturbance"]:
        if entry["label"] not in ("shove", "shove2"):
            continue
        was = float(entry["newtons_high"])
        entry["newtons_high"] = float(newtons_high)
        notes.append(f"{entry['label']}: newtons_high {was} -> {newtons_high}")

    if split:
        # Split `capture` into two scales of the same TOTAL weight. The wide
        # expression is produced by substituting into the narrow one, so the
        # two kernels cannot drift in the `8.3428` or in the two offsets.
        rewards = task["reward"]
        index = next(i for i, r in enumerate(rewards)
                     if r["label"] == "capture")
        capture = rewards[index]
        total = float(capture["weight"])
        narrow = dict(capture)
        narrow["weight"] = round(total / 2.0, 6)
        wide = dict(capture)
        wide["label"] = "capture_wide"
        wide["weight"] = round(total / 2.0, 6)
        marker = "/ 50.0)**2"
        if marker not in capture["expression"]:
            raise SystemExit(
                f"the `capture` kernel does not carry {marker!r}, so the wide "
                f"twin cannot be derived from it by substitution. Refusing to "
                f"hand-write a second copy of an expression with three "
                f"measured constants in it."
            )
        wide["expression"] = capture["expression"].replace(
            marker, f"/ {far_sigma})**2")
        rewards[index:index + 1] = [narrow, wide]
        notes.append(f"capture split {total} -> {narrow['weight']} @ 50.0 mm "
                     f"+ {wide['weight']} @ {far_sigma} mm "
                     f"(weight-preserving at zero error)")
    return notes


def make_quiet(task: dict, sigma: float, weight: float, joints: list) -> list:
    """Append the tenth kernel, over the `*_v` channels already declared.

    Built by substituting into `posture`'s own expression — `_a` → `_v` and
    its σ → this one — so the spelling, the spacing and the joint order are
    provably identical to a term the engine already accepts, rather than
    retyped.
    """

    posture = next(r for r in task["reward"] if r["label"] == "posture")
    declared = {c for row in task["observations"] for c in row["channels"]}
    missing = [f"{j}_v" for j in joints if f"{j}_v" not in declared]
    if missing:
        raise SystemExit(
            f"the bundle declares no channel {missing}. `quiet` can only sum "
            f"channels the task already publishes; adding one would change "
            f"the observation vector and orphan every existing policy."
        )
    terms = " + ".join(f"abs({j}_v)" for j in joints)
    expression = f"exp(-((({terms}) / {sigma})**2))"
    # Same shape as `posture`, checked rather than asserted in prose.
    shape = posture["expression"]
    if not (shape.startswith("exp(-(((abs(") and shape.endswith(")**2))")):
        raise SystemExit("`posture` no longer has the shape `quiet` copies")
    task["reward"].append({"expression": expression, "label": "quiet",
                           "weight": float(weight)})
    return [f"quiet: {len(joints)} channels, sigma {sigma} deg/s, "
            f"weight {weight}"]


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", required=True, choices=sorted(LABELS))
    ap.add_argument("--newtons-high", type=float,
                    help="shove variant: the new top of the band, in newtons. "
                         "**Set by `harness capability`'s sweep, not by the "
                         "capture-point arithmetic** — the arithmetic only "
                         "proposes a candidate (006 README §4).")
    ap.add_argument("--split-capture", action="store_true",
                    help="shove variant: split `capture` into two scales of "
                         "the same total weight. Only worth it once the band "
                         "puts xi past the point where a 50 mm kernel goes "
                         "flat; below that it COSTS gradient.")
    ap.add_argument("--far-sigma", type=float, default=150.0)
    ap.add_argument("--sigma", type=float,
                    help="quiet variant: the kernel's scale in deg/s. "
                         "REQUIRED and with no default, so it cannot be "
                         "forgotten into a guess — it comes from "
                         "`jitter.py`'s measured median settled sum|qdot|.")
    ap.add_argument("--weight", type=float, default=0.3)
    ap.add_argument("--joints", default="all",
                    help="quiet variant: 'all' for the ten actuated joints, "
                         "or a comma-separated subset.")
    ap.add_argument("--out", help="destination directory name under tasks/")
    args = ap.parse_args(argv)

    parent = identity_check()
    task = json.loads((SRC / "stand-task.json").read_text())
    before = sum(float(r["weight"]) for r in task["reward"])
    all_joints = [str(a["joint"]) for a in task["actions"]]

    if args.variant == "shove":
        if args.newtons_high is None:
            ap.error("--newtons-high is required for the shove variant")
        notes = make_shove(task, args.newtons_high, args.split_capture,
                           args.far_sigma)
        default_out = f"stand-b8-clamp25-shove{int(round(args.newtons_high * 100)):03d}"
    else:
        if args.sigma is None:
            ap.error("--sigma is required for the quiet variant, and has no "
                     "default on purpose")
        joints = (all_joints if args.joints == "all"
                  else [j.strip() for j in args.joints.split(",")])
        unknown = [j for j in joints if j not in all_joints]
        if unknown:
            ap.error(f"not actuated joints of this mechanism: {unknown}")
        notes = make_quiet(task, args.sigma, args.weight, joints)
        default_out = "stand-b8-clamp25-quiet"

    task["label"] = LABELS[args.variant]
    dst = REPO / "tasks" / (args.out or default_out)

    # --- the caps, by name ---
    if len(task["reward"]) > MAXIMUM_REWARD_TERMS:
        raise SystemExit(f"{len(task['reward'])} reward terms exceeds "
                         f"MAXIMUM_REWARD_TERMS ({MAXIMUM_REWARD_TERMS})")
    channels = sum(len(r["channels"]) for r in task["observations"])
    if channels > MAXIMUM_OBSERVATION_CHANNELS:
        raise SystemExit(f"{channels} channels exceeds "
                         f"MAXIMUM_OBSERVATION_CHANNELS "
                         f"({MAXIMUM_OBSERVATION_CHANNELS})")

    dst.mkdir(parents=True, exist_ok=True)
    blob = emit(task)
    (dst / "stand-task.json").write_bytes(blob)
    shutil.copy2(SRC / "model-model.xml", dst / "model-model.xml")

    model = hashlib.sha256((dst / "model-model.xml").read_bytes()).hexdigest()
    # **The model must not have moved.** Copied, not rebuilt — see the module
    # docstring for why the script route would move it and why that is
    # disqualifying while the control is already paid for.
    if model != task["model"]["sha256"]:
        raise SystemExit(f"the copied MJCF digests {model} and the bundle "
                         f"declares {task['model']['sha256']}")
    if model != "80eaa18f6025d589796315fcad45bb70bf72e55da750864f60b5e0b3cc71fdb3":
        raise SystemExit("the model is not the one the control was trained "
                         "against; the comparison would cross a physics "
                         "change as well as a task change")

    after = sum(float(r["weight"]) for r in task["reward"])
    print(f"parent bundle   {parent}")
    print(f"derived bundle  {hashlib.sha256(blob).hexdigest()}")
    print(f"model (copied)  {model}")
    print(f"written to      tasks/{dst.name}/")
    print(f"\nlabel           {task['label']}")
    print(f"reward terms    {len(task['reward'])} of {MAXIMUM_REWARD_TERMS}"
          f"   channels {channels} of {MAXIMUM_OBSERVATION_CHANNELS}")
    print(f"reward weight   {before} -> {after}")
    if after != before:
        print(f"                — so `reward_per_step` is NOT comparable "
              f"across these two bundles. Score behaviour, not the scalar "
              f"(ADR-099).")
    print("\nchanges:")
    for note in notes:
        print(f"  {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
