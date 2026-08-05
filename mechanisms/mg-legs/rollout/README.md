# mg-legs rollout — a policy, ready to open in Cadex

A duplicate of `mechanisms/mg-legs/` set up for one job: play a trained policy
and hand the result to the **Shell** for visual verification. `cadex.md` §2 is
the division of labour — cdx-rl drives the engine and the trainer, and sends
hero results to the Shell.

```bash
uv run python mechanisms/mg-legs/rollout/build.py --arm b8       # works
uv run python mechanisms/mg-legs/rollout/build.py --arm clamp25  # see §"Blocked"
```

| arm | policy | what it is |
|---|---|---|
| `b8` *(default)* | `stand10.001700` | experiment 003 seed 2 — **18/24** on the conjunction, and **51.8 %** resting duty above 90 % of servo rating |
| `clamp25` | `stand13.001800` | experiment 005 — the **same 18/24** (paired McNemar p = 1.000) at **12.6 %** duty. The buildable one |

## On the Mac

```bash
cd ~/cdx-rl && git pull
set -a; . ./config/env; set +a
uv run python mechanisms/mg-legs/rollout/build.py --arm b8
```

Then open `projects/mg-legs-rollout.cadex` in the Shell.

## Why the Mac, and what changed on 2026-08-05

**The old reason is gone. A new one replaced it.**

This file used to say the script would not *build* anywhere but the laptop,
because `sb1x` and `sb9x` refused `centre_of_mass_velocity` and
`centroidal_angular_momentum` by name. That table is **stale**: both kinds
landed upstream in `593f64e6` (ADR-116), sb1x now tracks `origin/main`, and
`script.py` builds here in about 8 seconds.

What still requires the Mac is **replay**, and it is a digest problem rather
than a capability one. `assembly.policy(..., sha256=)` makes the engine check
that the policy's recorded task digest equals the one the script just built.
That digest is a **hash of the whole task JSON**, and the JSON embeds the
MJCF's digest — which differs between macOS and Linux by **one float**:

```
pelvis inertial pos x   5.10066e-11  (macOS, where B8 was authored)
                        5.10087e-11  (sb1x, Linux)
```

2.1 × 10⁻¹⁵ m, on a coordinate that is **zero by symmetry**. Every other line
of the 14 179-byte file is identical. But it propagates into two digests, so
on Linux the b8 arm is refused:

```
trained on  5572adf265aa51cb…
declared against  0b4d160cd436fd16…
```

`cadex-wishlist.md` #15 is the entry. The refusal is *correct in principle* —
a policy is only meaningful for the task it was trained on — and wrong in this
instance, which is what makes it worth fixing rather than working around.

## Blocked: `--arm clamp25` cannot be replayed from the script yet

**Not a platform problem — it fails on the Mac too, and the cause is ours.**

`stand13.001800` was trained against `tasks/stand-b8-clamp25/`, which
experiment 004 produced by **editing the derived bundle by hand**
(`make_clamp_bundle.py`), because there was no way to say "±25° of a ±45°
joint" in the mechanism vocabulary. ADR-123 added that way, and
`script-clamp25.py` now expresses the same arm from source. Every
action-range **number** matches — all ten actuators, `low`, `high`, `unit`,
`scale`. But the bundles are not byte-identical:

| | hand-edited (what the policy was trained on) | from `script-clamp25.py` |
|---|---|---|
| `actions[].source` | `angle_limits_degrees` | `command_limits_degrees` |
| `label` | `stand12` | `stand` |

`source` changed **deliberately** — a bundle must not report the joint's
limits as the origin of a number they did not produce — and it moves the file
hash, so the task digest moves, so the policy is refused:

```
trained on  3d627ef4b9a509fe…
declared against  bd8071b50360eaab…
```

**This is the cost of ADR-123's honesty about provenance, and it is worth
naming.** `source` is deliberately absent from `_POLICY_ACTION_FIELDS`, so
`verify_policy` does not care about it *per action* — but the **task digest is
a whole-file hash**, which conflates *"this is a different task"* with *"this
bundle was produced by a different route"*. The action space here is
identical; only the provenance string differs. That is the same conflation as
#15 one level up, and it is the strongest argument for the entry.

### What to do about it

* **To view the clamp25 gait now:** nothing cheap. The policy is tied to the
  bundle it trained on and no script produces that bundle any more.
* **To make it viewable:** train the next clamp25 arm against the
  **script-generated** bundle (`bd8071b5…`), which is now the canonical one.
  Every policy from that run replays from `script-clamp25.py` on the Mac, and
  the hand-edited bundle can be retired to the record.
* **Do not** revert `source` to make the digests line up. It would buy one
  replay and reintroduce a bundle that misreports where its numbers came from.

`script-clamp25.py` is committed regardless, because it is the thing that
makes experiments 004 and 005 **reproducible from source** — which is success
criterion 5, and it was the whole point of ADR-123.

### …and what was built instead, 2026-08-05

The operator wanted eyes on a policy, and this path could not give them
today. `harness capture` was built instead:

```bash
uv run python -m harness capture --dir jobs/stand13-20260805-135926 \
    --iteration 1800 --seeds 4 --tile
# → video/stand13.001800.tile.mp4
```

It renders the policy driving its own episode to an MP4 on this box, in
about two seconds of CPU for four seeds, with no second machine and no GUI.
`harness/DESIGN.md` has the section.

**It is not a replacement for this path, and the difference is exactly the
one that matters here.** `capture` renders `mjVIS_INERTIA` boxes — the mass
distribution — because `model-model.xml` carries 5 geoms and would otherwise
render as an empty floor, and because adding visual geoms would move the
digest and produce a video of a different machine. **The Shell shows the real
CAD solids and `capture` never will.** So this page stays live.

What it is blocked on is unchanged and is stated once more because it is the
thing that unblocks it: **train the next clamp25 arm against the
script-generated bundle `bd8071b5…`**. That is a GPU run, not a pipeline. Do
**not** revert ADR-123's `source` change to make the digests line up — it
would buy one replay and reintroduce a bundle that misreports where its
numbers came from.

## Notes worth keeping

* **`put_asset` moves the project revision even though it is not a write.**
  `CadexdClient.put_asset` calls `refresh_revision()` for exactly this.
* **The digest is never pasted.** `assembly.policy(..., sha256=)` is checked
  against the bytes `build.py` is about to install, and `build.py` says which
  of the two files is wrong rather than letting the engine say it later.
* **`POLICY_CALL` is line-anchored on purpose.** This script keeps every
  retired policy as a `#`-prefixed record — six of them above the live one —
  and an unanchored match finds the oldest. Writing `script-clamp25.py` hit
  exactly that trap once before the anchor was respected.
