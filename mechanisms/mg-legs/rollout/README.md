# mg-legs rollout — a policy, opened on the real solids and pushed

A duplicate of `mechanisms/mg-legs/` set up for one job: play a trained policy
and hand the result to the **Shell** for visual verification, on the CAD solids,
where the mouse can shove the machine and the policy answers. `cadex.md` §2 is
the division of labour — cdx-rl drives the engine and the trainer, and sends
hero results to the Shell.

**Both arms work, on both machines, as of 2026-08-05.** This page used to have a
"Blocked" section; what unblocked it was three engine PRs, not a GPU run.

| arm | policy | what it is |
|---|---|---|
| `b8` *(default)* | `stand10.001700` | experiment 003 seed 2 — **18/24** on the conjunction, and **51.8 %** resting duty above 90 % of servo rating |
| `clamp25` | `stand13.001800` | experiment 004 — the **same 18/24** (paired McNemar p = 1.000) at **12.6 %** duty. The buildable one |

## The loop

```bash
# on sb1x — assemble a replay set and answer "will this replay?" HERE
uv run python -m harness replay --export \
    --dir jobs/stand13-20260805-135926 --iteration 1800 \
    --task tasks/stand-b8-clamp25/stand-task.json \
    --arm clamp25 --script mechanisms/mg-legs/rollout/script-clamp25.py \
    --label clamp25
uv run python -m harness replay --preflight clamp25      # → WILL REPLAY

# ship it — the driver prints this line for you with --scp mmini
scp -r replay/clamp25 mmini:~/cdx-rl/replay/

# on the Mac
set -a; . ./config/env; set +a
uv run python -m harness replay --import replay/clamp25   # digests re-checked
uv run python mechanisms/mg-legs/rollout/build.py --arm clamp25
uv run python tools/live_probe.py projects/mg-legs-rollout-clamp25.cadex
```

There is **no digest pasted anywhere** in that. The set carries every one, the
importer re-checks them on arrival, and the engine re-checks them again at
`live_open`.

`replay --scp mmini` prints the transfer line instead; `replay --pending` prints
the `prepare_artifact_uploads` payload if the graph is the transport. Both are
supported and the trade is stated out loud: scp is faster and leaves no record.

## Why a replay set and not a project store

A built `.cadex` store is 2–5 MB and would also work — but it makes the far
machine a pure consumer, and **the rebuild is the point**: it is what produces
the BREP solids the Shell renders. A store full of somebody else's BREPs cannot
be edited, varied or re-derived.

A set is four files and about 380 kB: the `.cxpolicy`, the **training** task
bundle, the **training** MJCF, and a manifest. The script does *not* travel —
it is committed, so both machines have it, and the manifest records its digest
so a mismatch is reported.

## What was actually blocking this, and what fixed it

Three things, and none of them was a capability.

**1. The platform split — one float.** `mg-legs`' pelvis centre of mass is zero
in x by symmetry. OCCT read it as `5.10066e-11` on macOS and `5.10087e-11` on
Linux, and that line was the *only* difference between the two 14 179-byte MJCF
files. It moved the model digest, which moved the bundle digest, which made
every policy refused on the other machine.

[**ADR-133**](https://github.com/theo-kirby/cadex/pull/3) snaps inertial
coordinates below a nanometre to exactly zero. Measured after, same script,
byte-identical engine sources:

| | script build digest | MJCF | task bundle |
|---|---|---|---|
| macOS 26, arm64 | `560a33a4bfce810e…` | `203f746e9bb8a857…` | `6dc1c580f4bcd01a…` |
| Ubuntu 24.04, x86-64 | `560a33a4bfce810e…` | `203f746e9bb8a857…` | `6dc1c580f4bcd01a…` |

`cmp` reports both pairs identical. **`cadex-wishlist.md` #15 is closed.**

**2. That fix orphaned every policy already trained** — snapping changes the
Linux and macOS MJCF alike, so a pre-snap policy's recorded `model.sha256` stops
matching a freshly built model. And `clamp25` had a second version of the same
problem: its bundle was produced by hand before ADR-131, so it reports
`actions[].source` as `angle_limits_degrees` where the script now honestly
reports `command_limits_degrees`. Every action *number* is identical.

[**ADR-134**](https://github.com/theo-kirby/cadex/pull/4) added
`assembly.policy(..., trained_task=)`. The policy is bound to its own
travelling bundle whole-file — **unweakened** — and the locally built bundle is
then proved *equivalent*: every behaviour-deciding field, plus the two models
compared as models rather than as hashes. This is stronger than relaxing the
digest check, not weaker.

**3. The store would not hold the two travelling files.**
[**ADR-135**](https://github.com/theo-kirby/cadex/pull/5): `put_asset` accepted
`.cxpolicy`, `.obj`, `.ply`, `.stl` and nothing else, so ADR-134's whole surface
was unusable and all 52 of its unit tests passed anyway. Found on the first
end-to-end run, at step one.

### Measured, both arms, both boxes

| arm | trained on | script built | same task (semantic) | verdict |
|---|---|---|---|---|
| `b8` | `5572adf265aa…` *(macOS, pre-snap)* | `6dc1c580f4bc…` | `6bb66e9bcafa…` | **accepted** |
| `clamp25` | `3d627ef4b9a5…` *(hand-edited, pre-ADR-131)* | `3dbc680589b1…` | `17f1f46fbfcf…` | **accepted** |

`clamp25` is the one worth reading twice. **No script could produce that bundle
any more**, and this page previously said the only fix was to retrain. It was
not: the fix cost no GPU time and reverted no correctness fix. ADR-131's
`source` string still says `command_limits_degrees`.

### …and the refusals still refuse

Five mutations of the *script*, each refused, each naming the field:

| mutation | what the refusal said |
|---|---|
| reward weight 0.2 → 0.9 | `reward[0].weight: 0.9 here, 0.2 there` |
| episode 6.0 s → 9.0 s | `episode.episode_seconds: 9.0 here, 6.0 there` |
| tip threshold 0.15 → 0.25 | `termination[0].above: 0.25 here, 0.15 there` |
| command range ±25° → ±30° | `actions[0].high: 30.0 here, 25.0 there` |
| bracket plate 2.5 → 2.9 mm | `body_ipos: 0.0363 relative drift (0.000727 absolute)` |

The last one changes **no field of the task bundle at all** — same joints, same
limits, same action table — and is caught only by the model comparison. That is
why that comparison exists.

## Live mode: the push, and the answer

`live_open` / `live_step` / `live_close` are `cadexd` ops (ADR-109/110/111), so
the loop is provable without a GUI. `tools/live_probe.py` is that proof, and it
is the same three ops the Shell's panel drives:

```
$ uv run python tools/live_probe.py projects/mg-legs-rollout-clamp25.cadex
live      50 Hz, 6.0 s, 24 components, 10 channels
policy    'balance'  stand13.001800.cxpolicy  c6bb20c4579c…  trained_label='stand13'

settled   pelvis z = 298.6 mm at t = 0.48 s
pushed    0.6 N at azimuth 0 for 60 ms on pelvis -> z = 298.2 mm
answered  z = 296.8 mm at t = 2.98 s
          terminated=False  termination=''  resets=0

THE POLICY ANSWERED THE PUSH: still standing 2.98 s in, -1.8 mm of the settled height.
```

Identical on sb1x and on the Mac. 0.6 N is inside the task's own declared
0.3–0.8 N band (ADR-106's revision, experiment 001 phase B), so it is a shove
the policy was trained to expect rather than a stunt.

`prepare_live` re-checks three digests — the model, the bundle and the weights —
and plays *the exact files the accepted rollout used*. A successful `live_open`
**is** those three checks passing.

## ⚠️ The Shell app needs reinstalling before the mouse works

**Everything above is `cadexd`. The GUI is a separate payload and it is stale.**

`/Applications/Cadex.app` ships its own engine under
`Contents/Resources/cadex/`, staged **2026-08-03**. It predates ADR-131, let
alone the three PRs above. Measured on 2026-08-05: pointed at that payload,
`open_project` on a project the current engine built refuses with

```
The restore pass digest does not match the accepted digest.
```

and opens only stores built before the Mac's own tree merged `origin/main`. So
the Shell is out of sync with the operator's own checkout, and was **before**
any of this work — this is not a regression introduced here.

The `CadexLiveSession.py` in the bundle is byte-identical to the repo's, so the
panel itself is current. It is the engine payload beside it that is not.

To fix it, from a Cadex checkout that has `main` at `a40656cc` or later:

```bash
pixi run install-app      # build-engine + build-shell + stage, then /Applications
```

Two things to know before running it. It is a **local** install — the bundle
resolves its libraries out of whichever repository built it, so that repo has to
stay put; and it therefore rebinds `/Applications/Cadex.app` to that checkout.
`docs/cadex-release-packaging.md` has the detail. This was left for the operator
deliberately: it overwrites an installed application and rebinds it, which is
not a change to make on someone's behalf.

## Notes worth keeping

* **The mechanism is now platform-identical; the simulation is not.** After
  ADR-133 the MJCF and the task bundle are byte-identical on macOS-arm64 and
  linux-64. The **rollout trace** still is not — `d7cf5c5faa19f171` against
  `d598a51eb615483f` for the same 152-frame episode — and neither is the policy
  receipt, which differs in exactly one field: `witness_error`,
  `1.2678048740610848e-07` against `1.2678048737058133e-07`. Nine significant
  figures of agreement, ~800× inside the tolerance. That is MuJoCo's disclaimed
  cross-platform reproducibility (hazard 3) and no snap can fix it. **It is also
  why the far machine rebuilds rather than receiving a store**: the script build
  digest covers the trace, so a store built on sb1x is refused at `open_project`
  on the Mac. The architecture was right; now there is a measurement saying why.
* **`uv` is not installed on the Mac.** `~/cdx-rl/.venv` exists, so use
  `.venv/bin/python -m harness …` there. `pixi` is at `~/.pixi/bin/pixi` and not
  on a non-interactive `ssh` PATH.
* **`--iteration 1800` means the file tagged `001800`, and the trainer calls it
  1799.** Two conventions live on disk: `series_checkpoints` reads the filename
  tag and `discover_policies` reads `progress.json`. `capture` and `replay` are
  both pinned to the tag, and `test_replay.py` asserts they agree — the first
  version of `replay` used the other one and refused `--iteration 1800` on a run
  holding exactly that file.
* **A live push needs a `body`.** Without one the reply is `live: false` with a
  reason and **zero frames** — a successful op that declined, exactly as
  `live_open` declines a project with no rollout. Reading `frames[-1]` without
  checking `live` gets an IndexError instead of the sentence.
* **A live frame carries `component_placements`**, keyed by component name, each
  `{position_mm, rotation_xyzw}`. Not a flat 16-float matrix.
* **`put_asset` moves the project revision even though it is not a write**, so
  every asset goes in *before* the script that names them.
  `CadexdClient.put_asset` calls `refresh_revision()` for exactly this.
* **`build.py --arm X` prefers `replay/X`** over the two committed policies in
  `assets/`, because the set's copy is the one whose digest was checked on
  arrival and it travelled with the bundle `trained_task=` names. With no set
  and a script that names a `trained_task`, it refuses and prints the two
  commands that fix it.
* **`POLICY_CALL` is line-anchored on purpose.** These scripts keep every
  retired policy as a `#`-prefixed record — six of them above the live one — and
  an unanchored match finds the oldest.
* **`--project` overrides where a build lands.** A store built by one engine is
  refused by another at `open_project`, which is correct, and a second path is
  sometimes the shortest way forward.

## `harness capture` is still the cheap look

```bash
uv run python -m harness capture --dir jobs/stand13-20260805-135926 \
    --iteration 1800 --seeds 4 --tile
```

Two seconds of CPU, no second machine, no GUI. It renders `mjVIS_INERTIA`
boxes — the mass distribution — because `model-model.xml` carries 5 geoms and
would otherwise render as an empty floor. **The Shell shows the real CAD solids
and `capture` never will**, which is the whole difference between the two pages.
Neither replaces the other.
