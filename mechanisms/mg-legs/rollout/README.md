# mg-legs rollout — the best policy, ready to open in Cadex

A duplicate of `mechanisms/mg-legs/` set up for one job: play the best policy
this repository has produced and hand the result to the **Shell** for visual
verification. `cadex.md` §2 is the division of labour — cdx-rl drives the
engine and the trainer, and sends hero results to the Shell.

## On the Mac

```bash
cd ~/cdx-rl && git pull
set -a; . ./config/env; set +a
uv run python mechanisms/mg-legs/rollout/build.py
```

Then open `projects/mg-legs-rollout.cadex` in the Shell.

## It must be built on the Mac, and that is not a preference

**The engines are not interchangeable, and only the Mac's is new enough.**
`script.py` declares observation kinds that the older checkouts refuse by
name:

| box | Cadex | builds this script? |
|---|---|---|
| `sb1x` | `06d1374b` (the pinned one) | ❌ *"invalid kind … Received `centre_of_mass_velocity`"* |
| `sb9x` | `ae8da6a6` | ❌ gets one kind further, then *"Received `centroidal_angular_momentum`"* |
| `mmini` (macOS) | `560935bd` | ✅ where B8 was authored |

mg-legs was authored on the laptop, and the script has moved ahead of both
training boxes. This is worth knowing beyond this directory: **the pinned
`06d1374b` cannot rebuild the mechanism that experiments 003 was run on.**
`tasks/stand-b8/` carries the built MJCF and bundle, so training and scoring
never needed the authoring engine — but anything that goes back through
`write_script` does.

Do not "fix" this by pulling `/home/theo/cadex` on a training box. Invariant 1
forbids it, and the pin is what every recorded run's provenance means.

## What it plays

`assets/stand10.001700.cxpolicy` — experiment 003, **seed 2, iteration 1700**.

```
sha256  c9ca1195713a1abd9fbc71a7ea784c50d95d063edc0774685e68819ea34f3de2
task    5572adf265aa51cb…      model 80eaa18f6025d589…
```

Scored at 24 evaluation seeds, against the previous best from every seed:

| policy | surv | step | **BOTH** | mean episode |
|---|---|---|---|---|
| **s2 001700** | **20**/24 | 21/24 | **18**/24 | **264.4** |
| s0 001150 (003's published pick) | 17/24 | 21/24 | 17/24 | 239.8 |
| s1 001750 | 18/24 | 22/24 | 16/24 | 251.1 |
| s0 001050 | 16/24 | 20/24 | 14/24 | 239.2 |

**It is not a proven winner and the table must not be read as one.** McNemar
over the discordant seeds cannot separate it from `001150` (5 discordant,
p = 1.000) or from `001750` (p = 0.625). What it has is every point estimate
and the continuous measure — mean episode length 264.4 of a 300 cap, which
carries far more resolution than a 24-episode binomial. Best available,
statistically tied.

## Why evaluation seed 4

The rollout is `seed=4`, and it is chosen rather than default: **10 steps,
longest 121.8 mm, highest lift 36.5 mm, and it survives all 300 control
steps.** It is the episode worth watching.

Six of this policy's 24 scored seeds do not survive. A failure is also worth
watching and is one edit away — change `seed=` in the `assembly.rollout(...)`
call at the bottom of `script.py`. Seeds that fail: play the scored JSON in
`experiments/003-position-action-space/results/three-seed-tiebreak-24.json`
and look for `survived: false`.

## What is different from `../script.py`

Two lines, both in the live `assembly.policy` / `assembly.rollout` pair at the
bottom. The weights name and digest point at seed 2's checkpoint instead of
B8 seed 0's final network, and the rollout seed is 4 instead of 7. Everything
above — the mechanism, the reward, the task — is byte-identical, which is what
keeps the task digest the engine checks against the policy unchanged.

The commented-out policy history above the live call is kept, as it is in the
original: this project records every retired policy rather than deleting it.
Note that both `build.py` and `drivers/install_checkpoint.py` anchor their
regex to the start of a line for exactly that reason — an unanchored
`sha256="…"` matches the oldest commented record, which is a real bug this
directory hit while being written.

## Switching to a different checkpoint

Do not hand-edit `script.py` in a built project — the store binds an accepted
digest to that file and every later entry point fails with *"the restore pass
digest does not match the accepted digest"*. Use
`../drivers/install_checkpoint.py`, which copies the asset, rewrites the one
live policy call with a digest computed from the bytes, re-runs the accept
path, and rebuilds twice to prove the digest is reproducible.
