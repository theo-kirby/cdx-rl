# `stand-b8` — the standing task with a **position** action space

```
stand-task.json           5572adf265aa51cb4cfea2c454695c42a4f1c2402a6788bc696c0129f14fc595
model-model.xml           80eaa18f6025d589796315fcad45bb70bf72e55da750864f60b5e0b3cc71fdb3
stand10.001150.cxpolicy   dc375a4255efedc00ddbb0ccacf6c899a38f1c08e71c65a0cd17ddfd7bb74947
```

295 KB. This is the bundle, the model and the winning policy of
[`experiments/003-position-action-space`](../../experiments/003-position-action-space/),
and unlike [`stand-b2`](../stand-b2/) **all three are reproducible from
source**: [`mechanisms/mg-legs/script.py`](../../mechanisms/mg-legs/script.py)
at accepted revision `feb5a884…` is what exported them.

## The trio is self-consistent, and that was checked rather than assumed

```
$ uv run python tools/cxpolicy.py --policy tasks/stand-b8/stand10.001150.cxpolicy
model       80eaa18f6025d589…     ← matches model-model.xml
task        5572adf265aa51cb…     ← matches stand-task.json
trainer     aacfa82318e4e239…     ← the digest tasks/stand-b2 pins
```

A policy played against the wrong MJCF still verifies and still produces
numbers, so this is the check that says these bytes are the ones that were
trained. The trainer digest is the same `aacfa823…` experiment 002 pinned,
which is why 003's numbers and 002's are comparable in the ways that matter:
different task, **same update rule**.

## What is different from `stand-b2`

| | `stand-b2` | `stand-b8` |
|---|---|---|
| action space | `motor` — the network output **is** joint torque | **`position`** — a PD servo in the solver; the output is a setpoint |
| zero action | motors off; **falls at 0.976 s** | holds the nominal pose; **stands** |
| nominal pose | straight-legged | **crouch**: hip 15°, knee 30°, ankle 15° dorsiflexion |
| joint limits | asymmetric (knee `[-5, 130]`) | **symmetric about the nominal pose** |
| reward | `alive` +1.0 minus 11–13 costs | **9 positive kernels** `w·exp(−(e/σ)²)`, total 5.3 |
| per-step reward | can be negative → terminating can pay | **bounded in [0.2, 5.3]** — cannot be negative |
| control rate | 100 Hz, 600 steps | **50 Hz**, 300 steps, ten solver substeps |
| shove band | 0.30–0.80 N | **unchanged** — that is the point |

The band, both shove windows, the reset variation and the wind are B6's
exactly, so the two tasks ask the *same question* of a differently-actuated
machine. That is what makes "17/24 against 6/12" a comparison rather than
two unrelated numbers.

## The action table is the whole change, and it is readable

```
hip_roll_l   [-30, +30] deg   zero action → +0
hip_pitch_l  [-45, +45] deg   zero action → +0
knee_l       [-30, +30] deg   zero action → +0
ankle_l      [-25, +25] deg   zero action → +0
ankle_roll_l [-20, +20] deg   zero action → +0
```

Every range is symmetric, so the trainer's `output_bias = (high + low) / 2`
is zero, so a network output of zero commands the nominal pose. In the MJCF:

```xml
<general name="knee_l/position" joint="knee_l" forcerange="-0.086 0.086"
         biastype="affine" gainprm="0.3" biasprm="0 -0.3 -0.01"/>
```

`gainprm 0.3` and `biasprm[1..2] = −0.3, −0.01` are kp = 0.3 N·m/rad and
kd = 0.01 N·m·s/rad — measured by the gate's own sweep at the crouch, and the
**softest** pair that stands. `forcerange ±0.086` N·m is the unchanged 86 N·mm
MG90S judgment: what changed is who computes the torque, not how much of it
there is.

## Using it

```bash
uv run python tools/train.py --bundle tasks/stand-b8/stand-task.json --label NAME …
uv run python -m harness feasibility --task tasks/stand-b8/stand-task.json --profile mg-legs
uv run python -m harness steps --policy tasks/stand-b8/stand10.001150.cxpolicy \
                               --task tasks/stand-b8/stand-task.json --seeds 24
```

**`stand10.001150.cxpolicy` is committed because it is an input, not just an
output.** B9 is the warm-start curriculum — walk the shove band up from 0.8 N
across short runs each initialised from the last — and this is the network it
starts from. 245 KB against a repository rule about *large* binaries.

The rest of the B8 checkpoints (24 of them, 4.6 MB) are in
`jobs/imported/b8/` on the laptop, gitignored. `results/` in the experiment
directory has the full sweep over all of them.
