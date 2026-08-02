# concept.md — what cdx-rl is

## The thesis

**A prompt should be able to become a robot.**

Not a rendering of a robot, and not a policy for a robot somebody else built.
The whole chain, with nothing hand-carried between the links:

```
  prompt
    → a parametric mechanism           (a Cadex script — the source of truth)
    → a dynamics model                 (MJCF, mass and inertia from the solids)
    → a task                           (observations, reward, termination,
                                        reset variation, disturbance)
    → a policy                         (PPO on MJX, on the 5090 in this box)
    → verification                     (played back, torque-checked, witnessed)
    → a manufacturable object          (STEP/STL out of the same script)
```

Cadex already has every link. What it does not have — deliberately, ADR-088
§6 — is the thing that *drives* them: the drivers, the measurement discipline,
and the record of what was tried and what it meant. That is this repository.

## What cdx-rl is

**A research repository and a Flywheel flywheel-graph.** Mechanisms get
designed here, policies get trained here, results get measured here, and the
evolution of the work is recorded as a DAG rather than as a directory of
folders named `stand8_final_v3`.

Three things live here and nowhere else:

1. **Generic drivers.** `rebuild`, `measure`, `feasibility`, `capability`,
   `compare` — the five ~100-line scripts `MUJOCO.md` §7 leans on, which
   ADR-088 §6 keeps out of the cadex repo on purpose and which **do not
   exist on this box at all**. cdx-rl owns them, generalized: parameterized
   by project rather than copied per project. Plus a **training supervisor**,
   which §7 assumes a human provides by watching.
2. **Experiments.** One directory each, with a question stated before
   dispatch and a metric decided before dispatch.
3. **The graph.** Every claim with its evidence attached, so a fresh agent
   recovers the state of the work from graph state alone.

## What cdx-rl is not

**Not a place to develop Cadex.** `/home/theo/cadex` is read-only from here:
no commits, no file edits, no branch changes, no builds. If cdx-rl finds it
wants something from Cadex, that want is written down in
[`cadex-wishlist.md`](cadex-wishlist.md) and acted on somewhere else, by
somebody with that repository open.

The boundary is not bureaucracy. Cadex has 106 ADRs of reasoning behind its
current shape, and several of the things cdx-rl will find inconvenient are
load-bearing decisions rather than oversights — the absent `train` verb most
of all. A research repository that starts patching its substrate stops being
able to say which of its results came from the substrate.

**Not a MuJoCo tutorial, and not a general RL library.** The scope is
Cadex-authored mechanisms. If the answer to a question is "use a standard
benchmark environment", that question belongs somewhere else.

**Not a place for GPU-hours without a stated question.** See
[`method.md`](method.md).

## Why now, and why this box

`sb1x` — RTX 5090 32 GB, Ryzen 9 9950X (32 threads), 60 GB RAM, 2.4 TB free,
Ubuntu 24.04 — is already the remote training target for Cadex work driven
from the laptop. What exploration established is that it can be *more* than
that: **the dynamics domain evaluates headlessly**, so a mechanism can be
authored, exported to MJCF, turned into a task, trained, and verified without
a display anywhere in the loop. Verified on 2026-08-02 with a 1-DOF pendulum
(see [`cadex.md`](cadex.md) §7).

That closes the loop on one machine. The macOS shell stays in the picture for
what it is uniquely good at — watching a policy move, and the Policy Outputs
panel, which is the only thing that catches hazard 15 in one glance — but it
is a promotion step for hero results, not a link in the chain.

## The first thing worth doing

`MUJOCO.md` §7 step 8 contains this sentence, about why a run must be
watched:

> *"the reward peaked at iteration 1200 of 2000 the last time and nobody
> could see it."*

The run sitting in `/home/theo/cadex-jobs/stand-task-20260802-200109` is a
sharper version of the same thing. From its own `progress.json`:

| | |
|---|---|
| best iteration | **598** |
| best reward/step | **0.337** |
| final iteration | 2499 |
| final reward/step | **0.146** |
| wall time | 14 050 s ≈ 3.9 GPU-hours |

`progress.json` emitted `best_iteration` and `best_reward_per_step` every
single iteration and nothing on this box read them. The checkpoints are all
still there, including `stand8.best.cxpolicy` written at 15:07, which is the
iteration-598 network.

The tempting conclusion — *76 % of that run was wasted* — is where this stops
being simple, and the reason is in the same file. The per-iteration series in
`train.log` carries **mean episode length** beside the reward:

| iteration | reward/step | mean episode (of 600) |
|---|---|---|
| **598** | **+0.3373** ← best reward | 277.7 |
| 1500 | +0.2161 | 407.6 |
| **1800** | +0.2422 | **468.1** ← longest |
| 2499 | +0.1461 | 370.7 |

**Reward fell while survival rose.** The machine was staying up roughly 70 %
longer at iteration 1800 than at the reward peak, while scoring lower per
step — plausibly paying a small posture penalty every step in exchange for
not falling over, which is the trade the metric we actually care about would
take every time.

So the first contribution is not "stop at the peak". It is **look at the
number that matters**, and the two cautions from `MUJOCO.md` say why:

* **The trainer's reward and survival can be anti-correlated** (ADR-099).
  In the M9 run, survival was 12/12 where the trainer reported its worst
  numbers and 0/12 exactly where it reported its best. So a reward-based stop
  is a *compute* optimisation, never a *selection* rule. Selection is by what
  a checkpoint **did when you played it** — that is `compare`'s job.
* **`<out>.best.cxpolicy` is best-by-reward, and early in a run it can be
  the untrained network**, which scores well by standing still before the
  disturbance distribution has bitten. Check its peak torque: a policy
  commanding 1–2 N·mm of 86 is not balancing, it is doing nothing.

A supervisor that reads its own progress file, records the series, and says
something is still the clearest first contribution this repository can make —
it just has a narrower mandate than it first appeared. **It stops the burn;
it does not choose the checkpoint.** See
[`harness/DESIGN.md`](harness/DESIGN.md) §6 and
[`experiments/001-stand-biped/`](experiments/001-stand-biped/), which exists
to settle exactly this.

## Success criteria

cdx-rl has done its job when:

1. **A trivial rig goes end to end unattended.** Script → MJCF → task →
   trainer → `.cxpolicy` → witness verification → local playback → a graph
   node with artifacts, on CPU, in minutes.
   (`experiments/000-loop-validation/`)
2. **A wasted run cannot happen the same way twice.** The supervisor sees the
   peak, and `compare` — not the reward curve — chooses the checkpoint.
3. **"The policy is bad" and "the task is out of range" are distinguishable
   before anyone concludes anything.** ADR-106 spent three runs learning
   this; `capability`'s sweep is how it becomes routine.
   (`experiments/001-stand-biped/`)
4. **A fresh agent can rebuild the picture from the graph.** Not from this
   conversation, not from a directory listing — from nodes, their artifacts,
   and the edges between them.
5. **A result that came out of here can be built.** The script is the source
   of truth; the same script that trained the policy exports the STEP.

## Where to read next

| | |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | Repo map and the read-this-first ordering |
| [`cadex.md`](cadex.md) | Cadex, verified, with the traps |
| [`method.md`](method.md) | The research protocol — read before any GPU time |
| [`flywheel.md`](flywheel.md) | The graph, as it actually is |
| [`cloud.md`](cloud.md) | Compute topology, and when to leave this box |
| [`harness/DESIGN.md`](harness/DESIGN.md) | The drivers, specified |
| [`cadex-wishlist.md`](cadex-wishlist.md) | Wants, captured rather than acted on |
