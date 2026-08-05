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

**Not a place to develop Cadex — but no longer forbidden to improve it.**
Until 2026-08-05 this section read "read-only, full stop": wants went into
[`cadex-wishlist.md`](cadex-wishlist.md) and were acted on somewhere else, by
somebody with that repository open. Nobody had that repository open, thirteen
entries accumulated, and three of them — the command range, the warm start,
and an engine too old to build `script.py` — became the binding constraint on
this research, ahead of the GPU. The policy was costing more than it bought.

What cdx-rl does now: **work in the PR clone at `/home/theo/cadex-prs` and
submit pull requests**, reviewed externally with the Cadex agents. Never push
to `theo-kirby/cadex`, and never touch the operator's tree at
`/home/theo/cadex`.

**The original argument was right about something, and it survives as a
discipline rather than a prohibition.** Cadex has 106 ADRs of reasoning behind
its current shape, and several things cdx-rl finds inconvenient are
load-bearing decisions rather than oversights — the absent `train` verb most
of all. A PR that "fixes" one of those is a misreading; read the ADR first.
And the deeper worry — *a repository that patches its substrate stops being
able to say which of its results came from the substrate* — is answered by
recording the substrate, not by freezing it:

* every run records the engine revision and the trainer digest
  (`runtime.json`, `smoke.py`), so a result is always attributable;
* the trainer must be **constant within a comparison**, not constant forever,
  and when it moves a **bridge run** measures the offset (`method.md` §8b);
* the MuJoCo pins do *not* move (invariant 2), because hazard 15 is a torque
  read off those dynamics and it is the measurement that decides
  buildability.

Freezing the substrate never actually made results attributable. Recording it
does, and it lets the blocking three get fixed.

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

## The thing that turned out to be worth doing

The section above is right and it is not where the result came from.

Experiments 001 and 002 did what they set out to do — the reward peak is not
the best checkpoint, in 2 of 3 seeds — and along the way they measured
something they were not looking for and could not act on. **Hazard 15
replicated 3 of 3**: with nothing pushing at all, every policy in the family
held a motor at 63–87 % of its rating. 001 wrote the consequence plainly —
*"this policy family does not describe a machine that can be built"* — and
`MUJOCO.md` hazard 16 explained why no reward term would fix it.

Both were correct, and both were about the wrong layer. Experiment 003
changed the **action space** — the policy emits a joint setpoint held by a PD
servo, instead of emitting torque.

> **This paragraph used to say the bracing "went to 31 % peak and 15.6 %
> static", and that was wrong.** Retracted 2026-08-04. Those are
> `feasibility.py` check 6 — *the model's own PD servo, one episode, zero
> action*, holding the nominal pose, with no trained network in it. 002's
> 63–87 % is a trained policy's commanded torque. A servo number and a policy
> number are not comparable, and the conclusion drawn from setting them side
> by side did not hold. Measured properly against the trained policies,
> **hazard 15 replicated 3 of 3 under the new action space too**, at 73–91 %
> mean of rating. `experiments/003-position-action-space/` carries the
> retraction; `mechanisms/mg-legs/drivers/hazard15.py` is the instrument that
> should have been used.

What actually happened is subtler, and experiment 004 is what isolated it.
The policy never commands torque under a position action space — but it
commands **position errors large enough that the servo saturates**. 003's own
gate says the PD saturates at 16.4° of error, and these policies command up
to 44° on a ±45° joint. The bracing moved from the network's output to the
servo's, where the harness's instrument could not see it.

**The layer that mattered was the command range, not the action space.**
Experiment 004 capped what the policy may ask for and measured the resting
duty above 90 % of rating: **51.8 % unclamped → 13.5 % at ±25° → 0.2 % at
±15°**, and at ±25° the cost in stepping is one the paired test cannot
distinguish from zero (15/24 against the control's 18/24, McNemar 4 : 1,
p = 0.375). 003 changed the right layer for a reason that turned out to be
wrong, and only 004 separated the two.

**And the cap buys time, not immunity** — experiment 005's CPU pre-flight
scored the bracing across each arm's whole checkpoint series and found duty
rising with training in *all three*, the clamp setting the rate and the
plateau rather than whether it happens. The unclamped arm saturates at 55 %
duty by iteration 1000; ±25° is still at 15 % at 1750, sitting at the exact
mean torque the unclamped arm occupied just before its own duty tripled.

The generalisable form, and the one worth carrying into the next mechanism:
**when a hazard replicates across every seed and no reward term touches it,
suspect the interface rather than the objective** — and then find out *which
part* of the interface, because the first plausible answer was not the right
one. Nine runs moved the disturbance five times and the reward three times
against a problem that was in neither; the tenth changed the action space and
got a real improvement for a reason it had not measured.

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

### Criterion 5, measured at last — and it mostly passes

The other four are being worked towards. Criterion 5 spent a week **failing
for a reason nobody could test**: the engine on the box with the GPU could not
build `script.py` at all, so "is `stand-b8` reproducible from source?" was an
open question dressed as a claim. On 2026-08-05 the engine got current and the
question got an answer.

**`mechanisms/mg-legs/script.py` rebuilds `tasks/stand-b8/` on sb1x, and the
entire discrepancy is one floating-point rounding artifact in a quantity that
is mathematically zero.**

```
harness rebuild --project projects/mg-legs-nopolicy.cadex \
    --script <script.py, policy output removed> --verify \
    --worker-cpu-seconds 3300 --worker-memory-mb 32768
# ok: true, verify.stable: true, 8.2 s
```

| | committed `tasks/stand-b8/` | rebuilt on sb1x |
|---|---|---|
| `model-model.xml` | 14179 B, `80eaa18f…` | 14179 B, `0fe04cfc…` |
| `stand-task.json` | 30213 B, `5572adf2…` | 30213 B, `0b4d160c…` |

Different digests, and the diff is **two lines total**. The task JSON's only
difference is the MJCF digest it embeds — a consequence, not an independent
change. The MJCF's only difference is the **pelvis** inertial `pos`
x-component: `5.10066e-11` m as built on the macOS laptop against
`5.10087e-11` m on sb1x, a gap of **2.1 × 10⁻¹⁵ m**. The machine is
symmetric, so that coordinate is *zero*; both values are noise around it, and
both sit 5 × 10⁻⁸ mm off the plane. Mass, quaternion and diagonal inertia are
bit-identical, and so is every other line of the file.

**So the honest status is:** reproducible in substance, not bit-identical
across platforms, and the one difference is physically meaningless. That is a
much better position than `stand-b2`, which remains a claim about committed
bytes because its authoring revision is genuinely lost.

**What it costs, and it is not nothing.** Cadex's digest chain is exact, so
that one ULP propagates: `script.py`'s own `assembly.policy` output is
*refused* on sb1x, because `stand10.cxpolicy` records the laptop's task digest
and the rebuild produces Linux's. The refusal is correct in principle — a
policy is only meaningful for the task it was trained on — and wrong here.
`cadex-wishlist.md` #15 has the analysis. The practical consequence for now:
**`tasks/` is the unit of provenance**, a policy travels with the bundle that
trained it, and mechanism work builds with the policy output removed.

Experiment 004's result — the ±25° command clamp — is the first policy this
project has produced that a bench could plausibly hold. It was also **not
reachable from `mechanisms/mg-legs/script.py`**, for two independent reasons.
As of 2026-08-05 one is fixed and the other is a PR:

* **`cadex-wishlist.md` #13 — the pinned engine could not build `script.py`
  at all. FIXED by getting current.** The mechanism was authored on the macOS
  laptop against a newer Cadex; the version gap meant the script was
  committed-and-readable here but not runnable here. The observation kinds it
  needs (`centre_of_mass_velocity`, `centroidal_angular_momentum`) landed
  upstream in `593f64e6` (ADR-116, 2026-08-03) and are in `origin/main`. The
  engine now driven from `/home/theo/cadex-prs` has them.
* **`cadex-wishlist.md` #12 — the clamp is a derived-bundle edit. OPEN, and
  it is the first PR.** Under a position action space an actuator's action
  range *is* its joint's limits; the task JSON derives one from the other, so
  there is no way to say "the policy may command ±25° of a ±45° joint" from
  the script. `stand-b8-clamp25` was produced by editing the derived bundle,
  which means the artifact defining the winning policy is downstream of the
  source of truth rather than generated by it. `cadex-engine-plan.md` §2 is
  the spec.

The consequence is concrete rather than theoretical: **the next mechanism
experiment is foot geometry, and it was blocked on the engine, not on the
GPU.** `script.py` §1232–1276 rules out the obvious alternative to
restricting the policy — sizing the motor for the ~230 N·mm the policy wants.
The centre of pressure cannot leave the sole, which reaches 45.5 mm ahead of
the ankle, so **past 2.581 N × 45.5 mm = 117 N·mm the foot rolls instead of
pushing** (ADR-082); 86 N·mm was chosen partly to keep one ankle below that,
because at MG90S stall a single ankle out-torques the footprint by 1.8× and
**the machine could tip itself**. The torque budget is not raisable without a
bigger foot.

**And the foot is a script *parameter*, not a code edit** — measured
2026-08-05, and it makes the experiment much cheaper than this section used to
imply:

```python
foot_len=num(70, unit="mm", min=55, max=110, label="Foot length")   # script.py:60
foot_w=num(40,  unit="mm", min=32, max=55,  label="Foot width")     # script.py:61
```

The 45.5 mm sole reach is `0.65 * p.foot_len` (`script.py:486`, `:1147`). At
`foot_len=110` — the slider's own maximum, no geometry code touched — that
becomes **2.581 N × 71.5 mm = 184.5 N·mm**, a **58 % larger torque budget**.
Against clamp25's measured ~46 N·mm of resting bracing that is a lot of room.

Two caveats to carry into that experiment. The 45.5 and 24.5 mm figures are
**hard-coded in nine comment blocks** holding the reward and capture-point
arithmetic (`script.py:1711, 2176, 2247, 2259, 2304, 2327, 2378, 2423, 2426`);
moving `foot_len` silently invalidates every one, so they get fixed in the
same commit or the next agent reads a stale number as ground truth. And
`foot_h = 8.0` (`:136`) and the 0.35/0.65 heel/toe split are hard-coded
constants that stay put. Mass, inertia and the support polygon all move
together, so this is **a genuinely different machine** and nothing from 003 or
004 transfers as a baseline.

Worth stating plainly alongside the 004 result: holding the stance costs
**2.39 N·mm** and the hand-written PD peaked at 4.5. Clamp25's settled mean of
53.8 % of 86 N·mm ≈ 46 N·mm is still **~19× what standing actually costs**.
"Buildable" is a real improvement over 001's 71 %; it is not a clean bill of
health.

`cadex-engine-plan.md` is where these are scoped for hand-off.

## Where to read next

| | |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | Repo map and the read-this-first ordering |
| [`cadex.md`](cadex.md) | Cadex, verified, with the traps |
| [`method.md`](method.md) | The research protocol — read before any GPU time |
| [`flywheel.md`](flywheel.md) | The graph, as it actually is |
| [`cloud.md`](cloud.md) | Compute topology, and when to leave this box |
| [`harness/DESIGN.md`](harness/DESIGN.md) | The drivers, specified |
| [`cadex-wishlist.md`](cadex-wishlist.md) | Thirteen things wanted from Cadex, and what each one cost. The record; status is `open` / `PR #N` / `merged` / `worked around` / `withdrawn` |
| [`cadex-engine-plan.md`](cadex-engine-plan.md) | The three that block research, scoped as PR specs |
