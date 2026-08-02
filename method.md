# method.md — the research protocol

This is cdx-rl's adaptation of `/home/theo/cadex/docs/MUJOCO.md` §7, with the
measurement discipline made explicit. **Read this before spending GPU time.**

`MUJOCO.md` §7 is written for a person driving three machines through the
same order by hand. What follows is the same order with the parts a person
was supplying — the watching, the deciding, the remembering — assigned to
something that will still be doing them at 3 a.m.

Its purpose is to keep experiments honest. Every rule below exists because
something already went wrong: the ADR number is the receipt.

---

## The rule the rest of the file serves

> **Decide the success metric before you dispatch.** (ADR-097)

Not after. Not "we'll see how the curve looks." The moment a task declares
`reset_variation` or `disturbance`, the reward curve gets noisier and stops
being comparable with the undisturbed run — so a metric chosen afterwards is
a metric chosen by looking at the answer.

The metric for anything whose word for success is *balance*, *hold* or
*stand* is **recovery rate**: episodes surviving a shove over episodes
shoved, **split by shove azimuth**. Aggregate survival averages a mechanism
limit together with a learning result and reports neither.

---

## The order

Steps 1–7 are cheap. Step 8 costs hours of GPU. Everything before it exists
to make sure step 8 is asking a question worth the money.

### 1. Check what is actually there

```bash
grep -c "assembly\." script.py
```

A parametric model with pose sliders is not a mechanism. The "joints" may be
`part.transform` calls that rotate solids at build time. If the count is
zero, authoring the dynamics layer is the large half of the job and the RL
loop is the small half.

### 2. Pose the joint frames, not the components

Give **both connectors of a joint the identical posed world frame**. The
tempting design — neutral solids, pose in `assembly.component(placement=…)` —
does not survive the native solver: an island the joints never reach from
ground has six free degrees of freedom and the solver answers with its own
member of the solution family. Measured: it zeroed all eight joints and
displaced a biped by (90, 18, 58) mm and 40°.

### 3. Give every part its own component frame, at its limb's middle

Not the origin, not the proximal joint. Both are fields of dust, and the MJCF
drift check refuses them at exactly 1.0 (hazard 12).

### 4. Measure before sizing anything

`measure` reads `model_evidence.inertials`, so mass and inertia come from
OCCT — not from a tessellation and not from a guess. Total mass, standing
centre of mass, each joint's height, and the mass hung below it. Every number
steps 5 and 6 use comes from here, **measured at the exported keyframe** and
not read off the drawing.

### 5. Choose the actuator honestly, and say which question you are answering

Torque motors rather than position servos, so that zero action is collapse
and there is no degenerate "hold the setpoint" solution (ADR-092).

Then decide whether the limit models **the hardware** or **the mechanism**.
An MG90S stalls at 216 N·mm; a mechanism-derived limit for the same biped was
750. A policy trained on the second will command torque the bench cannot
produce. Both are defensible; **only one is what you will build**, and the
experiment README must say which.

### 6. Declare what changes between episodes

`assembly.reset_variation` starts the episode tilted, lifted and moving.
`assembly.disturbance` pushes it while it runs. **Without both, "balance" is
not the task and bracing wins.**

> **Hazard 16, and it is why hazard 15 happened.** Before M9 every episode
> reset to the identical keyframe with every velocity zero. A posture found
> once was never asked a second question, and pinning four motors to hold a
> wide splay is a *stable* answer as well as a cheap one — effort was
> weighted −0.0002/N·mm, costing 0.17 against a +0.39 reward/step, while
> falling costs the alive bonus, the tilt penalty and the rest of the
> episode. **A reward term cannot fix this.** The problem is not that bracing
> is under-priced; it is that the task never tests the difference.

**Never perturb joint angles** (hazard 17). The reset pose is the *solved*
configuration with the soles exactly on the floor; a ±3° knee jitter moves a
foot ~5 mm *through* the floor and MuJoCo resolves that as an impulse nothing
could stand up to. Perturb the free root **rigidly** — tilt, lift, spin — and
perturb velocities. A rigid tilt cannot change the mechanism's shape, so it
cannot self-interpenetrate however far it leans.

And do not do the clearance arithmetic by hand: `mg-legs` was written with a
3 mm lift for a 6° tilt on the reasoning that 6° across a ±30 mm stance is
about 3 mm at the sole. The measured answer was **5.13 mm**, because a tilt
pivots about the base frame's origin and the far thing from a pelvis is a
toe, diagonally, most of a leg away.

### 6b. Size the shove before you declare it

The arithmetic that decides what the run is even asking (ADR-100):

```
ω₀ = √(g / h)          h = CoM height
Δv = F · t / m         the impulse over the mass
ξ  = Δv / ω₀           the capture point
```

Read ξ against two distances, both measured off the export:

| ξ vs | means | what the policy must learn |
|---|---|---|
| inside the support polygon | in-place recovery | ankle and hip torque; feet never move |
| polygon … polygon + `leg·sin(swing)` | one step | pick a foot up, place it, catch, return |
| beyond that | nothing | falls are a **mechanism** limit, not a learning failure |

**Sizing the ceiling wrong in either direction wastes the run.** Too small
and nothing is asked of the legs (M9 asked for ξ = 19.5 mm and got a policy
that never moved its feet). Too large and the falls are arithmetic — which is
exactly what three `mg-legs` runs turned out to be (ADR-106).

Declaring a range that spans the band is a **curriculum inside the
distribution** and needs no scheduling feature: `newtons=[0.15, 0.90]` puts
the in-place problem and the edge-of-polygon problem in the same batch from
iteration one.

A **sustained** force is not an impulse. It is a steady lean that offsets the
CoP by `F·h/W` and so *shrinks* the polygon ξ has to land in. Subtract it; do
not add it to the shove.

Two mechanism facts fall out of the arithmetic rather than out of training,
and both are worth checking before dispatch: whether the support polygon is
asymmetric front-to-back (`mg-legs` is — 45.5 mm forward, 24.5 back, because
the toe reaches and the heel does not), and whether the machine has any
lateral authority at all. Without ankle roll or hip yaw, sideways is hip_roll
plus a weight shift and the effective polygon is well under the geometric
half-width.

### 7. Run the gate, and read what it says rather than whether it is green

`feasibility` is six checks and none of them learn anything:

1. static arithmetic (**advisory** since ADR-099),
2. exact gravity compensation by `mj_inverse`,
3. whether the mechanism can reject the **worst declared shove** in place,
4. contact sanity,
5. a drop test that must **fall**,
6. a hand-written PD that must **hold**.

If a PD can stand it, PPO can. If the gate is red, find out *which check* and
*why*: hazard 14 is the gate being wrong, hazard 18 is the gate measuring
nothing, hazard 10 is the gate being right and the machine being unable to do
the task.

> **Hazard 18 is the dangerous one, and it happened three times in one
> afternoon.** `mj_inverse` with an external force applied returns leg
> torques **bit-identical** to the undisturbed case — inverse dynamics on a
> floating base solves for the force needed at *every* dof including the six
> unactuated ones, so the push is absorbed by the free joint's own residual.
> A green light from a check that computes nothing is worse than no check.
> **Every check must be able to fail**, and the way to know it can is to make
> it fail once on purpose.

### 8. Dispatch, and supervise

Detached, always: a run is over an hour and one ssh held open that long is a
closed laptop away from a lost run. Do not pipe the trainer's output through
`tail` (ADR-093 §4).

Pass `--checkpoint-every 100`. Each one is a complete `.cxpolicy` you can
pull mid-run and play, and `<out>.best` tracks the best so far. Each costs
about one iteration.

**The trainer proves its own witness before writing each file and prints the
margin. If that margin is under 100×, stop and read hazard 13** rather than
continuing.

**8b. Check the box's checkout first** (ADR-104). `remote_train.sh` runs the
*box's own* copy of `training/cadex_train.py`, so a trainer that predates a
surface addition silently ignores the new fields while recording the new
algorithm string in the policy header, and nothing fails loudly.

```bash
git -C /home/theo/cadex log --oneline -1
# 06d1374b Refuse to dispatch to a box running a different trainer (ADR-104)
```

`tools/smoke.py` records this on every run. It is not optional after any
change to `EPISODE_VARIATION_ALGORITHM`.

### 9. Ask what the task is asking, not just how the run went

`capability` sweeps a scale factor over the task's declared shove magnitudes
and prints survival at each, **split by azimuth**, with the termination mix
and how far into its own disturbance schedule each death got.

> **A run reading 0/12 at the declared band and 11/12 at a fifth of it has
> not failed to learn.** It was asked something out of range. That is
> ADR-106, and it is what three runs of `mg-legs` turned out to be:

| what it was asked | stood | steps of 600 |
|---|---|---|
| no shove, no reset variation | 12/12 | 600 |
| reset variation only | 11/12 | 556 |
| ×0.15 | 11/12 | 556 |
| ×0.30 | 8/12 | 469 |
| ×0.50 | 1/12 | 213 |
| **×1.00 — what the task declared** | **0/12** | **151** |

Two readings that only exist because the sweep prints them: it died on the
**first** shove every time, so the second disturbance window had never once
been exercised — half the episode's design had never run. And it died by
`collapsed`, not `tipped` — 8 of 12, upright and sinking, killed mid-squat,
which is exactly the state a recovery passes through.

**A curve that is flat across the whole sweep is a curve that measured
nothing**, and `capability` must say so out loud. A 50-iteration sanity run
reads 0/12 everywhere: it has learned nothing yet, and that is not a finding.

**Do not declare failure without this sweep.** It is the difference between
"the policy is bad" and "the task is out of range", and those two have
opposite next steps.

### 10. Compare the checkpoints before choosing one

`compare` plays every `.cxpolicy` in a directory locally against several
seeds — stock MuJoCo, no GPU, seconds — and prints survival, episode length,
final tilt, drift and **peak/mean torque per motor** as one table.

Three things it must get right, each learned by being caught:

* **Play a run against its own bundle, not the newest one.** A rebuild is
  keyed by script digest and replaces `script_artifacts/`, so a finished
  run's task can vanish locally while its checkpoints sit beside you. The
  bundle copy in the run directory is the one that survives.
* **Give every episode its own model** (ADR-103 §9). `evaluate_episode`
  multiplies the task's domain randomisation **in place** into the model it
  is handed and keeps no baseline, so an evaluator that loops episodes over
  one loaded model compounds every draw it has ever made and its last row is
  a different machine from its first. Reload per episode — four milliseconds
  against a two-second episode. **Play the same file twice and check the row
  is the same.** That two-second test is what found this.
* **Print the termination mix.** `compare.summarise` had collected it since
  M9 and `main` never printed it. That omission cost three runs.

Watching two policies animate at once is not available and should not be
faked: ADR-077 is exactly one simulation per script and the shell has one
timeline. The numbers compare side by side; the animations do not.

### 11. Bring it home through `put_asset`

The digest is **required and never inferred**: `assembly.policy` names a
policy by file *and* SHA-256, because VISION principle 3 says any state that
cannot be rebuilt from the script is a bug — and hours of stochastic GPU
compute genuinely cannot be. On rebuild the worker re-checks the bundle
digest, the model it references, the observation channels in order, the
action table, and re-evaluates the trainer's witness with its own float64
forward pass.

### 12. Report what the rollout does, rather than iterating on it

ADR-088's stopping rule. The trace is the evidence: frame count against the
episode length says whether it terminated early; pelvis height, tilt and
drift over the episode say what "it stands" actually meant.

### 13. Open the Policy Outputs panel before you believe any of it

ADR-096. It draws each actuator's command against its own limit. The
trajectory says what the mechanism *did*; this says what the policy
*decided*, and the two can disagree in a way only this shows. **A bar pinned
at an end is the finding.**

This is the one step that needs the macOS shell. Until `compare`'s torque
columns are trusted, it is the promotion gate for a hero result.

---

## The measurement discipline, stated separately

Because these are the four things most likely to be skipped.

### Peak versus final

Record both, always. From
`/home/theo/cadex-jobs/stand-task-20260802-200109/progress.json`:

```
best_iteration        598
best_reward_per_step  0.337
iteration             2499
reward_per_step       0.146      (final)
wall_time_s           14050      (≈ 3.9 GPU-hours)
```

**But do not stop at "76 % of that run was spent regressing."** The
per-iteration series in `train.log` — which is where the *curve* lives;
`progress.json` is rewritten each iteration and keeps only the current point
plus a checkpoint list — carries mean episode length beside the reward:

| iteration | reward/step | mean episode (of 600) |
|---|---|---|
| **598** | **+0.3373** ← best reward | 277.7 |
| 1500 | +0.2161 | 407.6 |
| **1800** | +0.2422 | **468.1** ← longest |
| 2499 | +0.1461 | 370.7 |

**Reward fell while survival rose** — the mirror image of hazard 19, and
exactly the disagreement ADR-099 is about. So:

* **Record both series, always.** A peak in one is not a peak in the other.
* **The supervisor's job is to notice and to stop the burn.** It is *not* to
  install `<out>.best`, and it is not to declare a winner.
* Which checkpoint is actually best is answered by playing them. See
  `experiments/001-stand-biped/`.

### The reward curve is not the result (ADR-099)

The M9 run, played through the engine's reference runner over 12 seeds
against **the bundle it was actually trained on**:

| iteration | trainer's reward/step | survived | steps of 600 |
|---|---|---|---|
| 500 | +0.034 | **12/12** | 600 |
| 900 | −0.050 | **12/12** | 600 |
| 1500 | +0.45 | 3/12 | 250 |
| 2000 (`best`) | **+0.5118** | **0/12** | **43** |

Anti-correlated across the whole run. The checkpoint the trainer labels
`best` — the one an unexamined pipeline installs — falls in 43 steps of 600,
from every seed and every direction, before the first shove window opens.

> **Judge a checkpoint by what it did when you played it, never by the number
> the trainer printed. Install by survival.**

Why they disagree is an open question with several candidates (MJX versus
MuJoCo dynamics, sampled versus mean action, the auto-reset batch mean not
being an episode return). One has since been eliminated and a fourth found
(ADR-101): the trainer read the bundle's episode length and never used it, so
an environment that never fell over never reset. **Every number in the table
above was measured against an unbounded episode and is not a baseline for
anything measured after that fix.** The operational rule is unchanged.

Read **mean episode length** beside the reward, every time — it is on the
stderr line, in `progress.json` as `episode_steps`, in the policy header's
curve rows and in the shell's Training panel. *A reward climbing while
episode length falls is hazard 19 happening in front of you.* M9b's fell
170 → 30 over 400 iterations with nothing recording it.

Also: **`<out>.best.cxpolicy` is best-by-reward, and early in a run it can be
the untrained network**, which scores well by standing still before the
disturbance distribution has bitten. Check its peak torque — a policy
commanding 1–2 N·mm of 86 is not balancing, it is doing nothing.

### Torque saturation — the bracing hazard (15 and 16)

The `mg-legs` standing policy plays as a clean stand and *is* one: it holds
the full 6 s, the curve is healthy, the engine verified it. It is also
holding `hip_pitch_l`, `hip_pitch_r` and `knee_r` above **95 % of the MG90S
limit on 100 % of frames** — a mean of 212–214 N·mm against a 216 N·mm bound
— while both ankles sit under 72. It braces rather than balances: the stance
widens from ±30.00 mm to ±37.2/37.4 and the right foot pulls 13 mm back, held
by torque.

216 N·mm is a **stall** rating, a momentary number. No real servo holds 98 %
of it for six seconds.

Nothing in the poses shows this. So:

* **`compare` reports peak and mean torque per motor, and % of frames above
  90 % of limit, on every row.** Not optional, not a `--verbose` flag.
* **Treat "the reward went up" and "the mechanism is doing something a
  machine could do" as two separate claims**, and check the second before
  spending GPU time on a harder version of the first.
* A policy pinned at its actuator limits has **no authority left for a
  disturbance**, so this also predicts the outcome of the first push.

The calibration table — what a genuinely good result looks like, `mg-legs`
(ADR-095), 263 g of PLA and eight MG90S, 2000 iterations at 4096
environments, 1 h 16 m on an RTX 5090:

| | |
|---|---|
| reward/step | −1.76 → +0.391 (peak +0.445 at 1200) |
| episode | 151 frames of 151 — never terminated |
| pelvis height | 284.00 → 283.60 mm, worst drop 0.84 mm |
| tilt | settles ~5.5° against a 45° termination |
| drift | 6.97 mm over 6 s |
| witness | 1.009e-07, 991× inside tolerance |
| **actuator duty** | **3 of 8 motors above 95 % of stall on 100 % of frames** |

Every number above the last row says the run went well, and they are all
true. Calibrate against the whole table, not the top of it. And the
comparison that makes it mean anything is the gate's own drop test: **zero
torque falls at 0.96 s.** A machine that stands for six seconds is balancing,
not merely stable.

### Range before failure (ADR-106)

Before writing "the policy failed to learn X" anywhere — a commit message, a
graph node, a sentence to a human — run `capability`'s sweep. If survival is
high at ×0.15 and zero at ×1.00, the finding is about **the task**, and the
next action is to re-size it, not to train longer.

---

## What a cdx-rl experiment must contain

Every directory under `experiments/` carries a `README.md` that states, in
this order, **before any dispatch**:

1. **The question**, in one sentence, phrased so that both answers are
   interesting.
2. **The metric**, named and defined, and *why that one*.
3. **The mechanism**: which script, which digest, what the actuator limit
   models — hardware or mechanism.
4. **The task**: episode length, control rate, reward terms with weights,
   terminations, reset variation, disturbance band, **and the capture-point
   arithmetic that sized it**.
5. **The gate**: `feasibility`'s six checks, with what each said.
6. **The budget**: iterations, environments, expected wall time, and the
   stopping rule.
7. **Pass criteria**, written before the run.

Afterwards, and separately, so it is visibly separate:

8. **What happened**: peak and final, `capability`'s sweep, `compare`'s
   table, the termination mix, the torque columns.
9. **What it means**, and what it does not mean.

The graph node mirrors 1–9. The artifacts — `progress.json`, the chosen
`.cxpolicy`, the sweep table — attach to it. See [`flywheel.md`](flywheel.md).
