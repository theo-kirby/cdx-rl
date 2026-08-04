# 003 — the action space, the reward sign, and the control rate

**An imported experiment.** It was designed and run on 2026-08-03 in
`~/cdx-mjc` on the macOS laptop, before this repository was its home, and
dispatched to `sb1x`. Sections 1–7 below are the plan as it was written
*before* dispatch — [`plan.md`](plan.md) is that document verbatim and
unedited, and [`research-note.md`](research-note.md) is the survey that
motivated it, also verbatim. Sections 8 and 9 were written after.

Everything in 8 and 9 has been **re-measured through this repository's own
`harness steps`**, against the committed bundle, and reproduces the original
numbers exactly. Where the re-measurement corrected something, it says so.

---

## 1. Question

Nine runs (m9a/b/c, a1c, B1–B7) moved the *disturbance* five times and the
*reward* three times, and never once moved the action space, the control
rate, or the sign convention of the reward. Those are the three things the
legged-RL literature is most consistent about.

> **Does a machine whose zero action *stands* learn push recovery, where nine
> runs of a machine whose zero action *falls over* did not?**

Both answers are interesting. If it does, hazard 15 — this project's
most-replicated finding — was an artefact of the action space rather than a
property of the mechanism. If it does not, the action space is exonerated and
the problem is somewhere nobody has looked.

## 2. Metric

**The conjunction: stepped >10 mm AND survived 300/300 control steps**, at 12
seeds, across every checkpoint.

Chosen before dispatch and stated in `plan.md` §5. Neither marginal will do,
and this is measured rather than argued — §8's table shows the two moving
independently, with checkpoint 500 stepping in 7 of 12 episodes while
surviving 3. Scored on stepping alone, a machine that throws a foot out and
falls over wins. Scored on survival alone, the metric is the one six previous
runs already maximised **without ever landing a step**, which is how this
project spent five runs producing a machine that absorbs a shove with its
joints and never moves its feet.

A *step* is a foot continuously out of contact for ≥30 ms that lands ≥10 mm
from where it left, read off MuJoCo's contact list rather than off a height
threshold (ADR-107).

**Baseline to beat: B6's 6/12 on the same criterion and the same task.**

## 3. Mechanism

[`mechanisms/mg-legs/script.py`](../../mechanisms/mg-legs/script.py), accepted
revision `feb5a884…`, sha256 `56bba536…`. 302.0 g, twelve links, ten revolute
joints, floating pelvis.

**The actuator limit is 86 N·mm and it models the HARDWARE** — the MG90S
continuous-duty judgment, ~40 % of its 216 N·mm stall rating. Unchanged from
every previous run: what B8 changes is *who computes the torque*, not how
much is available.

The nominal pose is a **crouch** — hip 15°, knee 30°, ankle 15° dorsiflexion —
and the joint limits are symmetric about it. Both are consequences of the
action space, not independent choices; see §9.

## 4. Task

[`tasks/stand-b8/`](../../tasks/stand-b8/), bundle sha256 `5572adf2…`, model
`80eaa18f…`.

| | |
|---|---|
| episode | 6.0 s at **50 Hz** = 300 control steps, ten solver substeps each |
| actions | **10 position servos**, kp 0.3 N·m/rad, kd 0.01 N·m·s/rad, ±86 N·mm |
| observations | 58 channels, unchanged from B7 |
| reward | **9 positive kernels** `w·exp(−(e/σ)²)`, total 5.3 |
| terminations | `tipped` above 0.15 on qx²+qy²; `collapsed` below 0.5·Z0 = 70.5 mm |
| reset variation | tilt 0–15°, lift 15–45 mm, spin ±90 °/s, velocity 0–250 mm/s |
| disturbance | two shoves **0.30–0.80 N** for 0.12 s at 0.3–1.5 s and 1.8–3.6 s; 0.06 N sustained wind |

**The task is B6's exactly** — band, both windows, reset, wind. With four
structural changes at once the task had to be the one a baseline is held on,
or the run answers nothing.

**Capture-point arithmetic** (ADR-100), from the gate's own output:

```
ω₀ = √(9810 / 140.944) = 8.34 rad/s
worst instant  0.060 N sustained + 0.800 N for 0.12 s
Δv = 0.096 N·s / 0.302 kg = 0.318 m/s
ξ  = 38.1 mm
```

against a polygon reaching 45.5 mm forward and 24.5 mm back. So ξ is **inside
the polygon forward and outside it backward** — the shove is aimed backward
at [210, 330]° precisely so that it asks for a step — and a 45° hip swing
places a foot 164 mm, giving 4.2–5.4× margin. In range, and asking for the
behaviour being measured.

The reward's nine terms, and the three σ that were **measured** rather than
chosen (`swirl_scale.py`, at the crouch, over the recovery regime):

| term | w | error | σ |
|---|---|---|---|
| `alive` | 0.2 | — | constant |
| `upright` | 1.0 | `pel_qx²+pel_qy²` | 0.02 (already quadratic) |
| `capture` | 1.5 | ξ | 50 mm |
| `over_feet` | 1.0 | CoM vs foot centroid | 40 mm |
| `height` | 0.5 | `com_z − Z0` | 30 mm |
| `arrest` | 0.3 | \|cv_xy\| | 300 mm/s (chosen; sweep said 356) |
| `swirl` | 0.3 | \|cam_xy\| | **14.34 N·mm·s** measured |
| `posture` | 0.3 | Σ\|joint\| | **35.10°** measured |
| `effort` | 0.2 | Σ\|τ\| | **191.32 N·mm** measured |

`posture`'s measured 35.10° against the plan's ~60° estimate is the one worth
noting: a kernel twice as wide as the states it shapes never quite pays out
and never quite stops.

## 5. Gate

`feasibility` green, and **two of its six checks had to be re-specified**
because a position action space changes what they mean.

| # | check | result |
|---|---|---|
| 1 | static arithmetic | red — advisory since ADR-099, does not gate |
| 2 | `mj_inverse` at the reset pose | **pass** — knee 13.41 N·mm = 15.6 % of 86, hip pitch 9.11 = 10.6 % |
| 3 | worst declared shove, stepping if it must | **pass** — ξ 38.1 mm, margin 4.16–5.43× |
| 4 | contact sanity | **pass** — all four soles flat at z = +0.0004 mm |
| 5 | zero action must not score a free episode | **pass, re-specified** |
| 6 | a PD must hold the stance | **pass, re-specified** |

**Check 4 is the one that decided whether the pose was right at all.** All
four sole geoms flat and identical is what says the sign chain in the crouch
is correct; a sign error stands the machine on an edge and everything
downstream measures a different robot. The first rebuild returned all four at
**z = +6.6444 mm** — flat, so the signs were right, but the crouch had
shortened the leg and lifted the machine off the floor. Nothing in the plan
anticipated that. See §8.

**Check 5 inverted and check 6 became meaningless as written.** Under torque
control, zero action is motors-off and must fall; under position control zero
action is *hold the nominal pose* and must stand — that is the whole premise.
So the degeneracy question moved to the task, and the gate now asks it there:

```
zero action, from the reset pose, 3 s      it stands  (upright +0.994)
zero action on the DECLARED task, 12 seeds  survived 0/12, mean 60.2 of 300, all tipped
```

Check 6's six-gain PD sweep also stops meaning anything once the PD is *in the
model* — it would run the same servo six times. It became: the model's own
servo, one episode, reporting settle, peak effort and the saturation angle.

```
kp 0.300 N·m/rad, kd 0.0100 N·m·s/rad
upright +0.994   settle −1.44 mm   peak effort 27.0 N·mm   3.18× margin
saturates at 16.4° of error (86 N·mm / 5.236 N·mm/deg)
```

## 6. Budget and stopping rule

1200 iterations, 2048 environments, checkpoint every 50, seed 0, on `sb1x`.
Expected ~2.9 h at 8–9 s/iteration.

**1200 and not 2400, and it is the same experience**: at 50 Hz an iteration
runs ten solver substeps rather than five, so it costs ~2× the physics.
1200 × 2048 × 40 × 0.02 s ≈ 546 h of robot time, which is B6's 2400 at 100 Hz
exactly.

Probe 100 iterations first. **The bar was stated before the probe:** with a PD
action space, iteration 0 should already show long episodes. B7's iteration 0
was 85 steps at 100 Hz = 0.85 s. If B8's first iterations were not clearly
better *in seconds*, the action-space change had not taken and nothing else
mattered until it did.

## 7. Pass criteria

Written before the run:

1. Probe iteration 0 clearly longer **in seconds** than B7's 0.85 s.
2. Witness margin ≥ 100×.
3. Standing-pose reward = 5.3 with every term at its own weight.
4. Reward never negative in any visited state.
5. **The conjunction beats B6's 6/12.**

---

# 8. What happened

## Verification before dispatch

| check | result |
|---|---|
| standing-pose reward | **5.3000**, every one of the nine terms at exactly its own weight |
| reward on `stand8`'s 895 visited states | mean **+3.9014**, worst single state **+0.9957** — never negative |
| `feasibility` | green on all five gating checks |
| exported MJCF | `biastype="affine" gainprm="0.3" biasprm="0 -0.3 -0.01"` on all ten actuators |
| action table | every range symmetric, so zero action = nominal pose |

## The probe (100 iterations, 16 min)

| | steps | seconds | reward/step |
|---|---|---|---|
| zero action (gate, no learning) | 60.2 | 1.20 s | — |
| B7 iteration 0 (torque) | 85 @ 100 Hz | 0.85 s | — |
| **B8 iteration 0** | 71.7 | **1.43 s** | +3.038 |
| B8 iteration 28 | 87.0 | 1.74 s | +3.762 |
| B8 iteration 99 | 131.1 | 2.62 s | +4.258 |

Monotone, `best_iteration` = the latest throughout, witness **1,017× inside
tolerance**. B7's probe by contrast fell monotonically 23.5 → 8.0 steps.

**Pass criterion 1 is the weakest of the five and should be recorded as
such.** 1.43 s against 0.85 s is 1.7×, which is "clearly better" and not
"dramatically" — the word the plan used. The stronger evidence that the
change took is the MJCF check and the fact that iteration 0 already beat the
*deterministic zero-action baseline* of 1.20 s while carrying ±0.4 of
exploration noise.

## The run (1200 iterations, 3 h 22 m on sb1x)

Final iteration 1199: episode **232.7 of 300** steps, reward/step +4.581,
witness 1.405e-07 (712× inside tolerance). Critic loss rose 544 → 1107 during
the probe and fell to 757 by the end; as a fraction of the return it improved
throughout, so the rise was a growing value target rather than divergence.

## The conjunction, all 28 checkpoints, 12 seeds

Re-measured through `harness steps` — [`results/sweep-12-seeds.json`](results/sweep-12-seeds.json).
Every figure reproduces the original run's exactly.

| checkpoint | surv | step | **BOTH** | mean | terminations |
|---|---|---|---|---|---|
| probe.000050 | 2/12 | 4/12 | 1/12 | 121.8 | tipped 10 |
| probe (100) | 4/12 | 4/12 | 3/12 | 150.2 | tipped 8 |
| 000050 | 2/12 | 3/12 | 1/12 | 122.2 | tipped 10 |
| 000250 | 3/12 | 4/12 | 1/12 | 140.2 | collapsed 4, tipped 5 |
| 000500 | 3/12 | 7/12 | 2/12 | 141.3 | collapsed 4, tipped 5 |
| 000550 | 4/12 | 9/12 | 4/12 | 158.8 | collapsed 3, tipped 5 |
| 000700 | 3/12 | 9/12 | 3/12 | 141.8 | collapsed 6, tipped 3 |
| 000800 | 3/12 | 9/12 | 3/12 | 142.3 | **collapsed 9, tipped 0** |
| 000850 | 4/12 | 10/12 | 3/12 | 158.1 | **collapsed 8, tipped 0** |
| 000900 | 6/12 | 11/12 | 6/12 | 196.1 | collapsed 6 |
| 001000 | 5/12 | 11/12 | 5/12 | 195.2 | collapsed 6, tipped 1 |
| **001050** | 7/12 | 11/12 | **7/12** | 230.2 | collapsed 3, tipped 2 |
| 001100 | 6/12 | 11/12 | 6/12 | 210.4 | collapsed 3, tipped 3 |
| **001150** | 7/12 | 11/12 | **7/12** | 222.8 | collapsed 2, tipped 3 |
| **best (1176)** | 7/12 | 11/12 | **7/12** | 226.8 | tipped 5 |
| final (1199) | 6/12 | 11/12 | 6/12 | 210.8 | collapsed 2, tipped 4 |

(Abridged; the JSON has all 28.)

**7/12 against B6's 6/12.** Pass criterion 5 met.

## The tie, and the correction

Three checkpoints tie at 7/12. Re-run at 24 seeds —
[`results/tiebreak-24-seeds.json`](results/tiebreak-24-seeds.json):

| | surv | step | **BOTH** | mean | longest step |
|---|---|---|---|---|---|
| 001050 | 16/24 | 20/24 | 14/24 | 239.2 | 105.2 mm |
| **001150** | 17/24 | 21/24 | **17/24** | 239.8 | **191.3 mm** |
| best (1176) | 16/24 | 22/24 | 16/24 | 236.0 | 112.4 mm |

> **This was first reported as "1150 wins cleanly at 24 seeds" and that was
> wrong.** 17/24 against 14/24 is 12.5 pp, against a 29 pp unpaired 2σ bound.
> The unpaired bound is also the *wrong test* — every checkpoint is played
> against the same seeds, so most episodes agree for reasons unrelated to the
> policy. `harness steps` now prints McNemar over the discordant seeds, which
> is the test the design supports, and it says:
>
> ```
> 001150 vs best     only-1150 4   only-best 3   discordant 7   p = 1.000
> 001150 vs 001050   only-1150 5   only-1050 2   discordant 7   p = 0.453
> ```
>
> **The three are statistically indistinguishable.** 1150 is installed
> because something must be, and because it leads on the point estimate and
> has the longest single step; it is *not* evidenced as better than the other
> two. Reporting it as a clean win overstated the evidence, and the driver
> now makes that hard to do again.

## The failure mode, and a second correction

Across the run the deaths shift **tipped → collapsed → mixed**. Counted over
the main run's periodic checkpoints only, so the probe's and `best`/final are
not double-counted:

| phase | collapsed : tipped |
|---|---|
| early (50–200) | **5 : 33** |
| middle (700–850) | **29 : 5** |
| late (900–1150) | **26 : 10** |

and within that last phase it is falling monotonically — 6:0 at 900, 3:2 at
1050, 2:3 at 1150, and 0:5 at `best`.

> **Also first reported wrong.** Mid-run I told the operator the collapse
> mode was "a real regression hiding inside a win", from an aggregate over
> "late checkpoints". The per-checkpoint mix shows collapse **peaking around
> iteration 800–850 and then receding** — 9:0 and 8:0 there against 2:3 at
> 1150 and 0:5 at `best`. It is a phase the policy passes through on its way
> to stepping, not a trend it is on. The aggregate I quoted was true and the
> reading of it was not, which is what a per-checkpoint table is for.

In B6, **every death at every force level was `tipped` and not one was
`collapsed`**. That is still a real difference and still worth watching: the
machine now starts crouched and `height` is only 0.5 of 5.3, so sinking is
cheap. But it is a transient of learning to step, not the run's direction.

## The behaviour

At checkpoint 1150, of the 7 surviving seeds at 12: **all 7 stepped.**
`survived` and `stepped-and-survived` are the same number at every late
checkpoint. Steps land 0.05–0.17 s after a push, reach 105 mm at 12 seeds and
191 mm at 24, and several episodes contain three to five of them. Easy draws
(0.39/0.36 N) are absorbed without a foot leaving the floor.

B6's `best` for comparison: survived 2/12, stepped 2/12, both 1/12. Its 2400
scored 6/12 of 10/12 stepped.

---

# 9. What it means, and what it does not

## ~~Hazard 15 was an artefact of the action space~~ — RETRACTED 2026-08-04

**This section was the headline finding and it was wrong. It compared a servo
measurement with a policy measurement.** The paragraphs below are kept
unedited, because `method.md` says a claim that has been published is
corrected in place rather than quietly deleted. The correction follows them.

> This is the finding, and it is bigger than the headline number.
>
> `MUJOCO.md` hazard 15 — *the bracing is the resting posture* — replicated in
> **3 of 3** of experiment 002's seeds: 86.6 %, 63.3 % and 87.0 % of the 86 N·mm
> rating **with nothing pushing at all**. Experiment 001 called it "this policy
> family does not describe a machine that can be built."
>
> Under a position action space the same mechanism holds its stance at
> **27.0 N·mm peak — 31 % of rating — and `mj_inverse` says the static stance
> costs 13.41 N·mm, 15.6 %.** The torque is computed by the solver at the
> solver's rate and the policy never has to command it.
>
> Hazard 16 says *"a reward term cannot fix this"* and is correct. The action
> space could. A policy whose output *is* torque has no representation of "hold
> still" other than to keep commanding torque, and bracing is the cheapest
> stable answer available to it.

### What the two numbers actually were

**27.0 N·mm is not a policy.** It is `feasibility.py` check 6 — *the model's
own PD servo, one episode, zero action*, holding the nominal pose — quoted in
§8's gate table above as `peak effort 27.0 N·mm, 3.18× margin`. It is a
property of the mechanism and the servo gains, and there is no trained
network anywhere in it.

**002's 63–87 % is a policy** — the torque a trained network commanded, at
rest, over evaluation episodes.

Putting the two side by side compared the machine with the machine plus a
controller, and read the difference as an effect of the action space.

### What the trained policies actually do

Measured 2026-08-04 with `mechanisms/mg-legs/drivers/hazard15.py`, which
reads `data.actuator_force` against `model.actuator_forcerange` (both 86.0
N·mm) — the disturbance schedule dropped entirely, 12 seeds each, the reset
drop excluded by a 1 s settling window:

| policy | peak | **mean of rating** | **frames above 90 %** |
|---|---|---|---|
| s2 `001700` | 100 % | **73.2 %** | 49.4 % |
| s1 `001750` | 100 % | **90.5 %** | 76.6 % |
| s3 `001750` | 100 % | **78.5 %** | 58.7 % |

**Hazard 15 replicates 3 of 3, at 001's 71 % and 002's 63–87 % — or worse.**
All three saturate the actuator at some point in every settled window.

### Why, and it is more interesting than the retraction

The policy genuinely never commands torque. But it commands **position
errors large enough that the servo saturates**. §8's own gate says this
servo saturates at **16.4° of error** (86 N·mm ÷ 5.236 N·mm/deg) — and these
policies command up to **44°** on a ±45° hip pitch. The bracing did not go
away; it moved from the network's output to the servo's, where the
experiment's instrument could not see it.

That instrument is the third thing wrong here, and it is now documented in
`CLAUDE.md`: `_episodes._torque_columns` derives "torque" from the action
written to `data.ctrl`, which under a position action space is an **angle in
degrees** scored against the joint's angle range. It reported
`peak_torque_nmm` of 44.3 against a `limit_nmm` of 45 — numbers that look
like torque, are not, and would have made this section look fine.

**Hazard 16 stands, and is stronger than before: a reward term cannot fix
this, and neither did the action space.** What changed is where the cost is
paid, not whether it is paid.

## What does not follow

* **This is one seed.** Experiment 002's whole lesson is that a headline from
  one seed may not replicate — it took three seeds to turn "the reward peak
  is not the best checkpoint" into "in 2 of 3". 003 is seed 0 only. The
  obvious next experiment is 003's seeds 1 and 2.
* **Four things changed at once**, so the attribution *within* B8 is
  argument, not measurement. The probe curve makes the action space the
  strongest candidate — iteration 0 beat B7's entire probe before any
  learning — but "the positive reward contributed X" is not measured here.
  An ablation is cheap now that the harness exists.
* **B6 is not a controlled comparison.** `stand8` is 55-channel and
  torque-trained, so the engine refuses it against this bundle by name.
  6/12 and 17/24 are the same criterion on the same *task* and a differently
  actuated *machine*. That is the comparison the design intended, and it is
  not the same as playing both on one bundle.
* **The 86 N·mm limit is still a judgment**, not a datasheet number. Nothing
  here validates it; what changed is that the policy stopped living against
  it.
* **`--patience 0` is untouched.** 003 does not bear on experiment 002's
  claim about reward-vs-survival, except to note that with an all-positive
  reward `best`-by-reward landed in the tied top group for the first time in
  this project — one observation, not a claim.

## What to do next

1. **Seeds 1 and 2**, the 002 treatment. Nothing else is worth building on a
   single seed.
2. **Raise `height`** before anything else if the collapse phase is to be
   shortened. It is 0.5 of 5.3 and the machine starts crouched.
3. **B9 — the warm-start curriculum.** `cadex_train.py` has no resume or
   init-from-policy option; adding `--init-from` is contained, and it is what
   makes ADR-100's "curriculum inside the distribution" into an actual
   schedule: walk the band 0.8 → 1.2 → 1.8 → 2.5 N across four short runs
   each initialised from the last. `tasks/stand-b8/stand10.001150.cxpolicy`
   is committed as that starting network.
4. **Split `capture` again when the band goes up.** `exp(−(e/σ)²)` is flatter
   past σ than `tanh` is, so B7's saturation defect arrives *sooner* under
   kernels, not later. At B6's band ξ tops at 39 mm against σ = 50 and one
   kernel is right; at 2.5 N it will not be.
