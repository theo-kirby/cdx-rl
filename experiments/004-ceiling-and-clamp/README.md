# Experiment 004 — whether a buildable policy can step

**Written before dispatch.** ADR-097: a metric chosen after looking at the
curve is a metric chosen by looking at the answer. Nothing below is edited
after the fact; where a run departs from it, §8 says so and says why.

> **Revised 2026-08-04 11:51Z, with nothing scored.** This experiment was
> dispatched as two arms, A (ceiling) and B (clamp). A was **cancelled at
> iteration 158 of 2500**, 30 minutes in, and **no checkpoint of it was ever
> played** — the revision is a cost argument, not a response to a result. §2
> is the accounting. The GPU-hours it freed went to a second *seed* of B,
> which is what 002's lesson says the answer needs.

## 1. Question

Experiment 003 at four seeds left the same policies holding the worst motor at
**73–91 % of the 86 N·mm rating on average, with nothing pushing**, above 90 %
of it for **49–77 % of frames**. Hazard 15 did not dissolve; it moved from the
network's output to the servo's.

***Can a policy that cannot saturate its motors still step?***

That is the whole experiment. A better policy that cannot be built is not
progress, so this question gates every other one — including the ceiling.

## 1b. Why the ceiling arm was cancelled

The original design paired the clamp against a fresh 2500-iteration run of the
unchanged task, seed 2, to ask where improvement stops. **The accounting does
not survive being written down.**

003 seed 2 already ran this exact configuration — same seed, same task, same
fourteen hyperparameters, same box — for **1800 iterations**, and it is on
disk with 37 checkpoints, scored, with hazard 15 measured. A 2500-iteration
re-run recomputes 1800 of those iterations to reach 700 new ones. That is
**6.9 h of card time to buy 1.9 h of novel computation**, and the other 5 h
reproduces something already owned. 002 measured same-seed reruns at
**r = +0.9885 in shape while 0 of 1500 bitwise identical**, so the recomputed
prefix is not even a free replication — it is a slightly different draw of a
curve whose shape is known.

**The reason it cost 3.6× what it should is that the trainer cannot warm-start
from a checkpoint** — `cadex-wishlist.md` #11. With `--init-from`, the ceiling
question is a 1.9 h continuation from `stand10.001700` and worth running
immediately. Without it, it is the most expensive way to ask the least urgent
question in the file.

And the question it asks is about a policy family this experiment exists to
show is **unbuildable**. Pushing that family further up a curve is not
progress until the clamp result says whether the curve is worth climbing.

**003 seeds 1, 2 and 3 are the control**, at no cost: 1800 iterations each,
same box, same hyperparameters, hazard 15 measured at 73.2 %, 90.5 % and
78.5 % mean of rating. Running the clamp at **1800 iterations** makes the
comparison length-matched and seed-matched against a control already paid for.

## 2. Metric

`harness steps`, the conjunction — **stepped ≥ 10 mm AND survived 300/300** —
at 12 evaluation seeds, escalating to 24 for ties, read through **McNemar over
the discordant seeds** and not the point estimate. Plus
`mechanisms/mg-legs/drivers/hazard15.py` for the resting torque, which is the
instrument 003 lacked.

## 3. The runs, and the control that was already paid for

**1800 iterations, checkpoint every 50** — 37 checkpoints, at the measured
**9.95 s/iteration** on `sb1x`, so **~5.0 h each**. There is one card now:
`sb9x` was retired 2026-08-04, so arms are serial.

| | task | bundle sha256 | seed | status |
|---|---|---|---|---|
| **control** | `tasks/stand-b8/` unchanged | `5572adf265aa51cb…` | 1, 2, 3 | ✅ **003, already on disk** |
| **arm 1** | `tasks/stand-b8-clamp15/` | `50e2d08bcb6b0785…` | 2 | dispatched 11:51Z |
| **arm 2** | selected by the §7 fork | — | — | chained |

**The control is free and it is exact.** 003's seeds 1, 2 and 3 are the same
task, the same fourteen hyperparameters, the same box and the same 1800
iterations. Matching the clamp's length to theirs is what turns this from
"a clamped run and a remembered number" into a paired comparison at the seed
level — every arm is played against a control that shares its seed, so the
reset draw and the whole disturbance schedule are held fixed.

**Arm 1 is seed 2** because that was 003's best (`001700` at 18/24). If the
clamp costs stepping anywhere, it costs it against the strongest control, and
a null there is worth more than a null against a weak one.

Note this is **not** a same-seed rerun of the control: the bundle differs, so
the trajectories diverge from iteration 0. The seed match buys paired
*evaluation*, not paired training.

## 4. What B changes, and it is one thing

The action table's ranges, capped at **±15°**:

```
hip_roll   ±30 → ±15      knee       ±30 → ±15
hip_pitch  ±45 → ±15      ankle      ±25 → ±15
                          ankle_roll ±20 → ±15
```

Nothing else. The MJCF is **byte-identical** (`80eaa18f6025d589…`) — it does
not need to change, because `ctrllimited="false"` on every actuator means the
MJCF never clamped `ctrl` at all. The range lived entirely in the bundle.

### Why ±15°

003's own gate measures the servo saturating at **16.4° of error**
(86 N·mm ÷ 5.236 N·mm/deg), and `forcerange="-0.086 0.086"` in the MJCF
confirms the 86 N·mm is a real model limit rather than the judgment it has
been called. The measured policies command up to **44°** on a ±45° joint —
nearly three times the saturation threshold. A ±15° cap puts every command
below it.

**This makes saturation much harder, not impossible, and the difference
matters.** The servo sees *error*, which is command minus actual joint angle.
A joint displaced 10° one way while commanded 15° the other still presents
25° of error. So the claim under test is "much less saturation", not "none",
and the measurement is the duty cycle rather than a yes/no.

### What it costs, stated in advance

This removes two thirds of the policy's authority on hip pitch, on a joint
whose full range it was using. **B may simply fail to step, and that is a
result rather than a failure of the experiment**: it would say this mechanism
cannot step within its torque budget, which is a statement about the
*actuator sizing* and lands squarely on ADR-077's lesson — the hopper's leg
was short by 2.2× and a training run was spent finding out.

## 5. Why this is a bundle edit and CANNOT be a script edit

`tasks/stand-b8-clamp15/` is derived from `tasks/stand-b8/` by
`make_clamp_bundle.py` in this directory — the cap applied programmatically,
never by hand.

This was first written up as a debt against the script-is-the-source-of-truth
rule, to be repaid by editing `script.py` on the laptop. **That was wrong, and
the reason is the more interesting half of this section.**

**In Cadex a position servo's action range IS its joint's physical limits.**
`CadexDynamics._ACTION_SOURCES` maps `("position", "angular")` to
`angle_limits_degrees`, with a stated rationale: *"a setpoint outside them is
a command the joint cannot obey."* The exported MJCF proves the coupling — its
joint ranges are the same ten numbers as the action table's:

```
hip_roll ±30    hip_pitch ±45    knee ±30    ankle ±25    ankle_roll ±20
```

So capping the range in `script.py` would narrow **the joint as well as the
command**, and the machine would physically be unable to flex past 15°. That
is a different experiment and a confounded one: removing reachable
configurations could prevent stepping on its own, with nothing to do with
torque. The question here is about what the policy may *command*, not about
what the machine can *do*.

Editing the bundle is therefore not a shortcut around the script — it is the
**only** way to express this experiment, because the mechanism vocabulary
cannot say "a joint that moves ±45° and a policy that may only ask for ±15°".
That is now `cadex-wishlist.md` #12.

004-B remains a claim about **committed bytes** rather than a re-derivable
one, as 001 and 002 are for `stand-b2`. But the fix is upstream, not a task
somebody forgot to do here.

### What the clamp actually buys, in error terms

The servo sees **error = command − actual joint angle**, so narrowing commands
does not bound error:

| | max |command| | max joint excursion | worst-case error |
|---|---|---|---|
| 003 (as run) | 45° | 45° | 90° |
| **004-B** | **15°** | 45° | **60°** |
| a script clamp *(not run)* | 15° | 15° | 30° |

All three exceed the 16.4° saturation threshold, so **none of them prevents
saturation** — the earlier phrasing "structurally impossible" was wrong twice
over.

What 004-B does is remove the policy's ability to *choose* saturation. Any
large error that remains comes from the machine being displaced rather than
from the network asking for a setpoint 44° away. **That makes the result
readable either way**: if the resting duty cycle collapses, the bracing was a
policy strategy; if it survives at ±15° commands, the torque is coming from
the dynamics, and the finding is about actuator sizing rather than about
learning. Both are worth 7 hours.

## 6. Budget and stopping rule

**~10 h across the two arms**, serial, on `sb1x` — against the ~13.8 h the
cancelled design would have spent to answer one question with one seed.
`--timeout 25200` (7 h) per run: 1800 iterations needs ~5.0 h, and 003's four
seeds all finished inside this cap, three of them at 4.96–5.04 h.

`--patience 0`. Reward patience stays off — 001 found no reward rule would
have found the right checkpoint, and 003's four seeds put the trainer's
reward-best at 4 of 4 never better than the run's best.

All fourteen hyperparameters passed explicitly, equal to 003's:
`envs 2048, unroll 40, epochs 5, hidden [64,64], lr 3e-4, discount 0.99,
gae_lambda 0.95, clip 0.2, entropy 2e-3, value_weight 0.5, initial_std 0.4`.
Note these differ from `train.py`'s `RUN_200109` defaults in two —
`discount` and `gae_lambda` — which is the silent-substitution trap.

## 7. Pass criteria, and the fork — written before the runs

1. **The clamp steps at all.** The best checkpoint scores ≥ **6/12** on the
   conjunction — B6's baseline, the number 003 was measured against. Below
   that, the clamp is too tight to control the machine.
2. **The clamp is measurably less braced.** Resting duty cycle above 90 % of
   rating **below 25 %**, against the control's 49–77 %. This is the point of
   the experiment; a clamped run that steps as well as 003 while still
   saturating has taught us nothing.
3. **The comparison is paired and reported as such.** Clamp-vs-control on the
   conjunction through McNemar over discordant seeds, not through the point
   estimates, and the tied set named.
4. **It replicates.** Criterion 2 must hold in **both** seeds, or in both
   clamp settings if the fork takes the relax branch. 003's hazard-15
   retraction is trustworthy *because* it was 3 of 3; a one-seed collapse of
   the duty cycle would be exactly the n=1 claim 003 had to retract.

### The fork, decided in advance

Arm 2 is chosen by criterion 1 applied to arm 1, **evaluated by a script, not
by judgement** — `jobs/chain-004-fork.sh` scores arm 1 at 12 seeds the moment
it exits and dispatches accordingly. The branch cannot be chosen by looking at
the curve, because nobody looks at the curve first.

| arm 1's best `both` at 12 seeds | arm 2 | why |
|---|---|---|
| **≥ 6** | **clamp15, seed 1** | the clamp is controllable — the open question is whether the effect replicates |
| **< 6** | **clamp25, seed 2** (`3d627ef4b9a509fe…`) | ±15° is too tight — find where on the range control returns, rather than replicating a failure |

±25° is the relax point because it is still **above** the 16.4° saturation
threshold — so it is a real test of the same hypothesis and not a retreat to
the control — while sitting far below the ±45° the 003 policies were using. It
narrows 6 of 10 actions; ±15° narrows all 10.

**Stated in advance: the likely outcome is a trade.** The clamp steps worse
and braces less. If so the deliverable is the *curve* between ±15°, ±25° and
±45°, not a winner, and the next question is where on it a buildable machine
sits — which is an actuator sizing question, not a policy one.

## 8. What happened

**Both arms ran on `sb1x`, 1800 iterations, seed 2, `rc 0`, 4.77 h and 4.95 h.
The fork took the relax branch on its own at 16:37:58Z** — `chain-004-fork.sh`
scored arm 1 at 4/12, below criterion 1's bar, and dispatched clamp25 without
anybody having looked at a curve.

### The curve, which is the deliverable

Conjunction at **24 evaluation seeds**; hazard 15 at 6 seeds, settled window,
against the 86 N·mm rating. Each policy is played on **its own** bundle — a
clamp25 network re-scaled onto a ±45° action table is a different policy — and
the bundles differ only in the action table, so the reset draw and the whole
disturbance schedule are identical across arms and the pairing is valid.

| max command | policy | **both** /24 | surv | step | mean % rating | **duty % > 90 %** |
|---|---|---|---|---|---|---|
| **±45°** (003 s2) | `stand10.001700` | **18** | 20 | 21 | 74.4 | **51.8** |
| **±25°** (arm 2) | `stand12.001750` | **15** | 18 | 20 | 53.8 | **13.5** |
| ±25° | `stand12.best` | 12 | 18 | 16 | 56.7 | 18.3 |
| ±25° | `stand12.001450` | — | — | — | 43.8 | 11.3 |
| **±15°** (arm 1) | `stand11.001600` | **5** | 11 | 12 | 33.9 | **0.2** |
| ±15° | `stand11.cxpolicy` | — | — | — | 34.7 | 1.5 |
| ±15° | `stand11.001300` | — | — | — | 29.4 | **0.0** |

Paired, McNemar over the discordant seeds (criterion 3):

| | discordant | p |
|---|---|---|
| ±45° vs **±25°** | 4 : 1 | **0.375** — indistinguishable |
| ±45° vs ±15° | 13 : 0 | **0.0002** — ±15° is worse |
| ±25° vs ±15° | 11 : 1 | **0.0063** — ±25° is better |

The unpaired 2σ bound at n=24 is 28.9 pp and separates none of them, which is
the point of carrying both tests.

### 1. The bracing was a policy choice, not a dynamics requirement

This is the finding, and it closes the question 003's retraction opened.

003 left it undecided whether the resting torque came from the machine or from
the network commanding setpoints 44° away on a ±45° joint. **It was the
network.** Cap what the policy may ask for and the duty cycle above 90 % of
rating falls from **51.8 % to 13.5 %, and to 0.0–1.5 % at ±15°** — while the
machine still stands (5/6 and 6/6 in the undisturbed episodes). Nothing about
the mechanism, the gains or the reward changed.

That is the difference between *this design cannot be built* and *this policy
could not be built*, and they call for completely different next steps.

### 2. ±25° is the operating point, and it is nearly free

**Criterion 1 passes at ±25° (7/12, 15/24) and fails at ±15° (4/12, 5/24).
Criterion 2 passes at both** — 13.5 % and 0.2 %, against a bar of 25 % and a
control at 51.8 %.

So ±25° buys a **3.8× reduction in saturation duty** for a stepping difference
the paired test cannot distinguish from zero (4 : 1 discordant, p = 0.375).
±15° buys near-total elimination and pays for it with stepping the paired test
separates decisively (13 : 0, p = 0.0002 — *every* discordant seed favours the
control).

Note the peak column is 100 % of rating in every row, settled window included.
The peak is not the statistic; it never was. §7 of `hazard15.py` and the mean
and duty cycle are.

### 3. What did NOT get established

**Criterion 4 is not met, and the relax branch is why.** The pre-registered
fork spends arm 2 on a second clamp *setting* rather than a second *seed*, so
the duty-cycle collapse is replicated across settings and **n=1 in seeds**.
003's hazard-15 retraction is exactly the failure mode this criterion exists
to prevent, and it would be dishonest to count a monotone three-point curve as
the replication that was asked for.

**`stand12` seed 1 was dispatched 21:45Z** to close it, on the criteria
already stated above. Until it lands, every claim here is one seed per point.

Also not established: the ceiling. §1b cancelled that arm and it stays
cancelled — it is a 1.9 h question the moment `--init-from` exists and a 6.9 h
one until then.

### 4. An instrument note

`harness steps --policy A --policy B` **silently scores only B.** `steps`
declares `--policy` as `nargs="*"` while `compare` and `capability` use
`action="append"`, so the repeated form overwrites instead of accumulating and
reports no error — the run above lost a checkpoint to it before the variadic
form `--policy A B` was used. Nothing was mis-scored, but a table could
quietly have been built on half the checkpoints it named.
