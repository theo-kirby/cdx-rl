# 002 — seed replication: does 001's shape survive a change of seed?

**Status: specified. Sections 1–7 written and committed before dispatch.**

Experiment 001 concluded two things and stamped the same caveat on both:
*one run is not a pattern.* Everything it found rests on a single run,
`stand-task-20260802-200109`. 001 §9 names this as the top risk, and Phase A
and Phase B each repeat it under "what it does not mean".

This experiment does the cheapest thing that converts those from anecdote to
finding: **it runs the same thing again, three more times, with different
seeds.**

It is **not** 001 Phase C. Phase C is about a *changed* task — the
torque-cost reward Phase B recommended — and stays blocked on the unbuilt
`measure` and `feasibility` drivers, and on the biped's authoring script,
which is not in this repository. See §5 and `../../CLAUDE.md` §5.

---

## 1. Question

**Does the shape 001 measured in `stand-task-20260802-200109` replicate
across seeds, or did 001 measure one run's accident?**

Specifically: in how many independent runs of the same task is the trainer's
reward peak *not* the best checkpoint by measured survival?

Both answers are interesting, which is the point.

* **If it replicates** — "the reward peak is not the best checkpoint" is a
  property of *this task*, not of one run. The supervisor rule 001 derived
  (stop on divergence, device and liveness; **never** on reward patience)
  generalises, and `status/provisional` comes off Phase A's node.
* **If it does not** — 001 measured one run's accident. Both its conclusions
  need the caveat made permanent rather than removed, and the anti-correlation
  it found becomes a thing that *can* happen rather than a thing that *does*.

## 2. Metric

**Unchanged from 001, and fixed before dispatch** (ADR-097). Restating it
here rather than cross-referencing it, because a metric you have to go and
look up is a metric that can quietly drift:

**Recovery rate: episodes surviving a shove over episodes shoved, split by
shove azimuth.**

Reported beside it, always, and never instead of it:

* mean episode length;
* the termination mix (`tipped` / `collapsed` / survived);
* peak and mean torque per motor against the 86 N·mm limit, and the
  percentage of frames above 90 % of it (hazard 15).

**The cross-seed statistics, also named now rather than after the curves are
visible.** Per seed:

1. the iteration of the trainer's **reward peak**;
2. the iteration of **best measured survival**;
3. the **sign of (survival at final − survival at reward peak)**.

The headline number is (5) in §7: in how many of four seeds is the reward
peak not the best checkpoint.

**The trainer's `reward_per_step` is an input to this experiment, not a
measure of it** (ADR-099, and 001's own finding of r = +0.06 over the run and
−0.34 after its peak).

## 3. Mechanism

**Unchanged.** The same biped, played from the same read-only run directory.

```
model_sha256  e3511559eeb3d5ab3a369852a6c7d7e5fcfc256a259f1f4f41284ffb2cef0ae1
joints        pelvis/free + hip_roll_l  hip_pitch_l  knee_l  ankle_l  ankle_roll_l
                            hip_roll_r  hip_pitch_r  knee_r  ankle_r  ankle_roll_r
actuators     10 torque motors, ±86.0 N·mm each   (source: torque_limit_nmm)
policy        8 394 parameters
```

**The actuator limit models the hardware**, at ADR-086's re-rating of 86
N·mm. That is what makes 001's hazard-15 finding — the later policies holding
a motor at 71 % of rating *with nothing pushing them* — a statement about a
machine that cannot be built, rather than a statement about a simulation
parameter. This experiment asks whether that resting posture is also a
property of the task or an accident of one seed.

Nothing here authors or modifies the mechanism.

## 4. Task

**Unchanged.**

```
task_sha256   21fe4171a5499258379e20e04a091fd33c8e1a04b5f2d19acb27264140a7f235
source        /home/theo/cadex-jobs/stand-task-20260802-200109/stand-task.json
```

`/home/theo/cadex-jobs` is a **read-only input**. `tools/train.py` copies the
bundle and its model into each run directory — the model matched by the
bundle's own `model.sha256`, not by filename — so the eight finished runs are
never written to and each new run is self-describing.

**No capture-point arithmetic is owed here, and that is a claim, not an
omission.** ADR-100's arithmetic sizes a disturbance band for a *new or
re-sized* task. This task is neither: it is the same bundle, byte for byte,
by digest. And 001 Phase B did better than estimate that band — it *measured*
the survival-versus-force curve empirically, which is the quantity the
arithmetic exists to approximate. Re-deriving an estimate of something
already measured would be a worse answer wearing more work.

## 5. Gate

**`feasibility` is not run, and this section is why.**

`method.md` step 7 gates a **new or re-sized** task behind `feasibility`'s six
checks. This task is unchanged by digest (§4), and 001 Phase B established
empirically that it is in range:

* **48/48** episodes survived unshoved;
* **~50 %** survived at the full declared 0.3–0.8 N shove magnitude.

The declared band is in range and was shown to be so by playing it, not by
estimating it. Running the six checks here would re-answer, with weaker
evidence, a question already answered with stronger. `feasibility` remains
unbuilt and remains owed — before the *next* task, not this one.

What *is* checked before the GPU starts is the pre-flight list, and its
results are recorded in [`results/preflight.md`](results/preflight.md):

| check | expected |
|---|---|
| `uv run python tools/smoke.py` | PASS, 13/13 |
| `git -C /home/theo/cadex log --oneline -1` | `06d1374b` (ADR-104 §8b) |
| `nvidia-smi` | ~0 % util, <1 GB used — one run at a time on this card |
| `df -h /home/theo` | 2.4 T free |
| bundle / model digests | `21fe4171…` / `e3511559…` |
| §1–7 of this README | committed **before** dispatch |
| divergence guard | both branches fired on purpose, output recorded |

## 6. Budget and stopping rule

**4 seeds × 1 500 iterations**, one at a time on this card (`cloud.md`).

| | |
|---|---|
| iterations | 1 500 per seed |
| environments | 2 048 |
| checkpoint every | 100 |
| per seed | **~2.34 h** at 200109's measured 5.62 s/iteration |
| **total** | **~9.4 h** (range 7.4–9.4 h; run 150139 managed 4.42 s/it) |
| **GPU-hours** | **~9.4 — the repository's first** |
| disk | ~40 MB of checkpoints |

**Why 1 500 and not 2 500.** Phase B measured survival as flat-ish from 1 500
to 2 500 (48–56 %). The last thousand iterations of 200109 bought nothing
measurable, and four seeds at 1 500 answers this experiment's question
strictly better than two seeds at 2 500 would.

### The hyperparameters, and the confound that nearly ran

Every one of the fourteen is passed explicitly, **even where it equals a
default**, so that §6 is checkable against what actually ran:

| | trainer default | 200109, and this run |
|---|---|---|
| `envs` | 256 | **2048** |
| `unroll` | 20 | **40** |
| `epochs` | 4 | **5** |
| `discount` | 0.97 | **0.995** |
| `gae_lambda` | 0.95 | **0.97** |
| `entropy` | 1.0e-3 | **2.0e-3** |
| `initial_std` | 0.3 | **0.4** |
| `hidden` | [64, 64] | [64, 64] |
| `learning_rate` | 3.0e-4 | 3.0e-4 |
| `clip` | 0.2 | 0.2 |
| `value_weight` | 0.5 | 0.5 |

Planning this experiment found that `tools/train.py` passed only
`--iterations --envs --checkpoint-every --seed`, letting the other ten fall
back to the trainer's defaults — **six of which differ from what 200109
ran**. Dispatched as written, this sweep would have run a *different
algorithm* from the run it exists to replicate; the seeds would have been
comparable to each other and to nothing else; and **no output would have
shown the confound**, because the trainer records what it was given, not what
it was meant to be given. The values above were recovered from 200109's own
`.cxpolicy` header (`training.hyperparameters`) and are now the defaults in
`train.py`. The resolved set is written to `hyperparameters.json` in each run
directory.

### Stopping rule

**No reward patience** (`--patience 0`). 001 Phase A found that no
reward-based stopping rule would have found the right checkpoint, so a rule
that stops on reward would actively destroy this experiment's subject.

A seed stops early, and the sweep continues to the next one, on:

* **non-finite** `loss` or `reward_per_step`;
* **σ collapse** — `action_std` below **0.02**. Threshold read off the data,
  not invented: 200109 decayed 0.4002 → 0.3375 over 2 500 iterations, so 0.02
  (5 % of `initial_std`) sits an order of magnitude below anything healthy.
* a `device` that is not `gpu`;
* a dead process.

Each seed is capped at **3 h** wall (`--timeout 10800`).

Whatever stops a seed is recorded. A seed that stops early is data, not a
failed experiment — see §7 criterion 1.

## 7. Pass criteria

Written before the run.

1. **All four seeds complete 1 500 iterations, or the reason each stopped is
   recorded.**
2. **Every witness margin printed is > 100×** (hazard 13). Under that, the
   witness is recording what the GPU rounded the network to rather than what
   the network is.
3. **`compare --seeds 48` produces a complete table per seed**, with
   same-file-twice passing on each.
4. **For each seed we can state the reward-peak iteration and the
   best-by-survival iteration, with the separation bound.** 48 seeds gives a
   2σ bound of ~10 pp on a difference in survival; **any claim inside that
   bound is reported as tied**, not as a winner. (Phase B established that
   `--seeds 12` cannot separate two checkpoints at all: 2σ ≈ 20 pp.)
5. **We can say in how many of four seeds the reward peak is *not* the best
   checkpoint — and that number, whatever it is, is the result.**
6. **Seed 0 is the replication control.** 200109 itself ran seed 0, with the
   same hyperparameters and the same trainer
   (`trainer_sha256 aacfa823…`, verified identical to the current
   `training/cadex_train.py`). Its curve should therefore track 200109's
   recorded `reward_curve[0:1500]`, read out of that run's `.cxpolicy`
   header. **A large divergence is not a failure of this experiment** — it is
   a finding about run-to-run comparability on this card, and it would
   qualify every seed-to-seed claim made here.

**Not a pass criterion: a better policy.** This experiment spends ~9
GPU-hours to find out whether two existing conclusions are real. It is
allowed to conclude that they are and produce nothing installable.

---

*Sections 8 and 9 are written after the run, below this line, and not
before.*

---

## 8. What happened

**The sweep was stopped after 2 of 4 seeds**, by request, because the GPU was
needed for other work. ~5.1 GPU-hours of the ~9.4 budgeted. Seeds 0 and 1 ran
to completion; seed 2 was signalled at iteration 537 and seed 3 never
dispatched. The full stop record, including how it was stopped and what it
costs, is [`results/stopped.md`](results/stopped.md).

**So this is a two-seed result and is reported as one.** §7 criterion 5 asked
"in how many of *four* seeds", and that question is not answered here.

### The seeds that ran

| | seed 0 | seed 1 |
|---|---|---|
| iterations | 1500 / 1500, exit 0 | 1500 / 1500, exit 0 |
| device | `gpu` throughout | `gpu` throughout |
| wall | 2.12 h | 2.12 h |
| **witness margin** | **517×** | **588×** |
| `supervise` trend | `survival-diverges` | `survival-diverges` |
| same-file-twice | **passes**, 96 episodes | **passes**, 96 episodes |
| checkpoints played | 16 | 16 |

Criterion 2 holds: both margins are far above the 100× floor (hazard 13).
Criterion 3 holds: `compare --seeds 48` produced a complete table for each,
with the determinism check passing on both.

### The headline — criterion 5

**In 2 of 2 seeds the trainer's reward peak is not the best checkpoint by
measured survival**, and in both the gap clears the separation bound by a
factor of two:

| seed | reward peak | survival there | best by survival | survival there | gap |
|---|---|---|---|---|---|
| 0 | iteration **594** | **16/48 = 0.333** | iteration **1299** | **36/48 = 0.750** | **+41.7 pp** |
| 1 | iteration **510** | **12/48 = 0.250** | iteration **1499** | **37/48 = 0.771** | **+52.1 pp** |

2σ separation bound at n=48: **20.4 pp**. Both gaps clear it outright.

For comparison, 001 measured the same thing on 200109 at 12 seeds: reward
peak 598 at 2/12, best 1699 at 7/12. **Same shape, on two runs 001 never
saw, with four times the seeds.**

#### A correction to §7 criterion 4

§7 says *"48 seeds gives a 2σ bound of ~10 pp"*. **That is wrong**, and the
number used above is not it. `√(2·0.25/n)` at n=48 is 10.2 pp, but that is
**one** standard error; 2σ is twice it, 20.4 pp. `harness/compare.py` has
always used `2 × √(2·0.25/n)` — `separation()` computes exactly that — so the
analysis is consistent with the harness and the README's figure was the
outlier. Recorded rather than quietly corrected, because §7 was supposed to
be the thing that could not move after the fact. **The conclusion is
unaffected**: 41.7 and 52.1 pp clear the stricter bound, so it would have
held under either convention.

### The correlation, restated per seed

| | 001 (200109) | seed 0 | seed 1 |
|---|---|---|---|
| survival vs trainer reward, whole run | +0.06 | **+0.11** | **+0.12** |
| survival vs trainer reward, **after its peak** | −0.34 | **−0.71** | **−0.83** |
| survival vs iteration, whole run | — | **+0.93** | **+0.93** |

The sign replicates in both seeds on both spans, and the post-peak
anti-correlation is **two to three times stronger** than 001 measured. The
whole-run figure is near zero for the reason 001 gave: it averages the early
stretch, where reward and survival climb together because the network is
going from nothing to something, with the span a stopping rule would actually
act on.

`survival vs iteration` at +0.93 in both is the plainest statement of it: on
this task, over this range, **survival just keeps improving with training**,
while the scalar the trainer optimises turns over at around iteration 500–600
and declines.

### Criterion 6 — seed 0 against 200109

Full output in [`results/seed0-vs-200109.txt`](results/seed0-vs-200109.txt).

| | |
|---|---|
| bitwise identical iterations | **0 of 1500** |
| first shared iteration | −0.941203 vs −0.940929 — **already differs** |
| mean \|Δ\| / max \|Δ\| | 0.0398 / 0.1038 |
| **Pearson r over the shared span** | **+0.9885** |
| reward peak | 598 vs **594** — four iterations apart |

Same seed, same `trainer_sha256`, same hyperparameters, same card, and not
one iteration reproduces. The divergence is present at iteration 0, so it is
not accumulated drift — MJX and XLA reductions on a GPU are not bitwise
reproducible, and PPO amplifies whatever they differ by.

**But the shape survives it.** r = +0.9885, and the statistic this experiment
actually uses — where the reward peak falls — lands within four iterations of
1500. That is the outcome that qualifies the seed-to-seed claims least.

The exception is **episode length**, which is markedly noisier: its peak
moved from iteration 1497 to 1330 and from 395.7 to 576.9 steps. It is
printed beside reward in every report, and this says plainly that the two
columns do not deserve equal confidence.

### Hazard 15 — the resting posture

`capability` on each seed's best-by-survival checkpoint, 48 seeds per row.
The `no-variation` row is the one that matters: **nothing is pushing at all.**

| | seed 0 (`001300`) | seed 1 (`final`) | 001 |
|---|---|---|---|
| **mean torque, nothing pushing** | **86.6 % of limit** | **63.3 % of limit** | 71 % |
| peak torque, nothing pushing | 99.8 % | 99.3 % | — |
| survival, nothing pushing | **48/48** | **48/48** | 48/48 |

001's 71 % sits between the two. **The resting posture replicates**: these
policies stand still by holding motors near their rating, and one of them
peaks at 99.8 % of an 86 N·mm limit with the robot doing nothing.

It is worse than "a property of the run", because it **tracks the thing we
selected for**. Mean fraction of limit climbs monotonically with the
checkpoints that survive best:

```
seed 0   iteration   99 → survival 0.083, mean torque 0.268 of limit
         iteration  594 → survival 0.333, mean torque 0.343   (reward peak)
         iteration 1299 → survival 0.750, mean torque 0.732   (best survival)
         iteration 1499 → survival 0.750, mean torque 0.828
```

The checkpoints that recover best are the ones bracing hardest. This is not
an artefact to be tuned away at the supervisor; it is the task's reward
buying survival with torque.

### The task is in range — and the tied set

`capability`'s sweep of the declared band, 48 seeds per row:

| scale | seed 0 | seed 1 |
|---|---|---|
| no variation | 48/48 | 48/48 |
| 0.00 (reset variation only) | 0.896 | 0.917 |
| 0.50 | 0.896 | 0.938 |
| **1.00 (full declared magnitude)** | **0.750** | **0.771** |

Nothing falls off a cliff, and full magnitude still recovers three times in
four. §5's claim that the declared band is in range holds against these runs
as well as against Phase B's.

And, exactly as 001 warned, `compare` **cannot crown a winner even at 48
seeds**. The best-by-survival checkpoint is statistically indistinguishable
from the next several in both seeds:

```
seed 0   tied: 001300, final, 001400, 001200
seed 1   tied: 001300, final, 001400, 001200, 001100
```

What the table supports is *"the late checkpoints, as a group, decisively
beat the reward peak"* — not *"iteration 1299 is the best one"*.

## 9. What it means, and what it does not mean

**It means 001's first conclusion is no longer resting on one run.** The
trainer's reward peak was the wrong checkpoint in 200109, in seed 0 and in
seed 1 — three independent runs, the latter two with 48 evaluation seeds and
gaps of 42 and 52 pp against a 20.4 pp bound. The supervisor rule 001 derived
follows from the task, not from an accident: **stop on divergence, device and
liveness; never on reward patience.** `--patience 0` stays the default, and
now has evidence behind it rather than one cautionary example.

**It means hazard 15 is a property of this task, not of one policy.** Two
fresh runs both stand at 63–87 % of an 86 N·mm rating with nothing pushing
them, bracketing 001's 71 %. Phase B's recommendation — a torque-cost term —
is not an idea about one bad policy; it is the obvious response to a reward
function that will buy survival with torque every time it is allowed to.

**It does not mean four seeds agreed, because four seeds did not run.** Two
did. Two runs plus 001's is a materially weaker claim than the design asked
for, and criterion 5's question — *in how many of four* — is unanswered. The
`status/provisional` tag comes **down a notch, not off**: seeds 2 and 3 are a
4.3-hour dispatch away and the exact command is in `results/stopped.md`.

**It does not mean survival keeps improving forever.** The measured range is
1 500 iterations. Phase B found survival flat-ish from 1 500 to 2 500 on
200109, and nothing here contradicts or extends that; `survival vs iteration`
of +0.93 describes the span that was run and must not be read as a trend line
to extrapolate.

**It does not mean a checkpoint was selected.** No policy from these runs is
installable, and §7 said in advance that a better policy is not a pass
criterion. The tied sets above are the reason: at 48 seeds the top four or
five checkpoints are indistinguishable, so "play the late ones" is the whole
of what is supported.

**It does not mean run-to-run comparability is free.** Seed 0 reproduced
nothing bitwise against its own recorded twin. Every claim here is a claim
about shape — where a peak falls, which group of checkpoints wins — and shape
is what survived at r = +0.9885. **A claim about a specific value would not
have survived**, and none is made.

### Two things worth carrying forward

**The azimuth split is half empty.** §2's metric is recovery rate *split by
azimuth*, and at full magnitude the two positive-quadrant bins have **n = 0**
in both seeds — every shove sampled fell in 180°–360°:

```
+x  (0°–90°)     n=0
+y  (90°–180°)   n=0
-x  (180°–270°)  n=17   survival 0.824
-y  (270°–360°)  n=31   survival 0.710
```

So the metric that was fixed before dispatch is being reported on half the
directions it names, in every experiment that has used it — 001 included.
Whether that is the disturbance sampler's convention or a genuine gap in the
task is **not** settled here, and it should be settled before the azimuth
split is quoted as evidence of anything.

**Episode length is the noisy column.** Criterion 6 showed its peak moving by
167 iterations and 181 steps between two runs whose reward curves correlate
at +0.9885. `supervise` prints it beside reward everywhere, for the good
reason that reward alone cannot tell surviving from scoring — but it should
not be read with the same confidence, and `results/` now has the numbers to
say so.
