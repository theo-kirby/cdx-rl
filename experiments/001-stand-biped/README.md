# 001 — stand biped: bad policy, or out-of-range task?

**Status: specified. Not run. Phase A needs no GPU at all.**

Inputs are the finished runs in `/home/theo/cadex-jobs/`, which are
**read-only** to this repository.

---

## 1. Question

Two questions, and the second only becomes askable once the first is settled.

**A. Which checkpoint of `stand-task-20260802-200109` is actually the best
one, and how would we have known without a human watching?**

**B. When that run's policy fails, is it failing because the policy is bad or
because the task is out of range?** (ADR-106.)

Both answers are interesting either way. If the reward peak turns out to be
the best checkpoint, early stopping saves ~3 GPU-hours per run and the rule
is simple. If it does not, then the naive supervisor is *wrong* and we have
learned something sharper than we set out to.

## 2. Metric

**Recovery rate: episodes surviving a shove over episodes shoved, split by
shove azimuth** — decided before dispatch, per ADR-097, and not the reward.

Reported beside it, always, and never instead of it:

* mean episode length of 600;
* the termination mix (`tipped` / `collapsed` / survived);
* peak and mean torque per motor against the 86 N·mm limit, and the
  percentage of frames above 90 % of it (hazard 15);
* how far into its own disturbance schedule each death got (ADR-106).

**The trainer's `reward_per_step` is an input to this experiment, not a
measure of it.** ADR-099 measured the two as anti-correlated across a whole
run.

## 3. Mechanism

The biped in `stand-task-20260802-200109`. It is not authored here — it comes
from the existing project — and this experiment does not modify it.

From its MJCF and bundle:

```
joints    pelvis/free + hip_roll_l  hip_pitch_l  knee_l  ankle_l  ankle_roll_l
                       hip_roll_r  hip_pitch_r  knee_r  ankle_r  ankle_roll_r
actuators 10 torque motors, ±86.0 N·mm each   (source: torque_limit_nmm)
observations 38 channels
standing CoM height  144.21 mm    (Z0, from the height reward term)
collapsed floor       72.105 mm   (= 0.5 × Z0 — ADR-106's revision)
policy    8 394 parameters
```

**The actuator limit models the hardware**, at ADR-086's re-rating of 86
N·mm. That matters for reading the torque columns: 86 is the number a real
motor has to produce.

**This machine has ankle roll.** Unlike `mg-legs`, it has genuine lateral
authority, so a sideways shove is a real question rather than an
arithmetically-guaranteed fall. Azimuth splitting is therefore meaningful in
both axes.

## 4. Task

The bundle is **already ADR-106-revised** — this is not the out-of-range task
that ADR-106 diagnosed. From `stand-task.json`:

```
episode         6.0 s, control 100 Hz, 600 steps, solver 0.002 s (5 steps/action)
reset keyframe  "solved"
reset variation pelvis: lift 15–45 mm, angular velocity ±90°/s,
                linear velocity 0–0.25 m/s
disturbance     shove   0.3–0.8 N, 0.12 s, azimuth 210°–330° (±60° sagittal),
                        at 0.3–1.5 s
                shove2  0.3–0.8 N, 0.12 s, full circle, at 1.8–3.6 s
                wind    0–0.06 N, sustained
randomisation   31 entries
reward          alive +1.0 · tilt −4.0 · height −0.01 · over_feet · capture
                drift −0.003 · stillness −0.0005 · spin −0.0005
                · effort · action · splay −0.001
termination     tipped   pel_qx² + pel_qy² above 0.15
                collapsed com_z below 72.105
```

Sagittal-biased first shove, full circle on the second, `collapsed` at
0.5 × Z0 — ADR-106's prescription, implemented. **So "the task is out of
range" is a live hypothesis but no longer the obvious one**, and that is what
makes question B worth asking again rather than assuming settled.

Capture-point arithmetic to be recomputed by `measure` against this
mechanism's own support polygon before anything is re-declared. It is not in
the bundle and must not be guessed.

## 5. Gate

Not re-run for Phase A — the run already happened. `feasibility` runs before
any *new* dispatch in Phase C.

What Phase A checks instead:

* the run's final witness: **2.820e-07, 355× inside tolerance** — above the
  100× floor, so hazard 13 does not apply;
* `device: "gpu"` in `progress.json` — it did not silently fall back to CPU;
* the bundle in the run directory matches `task_sha256`
  `21fe4171…` recorded by the trainer, and the model matches `model_sha256`
  `e3511559…`. **Play a run against its own bundle**, never the project's
  current one.

## 6. What the run already tells us — and why it is not what it looks like

From `progress.json` and the 2 500 per-iteration lines in `train.log`:

| iteration | reward/step | mean episode (of 600) | σ |
|---|---|---|---|
| 0 | −0.9412 | 85.3 | 0.4002 |
| 100 | −0.0721 | 72.4 | 0.3853 |
| 300 | +0.2698 | 204.3 | 0.3614 |
| 500 | +0.3128 | 258.4 | 0.3445 |
| **598** | **+0.3373** ← best | **277.7** | 0.3407 |
| 700 | +0.2468 | 212.8 | 0.3363 |
| 900 | +0.1933 | 196.9 | 0.3326 |
| 1200 | +0.1942 | 245.3 | 0.3302 |
| 1500 | +0.2161 | **407.6** | 0.3330 |
| **1800** | +0.2422 | **468.1** ← longest | 0.3387 |
| 2100 | +0.1496 | 354.6 | 0.3391 |
| 2499 | +0.1461 | 370.7 | 0.3375 |

Total 3 h 54 m (14 050 s).

**The obvious reading is that ~76 % of the run was spent regressing. The
episode-length column says that reading may be wrong.**

Reward per step peaked at 598 and fell to 43 % of its peak by the end. But
**mean episode length went the other way**: 277.7 steps at the reward peak,
**468.1 at iteration 1800** — the machine was surviving roughly 70 % longer
while scoring lower *per step*.

That is not hazard 19 (a reward climbing while episode length falls). It is
its mirror image, and it is exactly the shape ADR-099 warns about: survival
and the trainer's scalar disagreeing. A policy that survives longer with a
lower per-step reward is plausibly paying a small posture penalty every step
in exchange for not dying — which is the trade the metric we actually care
about would take every time.

**So the naive supervisor — "stop at 598" — might have thrown away the better
policy.** This is the finding that makes the experiment worth running, and it
sharpens what the supervisor is for: **it stops the burn, it does not choose
the checkpoint.**

For context, the other seven runs in `/home/theo/cadex-jobs`:

| run | iters | best iter | best r/step | final r/step | wall |
|---|---|---|---|---|---|
| `stand-task-20260801-182851` | 500 | 493 | 0.2149 | 0.2076 | 0.74 h |
| `stand-task-20260801-210806` | 500 | 499 | 0.1751 | 0.1751 | 0.72 h |
| `stand-task-20260802-015931` | 500 | 299 | 0.2034 | 0.1852 | 0.85 h |
| `stand-task-20260802-150139` | 1500 | 1393 | 0.2286 | 0.2169 | 1.84 h |
| `stand-task-20260802-173843` | 2000 | 1896 | 0.2450 | 0.2280 | 2.30 h |
| **`stand-task-20260802-200109`** | **2500** | **598** | **0.3373** | **0.1461** | **3.90 h** |

Every other run's best is near its end. **The 200109 run both found the best
policy this project has ever produced and then lost it** — and its best is
38 % above the next best run's. That is one run's behaviour, not a pattern,
which is another reason not to hard-code a stopping rule from it.

One more, and it is a supervisor requirement rather than a finding:
`job-task-20260801-155047/progress.json` reads `state: "training"` at
iteration 748 of 4000 with 0.18 h of wall time. The process is long dead.
**A stale `progress.json` is indistinguishable from a live one without a
liveness check**, so `supervise` must have one.

## 7. Plan

### Phase A — measure what already exists. No GPU. No dispatch.

1. `supervise --report-only` over all eight run directories. It reads
   `progress.json` for state and `train.log` for the per-iteration series
   (the series is only in the log — `progress.json` is rewritten each
   iteration and keeps just the current point plus a checkpoint list).
2. `compare` over the 52 retained checkpoints of `stand-task-20260802-200109`,
   **against its own bundle**, 12 seeds, model reloaded per episode. Full
   table: survival, episode length, tilt, drift, per-motor peak/mean torque,
   % of frames above 90 % of 86 N·mm, termination mix.
3. The same-file-twice test (ADR-103 §9) before trusting a single row.

**This is the experiment's core.** It costs CPU minutes and answers question
A directly.

### Phase B — capability, on the winner and on the reward-peak checkpoint

4. `capability` sweep at scales `[0, 0.15, 0.30, 0.50, 0.75, 1.00]` plus the
   reset-variation-only and no-variation rows, split by azimuth, on:
   * `stand8.best.cxpolicy` (iteration 598, the reward peak),
   * whichever checkpoint Phase A says survives best,
   * `stand8.cxpolicy` (the final network).
5. Answers question B. If survival is high at ×0.15 and zero at ×1.00, the
   0.3–0.8 N band is *still* out of range for this machine even after
   ADR-106, and the next action is to re-size — not to train longer.

### Phase C — only if A and B justify it

6. One dispatch with `supervise` active, a patience threshold chosen from
   the Phase A curve, and checkpoint selection by Phase A's rule rather than
   by reward. Budget stated before dispatch; `feasibility` run first.

## 8. Pass criteria

Written before the run.

| # | Passes when |
|---|---|
| 1 | `compare` produces a complete table over all 52 checkpoints, and the same-file-twice test matches |
| 2 | The table identifies a **best-by-survival** checkpoint, and we can state whether it is 598, ~1800, or neither |
| 3 | A stopping rule is written down that would have kept the best-by-survival checkpoint, with the compute it would have saved — **or** it is written down that no reward-based rule would have, which is a result |
| 4 | `capability` gives a survival-vs-scale curve that is **not flat** (a flat curve measured nothing) |
| 5 | Question B has an answer with the sweep table behind it |
| 6 | Every torque column is reported, and hazard 15 is either present or ruled out with numbers |
| 7 | A Flywheel node per phase, with `progress.json`, the compare table, the sweep table and the two candidate `.cxpolicy` files as artifacts |

**Not a pass criterion: producing a better policy.** Phase A and B spend no
GPU and are allowed to conclude that the existing runs already contain the
answer.

## 9. Risks

* **`compare` is not written yet.** This experiment is blocked on
  `harness/`. That is the point of the ordering in
  [`../../harness/DESIGN.md`](../../harness/DESIGN.md).
* **The evaluator is the instrument, and ADR-103 is about an instrument that
  was wrong.** Reload per episode; test with the same file twice; evaluate
  `drift` through the task's own expression rather than a re-derived one.
* **`stand8.best.cxpolicy` is best-by-reward.** Early in a run that can be
  the untrained network. Here it is iteration ~598, so not that — but check
  its peak torque anyway: a policy commanding 1–2 N·mm of 86 is not
  balancing, it is doing nothing.
* **One run is not a pattern.** Whatever Phase A concludes about 200109, the
  other six stand-task runs peak near their end. A stopping rule derived from
  one run's shape should be stated as provisional and tagged
  `status/provisional` in the graph.

## 10. What happened

### Phase A — 2026-08-02, sb1x, cdx-rl `5a49037`, Cadex `06d1374b` (clean)

No GPU. 612 episodes in **5.9 s** wall on 16 workers, mujoco 3.10.0 through
`CadexDynamics.evaluate_episode` — the engine's own reference runner, the one
ADR-099's table came from.

Raw output: [`results/compare-stand8.txt`](results/compare-stand8.txt),
[`results/supervise-stand8.txt`](results/supervise-stand8.txt),
[`results/supervise-all-eight.txt`](results/supervise-all-eight.txt), and the
envelopes beside them.

**The same-file-twice test (ADR-103 §9) passed** — 24 episodes, one worker,
identical rows — before any other number was looked at.
`verify_policy` accepted all 52 containers against `task_sha256`
`21fe4171…`; none was skipped. The model was matched to the bundle's own
`model.sha256` `e3511559…` by digest rather than by filename.

#### Question A is settled, and the answer is neither candidate

| checkpoint | iteration | survival | mean steps | median | trainer r/s |
|---|---|---|---|---|---|
| `stand8.000050` | 49 | **0/12** | 21.7 | 20 | −0.5822 |
| `stand8.best` (the reward peak) | **598** | **2/12** | 190.6 | 110 | **+0.3373** |
| `stand8.001700` | **1699** | **7/12** | **401.2** | **600** | +0.1739 |
| `stand8.001950` | 1949 | 6/12 | 371.4 | 434 | +0.2138 |
| `stand8` (final) | 2499 | 6/12 | 347.8 | 370 | +0.1461 |

**Best by measured survival is iteration 1699 — not 598, and not ~1800.**
Its median episode is the full 600 steps; the reward peak's median is 110.

The reward peak is **one of the worst checkpoints in the run after iteration
100**. 2/12 against 7/12: stopping there would have thrown away three and a
half times the survival, and the naive supervisor of §6 would have done
exactly that.

#### The trainer's scalar, measured against survival

| | Pearson r | n |
|---|---|---|
| survival vs iteration, whole run | **+0.83** | 51 |
| survival vs the trainer's `reward_per_step`, whole run | **+0.06** | 51 |
| survival vs the trainer's `reward_per_step`, from iteration 598 on | **−0.34** | 40 |
| survival vs iteration, from iteration 598 on | **+0.82** | 40 |

Over the whole run the trainer's scalar carries **essentially no
information** about survival. After its own peak it carries the **wrong**
information. What does predict survival is *how long the run has been going*
— which is the one thing a reward-based stopping rule is designed to cut
short.

This is ADR-099 again, on a different mechanism, against a post-ADR-106
task, and it is stronger than "anti-correlated": across the run as a whole,
r = +0.06 is not a weak signal, it is no signal.

#### Hazard 15 is present, and it is what survival costs

Every checkpoint touches its 86 N·mm ceiling at some instant — peak torque
is 99–100 % of the limit for all 51, so the **peak** column does not
discriminate. The **mean** and the **saturation** columns do:

| checkpoint | worst motor's mean | worst motor's % of frames above 90 % of limit |
|---|---|---|
| `stand8.best` (598) | 38 % of limit | 5 % |
| `stand8.001700` (1699) | **73 %** | **47 %** |
| `stand8` (2499) | **71 %** | **43 %** |

**38 of the 51 checkpoints are flagged**, and the 13 that are not are
iterations 49 through 599 — everything up to and including the reward peak.
From iteration 649 onward, every single checkpoint holds at least one motor
above 90 % of its rating on 27–59 % of frames.

So the policy that survives best is also the one **bracing hardest**, and the
two facts are the same fact. 86 N·mm is ADR-086's re-rating of what a real
motor has to produce; a mean command at 73 % of it, sustained for six
seconds, is not a number a servo of this class holds.

#### `collapsed` never fired. Not once.

Across all 612 episodes the termination mix is **454 `tipped`, 158 survived,
0 `collapsed`**.

ADR-106's revision set the `collapsed` floor at `com_z < 72.105` mm
(0.5 × Z0). No episode ever reached it: this machine always tips past
`pel_qx² + pel_qy² > 0.15` first. Half the termination design has never run,
which is precisely the shape of failure ADR-106 was written about — and this
time it is on the revised task rather than the one it diagnosed.

#### The supervisor, over all eight runs

`supervise --report-only` read all eight directories in
`/home/theo/cadex-jobs`, read-only, and reproduced this run exactly: best
598 at 0.3373 with mean episode 277.7, final 2499 at 0.1461 with 370.7, wall
14 050 s, witness 2.820e-07 at 355× — above the 100× floor, so hazard 13
does not apply. `device: "gpu"` throughout.

It flagged `job-task-20260801-155047` as **STALE**: `state: "training"` at
iteration 748 of 4000, process dead for 33 hours.

Two things the build corrected in §6's table, both from the log rather than
from the sample:

* The **episode-length peak is at iteration 1908 (542.5 steps)**, not 1800.
  Iteration 1800 does read 468.1 exactly as §6 quotes — that row was a
  sampled point, not the maximum. The gap between the reward peak and the
  survival peak is **1 310 iterations**, wider than §6 assumed.
* Four of the eight runs log an episode length at iteration 0 **above their
  own 600-step horizon** (1343.0, 1365.3, 1280.0, 1412.4). Whatever that
  first statistic is, it is not a mean episode length. Left in, it wins the
  peak search and the report announces those runs peaked at iteration 0.

## 11. What it means

### Question A: answered

The best checkpoint of `stand-task-20260802-200109` is
**`stand8.001700.cxpolicy`, iteration 1699, at 7/12 survival** — measured by
playing it, which is the only way it could have been known.

**Pass criterion 3 resolves the second way, and that is the result.** No
reward-based stopping rule would have kept it. Its trainer reward is 0.1739
against the peak's 0.3373 — 52 % of it — so any rule of the form *stop when
reward has not improved for N iterations* stops at 598 + N and, for any N
small enough to save compute, stops before 1699. The rule that would have
worked on this run is *do not stop*, which is not a rule.

What the numbers support instead, stated as provisional (§9's last risk
stands — one run is not a pattern):

* **A supervisor should stop on divergence, device and liveness, and not on
  reward patience.** Those three catch runs that are broken. Reward patience
  catches runs that are working.
* **Selection must be by `compare`, always.** This run cost 3.9 GPU-hours;
  choosing correctly from within it cost 5.9 CPU-seconds. The ratio is
  2 400:1 and it is the cheapest instrument in the repository.
* **Episode length is the better live proxy**, if a live proxy is wanted —
  it is monotone with survival here where reward is not (r = +0.83 vs
  +0.06 against iteration). It is still a proxy, and it is still not a
  selection.

### What Phase A does *not* mean

* **It does not mean the run should have gone longer.** Survival is 7/12 at
  1699 and 6/12 at 2499; the curve is noisy and flat-ish past ~1500, not
  climbing. Nothing here says iteration 4000 would be better.
* **It does not mean 1699 is a good policy.** 7/12 is a coin flip with a
  limp, and it is bought with motors held at 73 % of their rating for six
  seconds. Hazard 15 says this policy is not buildable as specified.
* **It does not settle question B.** That every death is `tipped` and none is
  `collapsed` is suggestive — the machine tips rather than sinks — but
  whether the 0.3–0.8 N band is in range is Phase B's measurement, not an
  inference from Phase A's.

### What changes because of it

1. `compare` runs before any checkpoint is named "the" policy. It is now
   cheap enough that not running it is a choice.
2. The `collapsed` termination on this task is **dead code**. Either the
   floor moves up or it should be dropped rather than left to look like a
   safeguard that has never fired.
3. Hazard 15 is a live constraint on this mechanism, not a hypothetical.
   Torque cost belongs in the reward or the actuator limit belongs in the
   task — and that is a decision to take before the next dispatch, not after.

*Phase B follows in §10 below once the capability sweep has run.*
