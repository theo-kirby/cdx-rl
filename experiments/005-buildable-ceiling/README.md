# 005 — Where does the buildable policy top out?

**Status: dispatched 2026-08-05. Everything below §7 was written BEFORE the
run started** (ADR-097: a metric chosen after looking at the curve is a metric
chosen by looking at the answer).

## 1. The question

Experiment 004 found the operating point: capping the command range at **±25°**
cuts resting duty above 90 % of the servo's rating from **51.8 % → 13.5 %**
while costing nothing decisive in stepping (15/24 against the control's 18/24,
McNemar 4:1, **p = 0.375**). Seed 1 replicated it at 15/24 and 12.8 % duty.

So there is a policy family that a bench could plausibly hold. **The question
this asks is where it stops improving.** Every run this repository has ever
dispatched at 1800 iterations was **still climbing**: 003's three seeds peak at
1700–1750, and clamp25's argmax is 1750 — the last checkpoint written. A
ceiling has never actually been observed; only the end of the budget has.

## 2. Why this was vetoed once, and what changed

005 was **proposed and vetoed on 2026-08-04** by its own CPU pre-flight gate,
for 70 s of CPU instead of ~10 h of card. That veto stands as correct on its
own terms and part of its reasoning was later retracted — see
`flywheel` node `small-recipe-2040`. The surviving finding: **bracing rises
with training time in all three arms**, across two seeds. What did *not*
survive is the stated mechanism, and the pre-registered rule flipped sign
between seeds.

**What changed is the goal, not the evidence.** If the objective is a policy
that steps, survives *and* is buildable, then *"it steps better **and** braces
more"* is a **tradeoff to measure**, not a reason to look away. This run
measures both curves against each other rather than asking whether one of them
is flat.

## 3. Shape of the run

**A continuation, not a fresh run**, using `--init-from` (ADR-124). This is
scoped honestly: it is **1850 further iterations on top of `stand12` seed 1's
checkpoint 1750**, reaching an effective iteration **3600**. It is *not* a
3600-iteration run from scratch, and no reading below treats it as one.

| | |
|---|---|
| bundle | `tasks/stand-b8-clamp25/` (`task_sha256` `3d627ef4b9a509fe…`) |
| warm start | `jobs/stand12-s1-20260804-214522/stand12.001750.cxpolicy` (sha256 `783c31603051e79f…`) |
| label / seed | `stand13`, seed 2 |
| iterations | 1850 (effective 1750 → 3600), checkpoint every 50 |
| hyperparameters | `stand12`'s: envs 2048, unroll 40, epochs 5, hidden 64 64 |
| cost | ~5 h, against ~9.9 h from scratch |

### The caveat this carries, stated up front

The warm start crosses the **trainer-digest boundary mid-stream**: checkpoints
up to 1750 were produced by `aacfa823…` and everything after by the
`--init-from` trainer. The **update rule is unchanged** — the flag is a
verified no-op when unused (two runs of the modified trainer and one of the
unmodified trainer produce the same digest over `observations`, `network`,
`normaliser`, `evaluation`), and when used it only sets initial weights. But
the digest moved, and `method.md` §8b's bridge-run protocol is the thing to
reach for if any comparison here turns out to need it.

### As dispatched

```
run dir   jobs/stand13-20260805-135926
trainer   /home/theo/cadex-prs/training/cadex_train.py
          (branch cdxrl/train-init-from, sha256 4c1f24f8bdf2368a…)
```

`init-from stand12.001750.cxpolicy  iterations=1750  sha256=783c31603051e79f…`,
then `iteration 0  reward/step +4.31394  episode 655.4` — against a cold
start's +3.27 and 88.2 on the same bundle. The warm start took.

**The first dispatch was killed after three minutes and is not in the
record.** `tools/train.py` fills any hyperparameter you do not pass from
`RUN_200109`, and two of those are not what `stand12` ran — **`discount`
0.995 against 0.99, and `gae_lambda` 0.97 against 0.95.** A continuation whose
update rule changes at the join is not a continuation, and nothing in the
output says so: the run started perfectly happily. The dispatch above passes
all fourteen explicitly and the run directory's `hyperparameters.json` now
differs from `stand12` seed 1's in exactly `seed` and `iterations`.

This is CLAUDE.md's *"pass all fourteen hyperparameters, always"* biting one
level up — the defaults are right for a fresh run and wrong for a warm start,
because a warm start has something to agree with.

## 4. Metrics, stated before dispatch

* **`harness steps`**, the **conjunction** (stepped ≥10 mm AND survived), at
  **24 evaluation seeds**, read through **McNemar over the discordant seeds** —
  not the unpaired 2σ bound, which is the wrong test and conservatively so.
  Use the **variadic** `--policy A B`, never the repeated form: `steps`
  declares `--policy` as `nargs="*"` and the repeated form silently keeps only
  the last.
* **`mechanisms/mg-legs/drivers/hazard15.py --series --stride 50`** at **6
  evaluation seeds** across the whole run, under
  `/home/theo/cadex-train-venv/bin/python` (never `uv run`). Reported as the
  **mean fraction of rating** with duty above 90 % read as a *consequence* —
  a duty cycle is a threshold statistic and a linear fit on one can predict
  140 %.

## 5. Pre-registered readings

1. **Late beats early, paired.** Best checkpoint in the last third of the
   continuation against the best in the first third, same 24 seeds, McNemar.
   *If the run has topped out, this is null* — and a null here is the result,
   not a failure.
2. **Resting duty above 90 % of rating at the best late checkpoint**, against
   clamp25's measured **13.5 %** (seed 2) and **12.8 %** (seed 1). This is the
   buildability half of the tradeoff.
3. **Against 003 seed 2's `001700` at 18/24**, the best unclamped result this
   project has, on the conjunction at 24 seeds through McNemar.

## 6. What would falsify the premise

If the conjunction at effective 3600 is **not** better than at 1750 while
resting duty **is** worse, the clamp25 family has topped out and further
training buys bracing only. That is a clean stop signal for this arm and
points at §5b — foot geometry — as the next move rather than more iterations.

## 7. Results

**The run finished `rc 0` in 4.78 h** (17 206 s), 1849 of 1850 iterations, 36
periodic checkpoints. Effective iteration **3599 of 3600**.

### The trainer's scalar was wrong again, and by more than usual

`reward_per_step` **peaked at iteration 6** (+4.8089) and declined for the
remaining 1843 iterations, ending at +4.6654. Episode length fell 655 → ~230.
Read as a training signal, this run looks like 1843 iterations of pure
regression.

**The measured conjunction went the other way: 13/24 → 18/24.** This is
experiment 001's finding (r = −0.34 after the peak) in its sharpest form yet,
and the strongest argument in this repository for `--patience 0`. Anyone
watching the reward curve would have killed this run in the first hour.

### The dip is the fresh critic, and ADR-124 predicted it

| effective iteration | 1800 | 1900 | 2350 | 2600 | 2950 | 3300 | 3450 | 3550 |
|---|---|---|---|---|---|---|---|---|
| conjunction /24 | 13 | 9 | 9 | 15 | 14 | 16 | **18** | **18** |

The warm start does **not** improve monotonically. It falls for ~600
iterations before recovering and exceeding. That is exactly what a trained
actor against a randomly-initialised critic should do, and it is the cost
`--init-from` documents rather than hides. **A 600-iteration warm-start run
would have concluded the transfer failed.**

`stepped` climbs steadily throughout — 19/24 → 23/24 — while `survived` is
roughly flat. The policy is learning to step more, not to stand still better.

### The three pre-registered readings

**1. Late beats early, paired.** Best in the last third, `001700` at
**18/24**, against the best in the first third, `000050` at **13/24**:
**6 discordant to 1, p = 0.125**. The direction is right and 24 seeds is not
enough to call it. `001700` and `001800` are 3:3, **p = 1.000** — the run's
last 100 iterations are indistinguishable from each other, which is the
closest thing here to a ceiling.

Against the continuation's **own starting point**, `stand12.001750` at
**15/24**: 6:3 discordant, **p = 0.508**. So 1850 further iterations bought
+3/24 on the point estimate and **the evidence does not establish it.**

**2. Resting duty at the best late checkpoint — the buildability half.**

| | duty >90 % of rating | mean % of rating |
|---|---|---|
| clamp25 seed 2 (004) | 13.5 % | — |
| clamp25 seed 1 (004) | 12.8 % | — |
| **stand13 `001800`** | **12.6 %** | **55.6 %** |

Over the whole continuation: **15.5 % → 12.6 %, slope −1.15 pp / 1000
iterations.** The underlying quantity — mean fraction of rating, which is what
CLAUDE.md says to fit rather than the threshold statistic — moves 53.3 % →
55.6 %, about **+1.5 pp / 1000 iterations**. From 55 % that is roughly 23 000
further iterations to approach the 90 % threshold.

**This is the finding that matters, and it contradicts the 2026-08-04 veto's
surviving premise.** That veto rested on *bracing rises with training time in
every arm*, extrapolated. Measured directly over 1850 further iterations of
the clamp25 arm, **it does not rise** — the duty falls and the mean creeps.
The veto was right to refuse the *number* it was given (a linear fit on a duty
cycle that predicted 140 %); it was wrong about the direction for this arm.

**3. Against 003 seed 2 `001700`, the best unclamped policy this project has
produced.** Both bundles were checked section by section first — `episode`,
`disturbance`, `reset_variation`, `randomisation`, `termination`,
`variation_algorithm`, `reward` and `observations` are **identical**, and only
`actions` differs — so a seed fixes the same reset draw and the same shove
schedule in both and the episodes genuinely pair.

| | 003 s2 `001700` (unclamped) | stand13 `001800` (clamp25) |
|---|---|---|
| conjunction | 18/24 | **18/24** |
| survived | 20/24 | 19/24 |
| stepped | 21/24 | **23/24** |
| longest step | 121.8 mm | 50.1 mm |
| hazard 15, mean % of rating | **74.4 %** | **55.6 %** |
| hazard 15, duty >90 % | **51.8 %** | **12.6 %** |

**Paired McNemar: 2 discordant to 2, p = 1.000.**

### What this experiment concluded

**The buildable arm has caught up.** A policy the servos can hold — 12.6 % of
the time above 90 % of rating, against the unclamped arm's 51.8 %, a **4.1×
reduction** — is now **indistinguishable on the conjunction** from the best
policy this project has ever trained, and it steps *more often* (23/24 against
21/24) while stepping *shorter* (50 mm against 122 mm).

Experiment 004 established the clamp cost 15/24 against a control's 18/24 and
called it indistinguishable at p = 0.375. **Training the clamped arm further
closes that gap on the point estimate too, and it does so without paying for
it in bracing.**

### What it did not establish

* **Neither improvement is significant at 24 seeds.** Late-vs-early is
  p = 0.125 and against its own start p = 0.508. The honest statement is *"the
  point estimate rose 15 → 18 and the evidence does not establish it"*, not
  *"1850 more iterations bought 3 episodes"*.
* **One seed.** 002's whole lesson applies. This is seed 2 of the clamp25
  family continued once.
* **Whether it has topped out.** `001700` and `001800` are indistinguishable
  (p = 1.000), which is suggestive and is also what any two adjacent
  checkpoints look like. §6's falsification did **not** fire — the conjunction
  is *not* worse and the duty is *not* worse — so the arm has not
  demonstrably stopped improving. It has stopped improving *fast*.

### Provenance

Trainer `/home/theo/cadex-prs/training/cadex_train.py`, sha256
`4c1f24f8bdf2368a…` — **not** the old `aacfa823…`, because `--init-from` was
the change in flight when this ran.

**That caveat resolved itself the same day.**
[theo-kirby/cadex#2](https://github.com/theo-kirby/cadex/pull/2) merged at
`75efe784`, and `main`'s `training/cadex_train.py` is now `4c1f24f8bdf2368a…`
— **the exact digest this run used.** So 005 was trained on what is now the
trunk trainer and needs no bridge run.

What still needs one is the other direction: **001, 002, 003 and 004 all ran
under `aacfa823…`**, which is now one merge behind. The update rule did not
change — verified, in that the modified trainer and the unmodified one produce
the same digest over `observations`, `network`, `normaliser` and `evaluation`
— but the digest did, and `method.md` §8b is the protocol for a comparison
that has to cross it. The §7 comparison against 003 seed 2 is one such
crossing; it is reported as a paired McNemar on measured behaviour rather than
on anything the trainer said, which is the reading least exposed to it.
