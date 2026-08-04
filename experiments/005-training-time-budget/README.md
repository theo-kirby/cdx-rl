# Experiment 005 — does the buildable policy keep climbing, and stay buildable?

**Sections 1–7 were written before anything was run**, and are transcribed
unedited from the dispatch plan of 2026-08-04. §8 and §9 are after.

> **THE RUN WAS NEVER DISPATCHED.** 005 was designed as a 3600-iteration
> extension of experiment 004's clamp25 arm, ~9.9 h of card time. Its own
> pre-flight gate — §6b below, written before it was run, costing **70 s of
> CPU** — returned the veto branch. §8 is what the gate found. **The gate is
> the experiment**, and it answered the question the 10 h run was going to
> ask, for 1/500 000 of the compute.

---

## 1. Question

**Four of four runs now stop mid-climb.** 003's seeds 1, 2 and 3 peaked at
1700–1750 of 1800, and clamp25's argmax is **1750** — its best checkpoint is
the last periodic one written, with best ≤1200 = 6/12 against best >1200 =
7/12. Every run this project has dispatched at 1800 iterations has been
improving when it stopped.

004 answered *can a buildable policy step*. 005 asks **how well** —

***Does the ±25° policy keep improving past 1800 iterations, and is it still
buildable when it gets there?***

Both answers are interesting. If it keeps climbing, the deliverable is better
than measured and the 1800-iteration budget is the thing to change. If it
stops, 1800 is the right length and the next lever is elsewhere.

The difference from the ceiling arm cancelled in 004 §1b: **this is the
deliverable's own curve**, on the arm that passes both criteria, not a detour
up the unbuildable one that hazard 15 disqualifies.

## 2. Metric

`harness steps`, the conjunction — **stepped ≥ 10 mm AND survived 300/300** —
at **24 evaluation seeds**, read through **McNemar over the discordant
seeds**, plus `hazard15.py` at 6 seeds on the best checkpoint. `--patience 0`.

Both halves are needed and neither substitutes for the other: 004's whole
point is that a stepping result which is not buildable is not a result.

## 3. Mechanism

`mechanisms/mg-legs/script.py`, unchanged. Actuator limit **86 N·mm**, which
models **the hardware** re-rated to continuous duty (~40 % of MG90S stall,
ADR-086) — and, per §1232–1276, is additionally bounded by the **foot**: past
2.581 N × 45.5 mm = **117 N·mm the foot rolls instead of pushing** (ADR-082).
The torque budget is not raisable without a bigger foot.

## 4. Task

`tasks/stand-b8-clamp25/stand-task.json`, digest `3d627ef4b9a509fe…` —
identical to experiment 004 arm 2. 300 steps at 50 Hz, position action space,
command range capped at ±25° of each joint's own ±45°. Reset variation,
reward terms and the 0.3–0.8 N shove band are unchanged from 003, so the
comparison against 003 seed 2 is length- and seed-matched.

## 5. Gate

`feasibility`'s six checks were passed by this bundle for experiment 004 and
the bundle is byte-identical; they are not re-run. **The gate that matters
for 005 is a new one**, §6b, because the failure mode 005 risks is not a
modelling error — it is spending 10 h to un-do the previous result.

## 6. Budget and stopping rule

| | |
|---|---|
| label | `stand13` |
| seed | 2 — matches 004 arm 2 and 003 seed 2, so all three share evaluation pairing |
| iterations | **3600**, checkpoint every 50 → 72 checkpoints |
| wall | ~9.9 h at the measured 9.9 s/iter; `--timeout 43200` (12 h) |
| hyperparameters | all fourteen, identical to 003 and 004 |

**The recomputation cost, stated up front**: 3600 iterations recomputes the
1800 already on disk to reach 1800 new ones — **50 % waste, ~5 h**. That is
the `--init-from` tax (engine plan item 1) and it is the reason to go *long*
rather than to 2700: the recomputed prefix is a fixed cost, so a longer run
has a better ratio, not a worse one.

## 6b. The pre-flight gate — a CPU check that can veto the run

**Before spending ~10 h, ask whether bracing grows with training.** 004
measured hazard 15 at *one checkpoint per arm*. If resting duty is a function
of training time, a longer run walks the deliverable back out of buildability
and the 10 h buys a worse policy.

Run `hazard15.py` across clamp25's checkpoint **series** — every 250
iterations, 6 seeds. CPU MuJoCo, so it does not contend with the card.

**The decision rule, stated before looking (ADR-097):**

| duty % > 90 % across the series | reading | action |
|---|---|---|
| flat or falling | bracing is not a function of training time | **run 005 as designed** |
| rising, and > 25 % by extrapolation to 3600 | longer training un-does the 004 result | **do not extend** — the finding is that ±25° has a *training-time* budget, which is a better result than a longer run |

This is the check that was not done before dispatching 004's arms.

## 7. Pass criteria — written before the run

1. **The climb is real or it is not.** Best checkpoint in (1800, 3600] vs best
   in [0, 1800], McNemar at 24 seeds. If late still wins, 3600 is *also* short
   and that is the finding.
2. **It stays buildable.** Resting duty above 90 % of rating **below 25 %** at
   the best late checkpoint. If the policy buys its improvement with torque,
   005 has reproduced hazard 15 at a new clamp and that is the result.
3. **It reaches the control.** Best late checkpoint vs 003 s2 `001700`
   (18/24), paired. The prize is a policy that matches the unclamped arm while
   braced at a quarter of its duty cycle.
4. Reported paired, with the tied set named.

---
---

## 8. What happened

**The gate ran, and it vetoed the run.** Nothing was dispatched. No GPU time
was spent.

`hazard15.py` grew a `--series` mode so the gate is one command rather than a
shell loop, and the series was scored for **all three of 004's arms**, not
just clamp25 — the control was already on disk and cost seconds, and without
it there is no way to tell a clamp property from a general one.

```
/home/theo/cadex-train-venv/bin/python mechanisms/mg-legs/drivers/hazard15.py \
    --series jobs/stand12-s2-20260804-163759 --stride 50 \
    --task tasks/stand-b8-clamp25/stand-task.json --seeds 6
```

**Cost: 22.91 s wall for 35 checkpoints × 6 seeds; ~70 s for all three arms.**

### 8.1 The series, at stride 250

Resting duty above 90 % of rating, settled window, nothing pushing, 6 seeds:

| iteration | unclamped (003 s2) | ±25° (004 arm 2) | ±15° (004 arm 1) |
|---|---|---|---|
| 250 | 0.0 % | 0.6 % | 0.0 % |
| 500 | 41.8 % | 5.7 % | 0.0 % |
| 750 | 36.1 % | 0.1 % | 0.0 % |
| 1000 | 49.7 % | 16.9 % | 0.0 % |
| 1250 | 48.5 % | 16.7 % | 0.0 % |
| 1500 | 55.8 % | 14.2 % | 0.3 % |
| 1750 | 57.7 % | 13.5 % | 0.9 % |

Whole-series least-squares slope, at stride 50 (35 checkpoints):

| arm | first → last | slope | extrapolated to 3600 |
|---|---|---|---|
| unclamped | 0.0 % → 57.7 % | **+38.69 pp / 1000** | **140.6 %** |
| ±25° | 0.0 % → 13.5 % | **+11.36 pp / 1000** | **38.7 %** |
| ±15° | 0.0 % → 0.9 % | +0.15 pp / 1000 | 0.5 % |

**The rule fires the veto branch: rising, 38.7 % > 25 %.**

### 8.2 …and the rule's own instrument is broken, which has to be said

**A duty cycle cannot be 140.6 %.** The control's extrapolation is
arithmetically impossible, which proves the linear model wrong for every row
in that table — including the one the veto was read off. The veto cannot rest
on the magnitude **38.7 %**. It can only rest on the **direction**, which is
unambiguous.

So the honest question is where each arm *saturates*, and that needs a
statistic the gate did not pre-register. Split at iteration 1000, stated as
**post-hoc** and not as the decision rule:

| arm | half | n | duty mean | duty slope | mean % of rating | mean-% slope |
|---|---|---|---|---|---|---|
| unclamped | < 1000 | 19 | 20.4 % | +53.98 | 53.9 % | +71.93 |
| unclamped | ≥ 1000 | 16 | **54.9 %** | +7.22 | **78.0 %** | **−1.71** |
| ±25° | < 1000 | 19 | 2.2 % | +3.01 | 34.3 % | +19.93 |
| ±25° | ≥ 1000 | 16 | **15.0 %** | **−0.86** | **54.6 %** | **+10.01** |
| ±15° | < 1000 | 19 | 0.0 % | +0.00 | 22.6 % | +7.33 |
| ±15° | ≥ 1000 | 16 | 0.1 % | +0.44 | 30.6 % | +10.97 |

Two things fall out, and they point opposite ways:

* **Duty has plateaued for ±25°** — 15.0 % mean over the second half, slope
  **−0.86 pp/1000**, i.e. flat-to-falling. Read on duty alone in the regime
  that matters, the veto's premise is false.
* **Mean torque has not.** ±25° is at **54.6 % of rating and still climbing at
  +10.01 pp/1000**, while the unclamped arm's mean has **saturated at 78.0 %**
  (slope −1.71).

### 8.3 Why the veto holds anyway, for a better reason than the fit

**Duty is a threshold statistic on a rising distribution.** It stays near zero
while the mean is low, then moves sharply once the mean approaches the
threshold. The control shows exactly that transition: below iteration 1000 the
unclamped arm sat at **mean 53.9 %, duty 20.4 %**; its mean then rose to 78.0 %
and its duty **tripled to 54.9 %**.

**±25° at iteration 1750 sits at mean 54.6 % — the unclamped arm's
pre-transition operating point — and is still climbing at +10 pp/1000.**
Extending its mean at the measured rate reaches the control's 78 % plateau at
roughly iteration **4100**. A 3600-iteration run lands it near **73 %**, inside
the band where the control's duty was 55 %.

So the flat duty over 1000–1750 is not evidence that ±25° has found a
buildable equilibrium. It is what the control also looked like just before its
duty went up. **Do not extend.**

### 8.4 What was actually learned, which 004 could not have said

004 measured hazard 15 at one checkpoint per arm and concluded the bracing was
a policy choice. The series says something stronger and more specific:

**Bracing rises with training time in all three arms.** The command range does
not decide *whether* the policy learns to brace — it sets the **rate** and the
**plateau**. Unclamped reaches 55 % duty by iteration 1000; ±25° is at 15 % at
1750 and still walking its mean up; ±15° is at 0.1 % and climbing its mean at
the same +11 pp/1000 as ±25°, from far enough back that it has not arrived.

## 9. What it means, and what it does not mean

**004's headline survives, with a qualifier it did not have**: the bracing is a
policy choice, **at a fixed training budget**. Resting duty is a function of
*both* the command range and the training time, and 004 varied only the first.
Quoting "13.5 % at ±25°" without "at 1800 iterations" overstates it.

**The ±25° operating point has a training-time budget, and 1800 iterations is
inside it.** That is a real result and a more useful one than the extension
would have produced: it says the 1800-iteration length that four runs have now
been suspected of truncating is, on the buildable arm, *load-bearing* rather
than arbitrary. The runs stop mid-climb on the stepping metric and stop
**about where they should** on the buildability metric. Those two facts were
not known to be in tension until now.

**What it does not mean:**

* **It is not a measurement at 3600.** Everything past iteration 1750 is
  extrapolation from a 35-point series on one seed, and §8.2 shows how badly
  a linear model can behave here. The claim is *"the risk is real enough not
  to spend 10 h"*, not *"duty would have been 38.7 %"*.
* **It is n=1 in seeds** — 004's own unmet criterion 4. `stand12` seed 1 was
  dispatched 21:45Z and lands ~02:15Z; the series should be re-scored on it.
  If seed 1's duty trend differs in *shape*, this section is the one to revisit
  first.
* **It does not say ±15° is the answer.** ±15° holds duty at 0.1 % but costs
  stepping decisively (5/24, McNemar 13:0, p = 0.0002, 004 §8). Its mean is
  climbing at the same rate as ±25°'s from a lower start, so its buildability
  is a head start, not an immunity.
* **It says nothing about the stepping ceiling.** Whether the conjunction keeps
  improving past 1800 is still unmeasured. The gate did not answer §1's first
  half; it established that the second half fails first, which is why the run
  was not worth making.

### What runs next

Not `stand13`. In order:

1. **`stand12` seed 1** (in flight, lands ~02:15Z) — closes 004's criterion 4,
   and re-scores this series at a second seed.
2. If the duty collapse replicates, **the next lever is the mechanism, not the
   optimiser** — specifically **foot geometry**, since §3's 117 N·mm foot bound
   is what makes "buy a bigger servo" unavailable. That is blocked on
   `cadex-engine-plan.md` item 3 (the pinned engine cannot build `script.py`),
   **not on the GPU**.
3. If it does *not* replicate, **clamp25 seed 3** — a 2-of-3 split is 002's
   situation and needs the third seed before anything is built on it.

### Artifacts

| | |
|---|---|
| `results/series-clamp25-s2.json` | 35 checkpoints × 6 seeds, ±25° |
| `results/series-unclamped-s2.json` | 35 checkpoints × 6 seeds, 003 seed 2 control |
| `results/series-clamp15-s2.json` | 35 checkpoints × 6 seeds, ±15° |
| `results/split_half.py` | §8.2's table, reproducible from the three JSONs |
