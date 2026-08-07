# 007 — price the bracing

**Phase A is a CPU diagnostic and it is measured. Phase B is a training arm
and it is NOT WRITTEN: §§4b, 6b and 7b are empty on purpose, and nothing may
be dispatched until they are filled in (ADR-097).** §§8 and 9 below report
Phase A only, and say so.

The thresholds Phase A is read against were pinned in
`mechanisms/mg-legs/drivers/reward_audit.py` **before it was run** —
`SHARE_DEAD = 0.15`, `SHARE_ACHIEVED = 0.50`, `SD_LIVE = 0.01` — which is what
makes §8's verdicts checkable rather than chosen.

---

## 1. Question

**Does the reward price bracing at the states the machine actually reaches?**

Both answers are interesting. If `effort` is flat there, then 006's seed split
has a mechanism — PPO found a basin that stiffens against the servo because
nothing charged it for doing so — and the fix is a scale, not a new term. If
`effort` is live and simply lost an argument to `capture`, then bracing is
*priced and outbid*, the reward is not the lever, and the next move is the
mechanism (foot geometry) rather than another kernel.

### What prompted it

006 shipped `quiet` with its σ sized by a pre-registered rule to read **0.368
at the control's resting state** — deliberately live where the machine lives.
It replicated on its own subject (settled `Σ|q̇|` −69.4 % / −53.9 %) and then
**split on the falsifier**: seed 2 rests at 0.98 % duty above 90 % of rating,
seed 1 at 45.19 %, against a 25 % ceiling. `quiet` does not say *how* to be
still, and holding a joint still by not commanding motion and by stiffening
against the servo read identically on `Σ|q̇|`.

**The term that was supposed to tell those two apart is `effort`, and the
discipline 006 applied to its new kernel was never applied to the old one.**
`reward_standing.py` and `check_reward_pays.py` check hazard 9 — that every
term pays its weight *at the nominal pose*. A kernel can pass that and be flat
where the trained machine lives. That is the question this experiment asks.

---

## 2. Metric — Phase A

Decided before the audit ran, and implemented as the driver's own constants.

**Primary — `sd_paid`, the standard deviation of `w · kernel` over the visited
states, in absolute reward units**, beside `share`, the fraction of its own
weight the term collects. Both are needed and neither may be read alone:

> **`achieved` and `dead` are the same standard deviation and opposite
> findings.** A term flat because the policy solved it (`upright`, share ~0.96)
> and a term flat because it is pinned near zero (share ~0.10) are
> indistinguishable on spread. The share column separates them.

**Reported beside them:**

* **`steer`** — each term's share of the *total* spread. The declared weight is
  what the objective says a term is worth; this is what it is worth where the
  policy is.
* **`headroom`** — `w · (1 − share)`, the reward still on the table.
* the **settled window** (1.0 s, 50 frames at 50 Hz) as well as the whole
  episode, because the reset drop saturates everything — the same window
  `hazard15.py` and `jitter.py` use.

**Scored on the common bundle `tasks/stand-b8-clamp25/`** at **12 evaluation
seeds** with the **declared** disturbance. The scoring bundle's term list is
what the table shows: `stand15`'s policies trained under
`stand-b8-clamp25-quiet` and are scored here **without** their own `quiet`
term. That is deliberate — the comparison is against the control's objective —
and it is legitimate because `evaluate_episode` does not call `verify_policy`
(006 §4c, proved by running it). **`effort` is byte-identical in all three
bundles** (`w = 0.20`, σ = 191.32), so the column being compared is the same
column everywhere.

**The audit's own self-check is part of the metric, not decoration.** `alive`
is the expression `1`; its spread must be exactly zero, and the driver raises
if it is not. A statistic that cannot detect its own miscalibration is not a
measurement.

---

## 3. Mechanism

Unchanged from 004, 005 and 006. **No geometry moves in this experiment.**

| | |
|---|---|
| script | `mechanisms/mg-legs/script.py` |
| actuator rating | **86 N·mm** — models *the hardware*, the servo that would be bought |
| foot bind | 117 N·mm (ADR-082), which is why "buy a bigger servo" is not available |
| command range | ±25°, the 004 operating point |

---

## 4a. Task — what Phase A audited

Six rows against `tasks/stand-b8-clamp25/stand-task.json` (reward budget
**5.30**, 300 steps at 50 Hz, declared disturbance band 0.3–0.8 N):

| row | run directory | what it is |
|---|---|---|
| zero-action | — | the servo floor: the reward with no network in it |
| `stand12` s2 @1750 | `stand12-s2-20260804-163759` | 004's ±25° control, seed 2 |
| `stand12` s1 @1750 | `stand12-s1-20260804-214522` | 004's ±25° control, seed 1 |
| `stand15` s2 @1750 | `stand15-s2-20260806-131500` | 006's `quiet` arm, **the still seed** (0.98 % duty) |
| `stand15` s1 @1750 | `stand15-s1-20260806-180908` | 006's `quiet` arm, **the bracing seed** (45.19 % duty) |
| `stand13` @1800 | `stand13-20260805-135926` | 005-ceiling, 1850 warm-started iterations |

The two `stand15` rows are the whole point: 006's finding was that these two
runs differ **only in seed** and land in opposite basins. If `effort` cannot
separate them, it is not pricing what it claims to price.

## 4b. Task — Phase B

**NOT WRITTEN.** No arm is chosen. Candidate levers, unranked and uncosted, so
that nothing here reads as a decision:

* re-scale `effort`'s σ by `swirl_scale.py`'s median rule — the rule that gave
  `quiet` σ = 1611, applied to `Σ|τ|` over visited states rather than at the
  nominal pose;
* raise `effort`'s weight without moving σ;
* leave the reward alone and move the mechanism (foot geometry), which the
  research lane already names as the next lever and which is blocked on
  `cadex-engine-plan.md` item 3, not on the GPU.

**Before any of these is written up, print the candidate kernel's value at
three known states** — the servo floor, the control's resting state, and half
of it. That is 006's lesson and it cost nothing to apply.

---

## 5. Gate — Phase A, on CPU, in seconds

```
set -a; . ./config/env; set +a
/home/theo/cadex-train-venv/bin/python mechanisms/mg-legs/drivers/reward_audit.py \
  --task tasks/stand-b8-clamp25/stand-task.json \
  --zero-action \
  --policy jobs/stand12-s2-20260804-163759/stand12.001750.cxpolicy \
  --policy jobs/stand12-s1-20260804-214522/stand12.001750.cxpolicy \
  --policy jobs/stand15-s2-20260806-131500/stand15.001750.cxpolicy \
  --policy jobs/stand15-s1-20260806-180908/stand15.001750.cxpolicy \
  --policy jobs/stand13-20260805-135926/stand13.001800.cxpolicy \
  --seeds 12 --json > experiments/007-price-the-bracing/results/audit-clamp25.json
```

**5.7 s of CPU**, `rc 0`, on `sb1x`. Trainer interpreter, never `uv run` — the
driver imports the engine, which imports mujoco, which cdx-rl's own venv
deliberately does not pin. The pure half (`mean_sd`, `term_stats`, `classify`,
`shaping_shares`, `settle_index`, `policy_label`, `check_labels_unique`) is
tested separately under cdx-rl's interpreter: **23 checks, no mujoco.**

Two gates, both passed:

1. **The self-check** — `alive`'s spread is exactly 0.0 in all six rows.
2. **Reproducibility** — the file was regenerated after the labelling fix
   (§9) and every statistic in all six rows is **bitwise identical** to the
   first run. CPU MuJoCo evaluation is deterministic across processes here.

---

## 6a. Budget — Phase A

5.7 s of CPU, no GPU, no card contention. Total GPU-hours spent by this
experiment: **0.00**.

## 6b. Budget and stopping rule — Phase B

**NOT WRITTEN.** A training arm on this bundle costs ~4.8 GPU-h per seed at
1800 iterations (006, measured), and there is one card, so runs are serial.

---

## 7a. Pass criteria — Phase A, pinned in the source before the run

| verdict | rule |
|---|---|
| `dead` | `sd_paid < 0.01` **and** `share ≤ 0.15` — pinned low, no gradient |
| `achieved` | `sd_paid < 0.01` **and** `share ≥ 0.50` — the policy won this term |
| `live` | `sd_paid ≥ 0.01` — exploitable, regardless of share |
| `middling` | flat, share between the two |

**The pre-registered claim was that `effort` would come back `dead`.**

## 7b. Pass criteria — Phase B

**NOT WRITTEN.**

---

# ↓ Everything below was written after the measurement ↓

## 8. What happened — Phase A only

### The pre-registered verdict did NOT fire

**`effort` classifies `live` in every row, not `dead`.** Its `sd_paid` is
0.020–0.032 against a 0.01 floor, so it has gradient. The claim in §7a is
**refuted as stated**, and the driver's own docstring — written before the run
— overstated it by saying `effort` "lives on the far tail of its own
Gaussian".

### What is true instead: it is priced, and the price is trivial

`share` — the fraction of its own weight each term collects, settled window:

| term | weight | zero-action | stand12 s2 | stand12 s1 | stand15 s2 | stand15 s1 | stand13 |
|---|---|---|---|---|---|---|---|
| `alive` | 0.20 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `upright` | 1.00 | 0.33 | 0.95 | 0.96 | 0.97 | 0.97 | 0.96 |
| `capture` | 1.50 | 0.31 | 0.90 | 0.92 | 0.93 | 0.91 | 0.92 |
| `over_feet` | 1.00 | 0.37 | 0.93 | 0.94 | 0.95 | 0.92 | 0.95 |
| `height` | 0.50 | 0.93 | 0.98 | 0.99 | 0.99 | 0.98 | 0.98 |
| `arrest` | 0.30 | 0.60 | 0.92 | 0.93 | 0.94 | 0.93 | 0.93 |
| `swirl` | 0.30 | 0.80 | 0.97 | 0.98 | 0.98 | 0.97 | 0.97 |
| `posture` | 0.30 | 0.39 | 0.59 | 0.60 | 0.76 | 0.47 | 0.60 |
| **`effort`** | **0.20** | **0.41** | **0.14** | **0.12** | **0.28** | **0.10** | **0.10** |

Every other shaping term sits at 0.47–0.99 of its weight. `effort` sits at
0.10–0.28, and it is the **only term a trained policy collects less of than
the zero-action servo does** — 0.41 with nothing in the loop, 0.10 once
trained. The trained policies are strictly more torque-expensive than doing
nothing, the reward notices, and it charges them **0.06 of a 5.30 budget** for
it.

`steer` — each term's share of the total spread — says the same thing from the
other side. On `stand13`: `capture` 38.9 %, `over_feet` 21.0 %, `upright`
13.9 %, `posture` 9.0 %, `arrest` 6.5 %, `height` 4.6 %, `swirl` 3.5 %,
**`effort` 2.5 %**, `alive` 0.0 %. Last of the nine, and 15× behind `capture`.

### And it ranks 006's two basins correctly — for 0.69 % of the budget

This is the finding.

| | `stand15` s2 (still) | `stand15` s1 (bracing) | gap |
|---|---|---|---|
| settled duty > 90 % of rating (006) | **0.98 %** | **45.19 %** | 44.2 pp |
| `effort` paid | 0.0568 | 0.0200 | **0.0368 = 0.69 % of the 5.30 budget** |
| `posture` paid | 0.2269 | 0.1404 | 0.0865 |
| total paid | 4.8715 | 4.6765 | 0.1950 = 3.68 % |

**The sign is right and the magnitude is not.** `effort` is not blind to
bracing — it separates the two basins by 2.8× in share, in the correct
direction, from a measurement it has never seen. It simply does not charge
enough for PPO to care: the entire reward difference between a machine resting
at 0.98 % duty and one resting at 45.19 % is **0.037 out of 5.30**, and the
bracing seed still collected 88.2 % of the total budget.

Full per-term tables, `sd_paid`, `headroom` and `steer` for all six rows are
in [`results/audit-clamp25.json`](results/audit-clamp25.json).

---

## 9. What it means, and what it does not mean

### What it means

* **A term can pass hazard 9 at the nominal pose and be worth nothing where
  the machine lives.** `effort` pays 41 % of its weight at the servo floor and
  10 % at the trained resting state; a check at the pose would have called it
  healthy. The audit is the check that catches this, and it is 5.7 s.
* **006's seed split has a candidate mechanism.** Nothing charged the bracing
  basin more than 0.7 % of the budget for a 44-point duty difference. `quiet`
  then paid *both* basins for stillness, and PPO had no reason to prefer the
  cheap one.
* **The instrument distinguishes the two failure modes it was built for.**
  `achieved` (share ~0.96, `upright`) and pinned-low (share 0.10, `effort`)
  come apart cleanly on the share column, which is why the 2×2 exists.

### What it does not mean

* **It does not establish that re-scaling `effort` fixes anything.** That is a
  training experiment, it is Phase B, and Phase B is not written. The
  measurement licenses the *question*, not the answer.
* **It is correlational and the caveat is load-bearing.** A term flat at the
  states *this* policy visits may be flat because the policy solved it,
  because it is unreachable, or because the policy never goes where it varies.
  The driver reports; it does not conclude which.
* **Five policies, one bundle, one mechanism, 12 seeds.** No replication of
  this measurement under a different scoring bundle, and the `stand15` rows are
  scored without their own `quiet` term (§2).
* **`share` is a mean over a Gaussian kernel, not a torque.** It is not
  comparable with `hazard15`'s duty cycle except in sign; the 44.2 pp and the
  0.0368 in §8 are different quantities and the table says which is which.

### A defect in this experiment's own instrument, found while reading it

**The audit recorded rows by BASENAME, so two seeds of one checkpoint were
distinguishable only by row order.** `stand12.001750.cxpolicy` appeared twice
and `stand15.001750.cxpolicy` appeared twice; nothing was summed or corrupted —
the rows stayed separate, unlike `harness steps`, which keys `results` by
basename and silently returns `survived 36/24` — but the record could not be
read without the command that produced it. Fixed: rows now carry a
run-qualified `policy` (`<run dir>/<file>`), the resolved `policy_path`, the
`run`, and the audit records the `task` and `model` it scored under.
`check_labels_unique` refuses the same file twice. Four new tests assert both
directions.

**This is the third distinct instrument defect in this family**, after the
harness's torque columns and `harness steps`' basename keying. The pattern is
the same each time: *the identifier that is convenient to print is not the
identifier that identifies the thing.*
