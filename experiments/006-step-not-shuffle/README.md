# 006 — one step, not twenty

**Sections 1–7 are written before dispatch (ADR-097). 8 and 9 are empty until
the runs land, and are visibly separate below.**

---

## 1. Question

**Does costing joint velocity in the reward make the machine stand still — and
what does that cost in stepping?**

Both answers are interesting. If `quiet` reduces the fidget without costing
the conjunction, the reward gains a term the mechanism wanted and every later
experiment inherits it. If it reduces the fidget *and* reduces stepping, that
is a real trade-off finding about a reward that has no velocity term at all,
and it is worth knowing before anything is built.

### What prompted it, and what Phase A did to the premise

Watching the trained policies play, two things looked wrong: the legs jitter,
and a push is answered with many small foot corrections rather than one step.
**Phase A confirmed both as measurements and refuted the explanation the plan
started from.** The original plan had two arms — a band raise (arm S) and this
one. Arm S was dropped on the evidence in §4b. This experiment is arm Q, twice.

---

## 2. Metric

Decided before dispatch, and scored on the **common** bundle
`tasks/stand-b8-clamp25/` — the one the control trained on. `harness steps`
and `mechanisms/mg-legs/drivers/jitter.py` do not call `verify_policy`, so
they score across bundles; **`harness capability` and `harness compare` do
not**, and that was proved by running it, not by reading it (§4c).

**Primary — settled `Σ|q̇|`, in deg/s**, over the ten actuated joints, mean
and RMS, from `jitter.py --seeds 12`, measured after a **1.0 s settle window**
because the reset drop saturates everything. This is exactly the quantity the
`quiet` kernel sums, which is the point: the term is measured on its own
subject.

**Reported beside it, so 006 stays comparable with the published tables:**

* the **conjunction** — stepped ≥ 10 mm AND survived 300/300 — at **24
  evaluation seeds**, read through **McNemar over the discordant seeds**, not
  the point estimate. The unpaired 2σ bound at n=24 is **28.9 pp** and cannot
  separate anything of interest.
* **lifts per surviving episode** and the **per-episode median
  `longest_step_mm`**. `harness steps` prints `longest_step_mm` as a **max
  over all episodes**, so the published 50.1 mm and 121.8 mm are single best
  steps and one outlier moves either; the per-episode rows are in the envelope,
  so the median is computable from the same JSON. **The max is reported too,
  labelled as a max.**
* **`hazard15` settled duty > 90 % of rating**, which is criterion Q3.
* **command jitter in degrees** and the **sign-reversal rate per second** —
  because a magnitude alone cannot tell fast tracking from chatter.

**`reward_per_step` is NOT a metric here and cannot be.** Adding `quiet` moves
the reward's total weight 5.3 → 5.6, so the scalar is not comparable across the
two bundles even in principle. ADR-099 again, with a second reason.

---

## 3. Mechanism

Unchanged from 004 and 005. **No geometry moves in this experiment.**

| | |
|---|---|
| script | `mechanisms/mg-legs/script.py` (clamped variant `rollout/script-clamp25.py`) |
| model | `tasks/stand-b8-clamp25/model-model.xml`, sha256 **`80eaa18f6025d589796315fcad45bb70bf72e55da750864f60b5e0b3cc71fdb3`**, 14179 bytes |
| actuator limit | 86 N·mm — **the hardware**, an MG90S continuous-duty judgment at ~40 % of the 216 N·mm stall rating. The model's own `forcerange` is the same number. |
| command clamp | **±25°** on eight joints, ±20° on the ankle rolls. 004's operating point, and it **holds** — this experiment does not touch buildability. |

**Measured from the compiled model on sb1x, 2026-08-06**, because two of these
are wrong in the sources that quote them:

| | |
|---|---|
| robot mass | **0.302011 kg** (sum over the 23 robot bodies; the MJCF's 155.822 total includes the 155.52 kg grounded plane) |
| weight | **2.9627 N** |
| CoM height at the reset keyframe | **140.944 mm** → ω₀ = **8.34278 rad/s**, which is the `8.3428` the reward's own `capture` kernel carries |
| capture point | ξ = F·t/(m·ω₀) = **47.626 · F mm** at the declared t = 0.12 s |

**Two corrections to constants in `script.py`, both load-bearing:**

1. **The foot roll-over budget is 134.8 N·mm, not 117.** `script.py:1234`,
   `:1266` and `:2247` compute it as `2.581 N × 45.5 mm`, using a **0.263 kg**
   machine. The compiled model weighs **0.302011 kg / 2.9627 N**, so the budget
   is `2.9627 × 45.5 = 134.8 N·mm`. The capture-point arithmetic at `:2376`
   already uses the correct 0.30201 kg, so only the roll-over figure moves.
2. **`script.py`'s ξ = F × 48.2 mm is 1.2 % high**, because it uses a stale
   144.210 mm CoM height (ω₀ 8.248). Against the compiled model it is
   **47.626**. Both are quoted here so the discrepancy is on the record.

---

## 4. Task

Both arms train on **`tasks/stand-b8-clamp25-quiet/`**, derived from
`tasks/stand-b8-clamp25/` by `make_arm_bundle.py`.

| | |
|---|---|
| bundle sha256 | **`5d8dd7c1dfe7be5d39d7b62fa4c80f29667b959c3c9c8827d47b2003b7fb7c01`** |
| parent (the control's bundle) | `3d627ef4b9a509feed62a0a0f2e1f02a75965379c01ad8bf53f89a68a33712d8` |
| model | `80eaa18f…` — **byte-identical, copied not rebuilt** |
| label | `stand15` |
| episode | 6.0 s, 300 steps, 50 Hz control, 500 Hz solver, `solved` keyframe |
| terminations | `tipped` at `pel_qx² + pel_qy² > 0.15` (**45.57° of tilt**), `collapsed` at `com_z < 70.472` |
| disturbance | **UNCHANGED**: `shove` 0.3–0.8 N over 210–330°, `shove2` 0.3–0.8 N full circle, `wind` 0–0.06 N sustained |
| reward | the control's nine kernels, unchanged, **plus one** |

**The whole treatment is one appended reward term.** Nothing else in the
bundle moves — not the model, not the disturbance, not the observations, not
the action table, not the terminations.

```
quiet = 0.3 · exp( -(( Σ|<joint>_v| ) / 1611.0)² )
```

over all ten actuated joints, in bundle order. It is generated by substituting
into `posture`'s own expression — `_a` → `_v`, and its σ → this one — so the
spelling, the spacing and the joint order are provably identical to a term the
engine already accepts rather than retyped.

**No observation is added.** The ten `*_v` channels are already declared
(`obs/19`–`obs/28`, adr 38–47), so the policy's input vector is unchanged at
**58 of 64** channels and the reward at **10 of 16** terms. That is also what
makes the cross-bundle scoring in §2 sound: all three bundles publish the
identical 58 channels under identical names, and `policy_forward` orders by the
*policy header's* list.

### 4a. Where σ = 1611.0 comes from

`swirl_scale.py`'s rule, which is where every other σ in this reward came from:
**the median of the quantity over the regime the term operates in**, so that
half the samples sit above it and half below and the term is a *gradient*
rather than a constant. Measured over **4454 settled frames pooled across both
control seeds** (`stand12` s1 and s2 at iteration 1750, 12 evaluation seeds
each, declared disturbance):

| | |
|---|---|
| `stand12` s2 | median settled Σ\|q̇\| **1630.67** deg/s |
| `stand12` s1 | median settled Σ\|q̇\| **1596.34** deg/s |
| **pooled** | **1610.99 deg/s** → σ = **1611.0** |

The two seeds agree to 2.1 %, so the fourth digit is not meaningful and the
value is robust. What the kernel then reads:

| at | Σ\|q̇\| | `quiet` |
|---|---|---|
| the zero-action servo | 99.2 deg/s | **0.9962** — a still machine pays essentially the full weight |
| the control itself | 1611.0 | **0.3679** — 1/e by construction, a real gradient |
| half the control | 805.5 | **0.7788** |

**This rule inverts `swirl_scale.py`'s own convention, deliberately.** That
driver measures its scales over the *recovery* regime only, excluding the
frames where the machine is merely standing, because folding those in halves
every scale. `quiet` is the one term whose entire subject **is** the machine
standing still, so its σ must come from exactly the regime the others throw
away.

> **The plan pre-registered two σ rules and they conflict.** Beside
> "`swirl_scale.py`'s method" it also said *"σ = 2× the settled Σ|q̇| of the
> zero-action servo"*, which gives **σ ≈ 158**. At that scale the kernel reads
> `exp(−(1857/158)²) = exp(−138) ≈ 0` at the control's actual behaviour: a dead
> constant with no gradient anywhere the machine ever is — precisely the failure
> `swirl_scale.py`'s docstring exists to prevent. The median rule was taken and
> the conflict is recorded rather than quietly resolved.

### 4b. Why there is no band arm — the capability sweep vetoed it

The plan's arm S was to raise `newtons_high` 0.8 → **1.8 N** so the capture
point would be asked to leave the support polygon. It pre-registered the rule
that decides the number: ***`capability`'s sweep sets it, and the arithmetic
only proposes the candidate*** — put the top where the incumbent survives
~50 %. Measured on `stand13.001800`, 12 seeds, 7 scales, **70 s of CPU**:

| scale | top N | survival | termination mix |
|---|---|---|---|
| 0.50 | 0.40 | **100 %** | survived 12 |
| 1.00 | 0.80 | **66.7 %** | collapsed 3, survived 8, tipped 1 |
| 1.14 | **0.91** | **~50 %** | ← *the rule's answer* |
| 1.50 | 1.20 | **8.3 %** | collapsed 8, survived 1, tipped 3 |
| 2.00 | 1.60 | **0 %** | collapsed 8, tipped 4 |
| 2.25 | **1.80** | **0 %** | ← *the plan's number* |
| 3.00 | 2.40 | **0 %** | collapsed 2, tipped 10 |

The sweep's answer is **0.914 N**, and the plan's 1.8 N sits where the
incumbent survives **0 of 12** — the same shape of band `script.py:2277`
records B3 retracting, twice, as *"OUT OF RANGE, AND THAT IS WHY NOTHING
STOOD."*

**But the finding that actually killed the arm is the corollary.** The
*current* top of 0.80 N already sits at 66.7 % survival, so by B3's own
construction the band is **already nearly optimally placed** and 0.914 N is
only 14 % away. Raising it is not a lever; it is a rounding error with a
survival cost. The operator's call was to spend both slots on this arm instead
— which also repairs the specific mistake 004 made, where the fork bought a
third *treatment* rather than a second *seed* and criterion 4 went unmet.

**And the premise the band arm rested on was wrong anyway.** The plan said the
task *"almost never requires a step"*; `script.py:2448` says **58 %** of aimed
shove draws demand one. Both are wrong, and they are wrong for the same
reason. Computed against the compiled model's real support polygon — the
convex hull of all four sole geoms, with the ray-to-edge distance taken from
the CoM in every direction of the declared arc:

| | from the CoM |
|---|---|
| forward (az 90°) | **40.83 mm** |
| backward (az 270°) | **29.17 mm** |
| lateral (az 0°) | **50.00 mm** |

`script.py` used the profile's **24.5 mm** as the backward margin — but 24.5
is measured from the **ankle bracket**, and the CoM sits 5.96 mm forward of
it — *and* treated the whole ±60° arc as though every draw faced the narrow
heel, where the margin at ±60° off-axis is **57.7 mm**, nearly double. The
backward step boundary is **F = 0.612 N**, and the honest step-demand
fractions are:

| band | `shove` (210–330°) | `shove2` (full circle) |
|---|---|---|
| **[0.30, 0.80]** — current | **17.4 %** | **5.8 %** |
| [0.30, 0.90] | 26.6 % | 9.4 % |
| [0.30, 1.20] | 47.8 % | 26.3 % |
| [0.30, 1.80] | 68.5 % | 55.5 % |

So the task demands a step in **17.4 %** of first-shove draws, not 58 % and
not "almost never". **This is ADR-107's lesson at one more level down: three
numbers in a profile were quoted in three different frames**, and every
consumer that compared them to a CoM-relative quantity was wrong by 4.7 mm and
by a factor of two off-axis. `harness/profiles/mg-legs.json` now says so.

### 4c. Cross-bundle scoring — proved by running it

Before anything was designed around it, on sb1x 2026-08-06:

| | |
|---|---|
| `harness steps --policy stand13.001800 --task <a foreign bundle>` | **returns rows** — `both 1/4`, `longest_step_mm 57.81` |
| `harness capability`, same pair | **refuses**, `verify_refused`, rc 1: *"was trained on a task bundle whose digest is '3d627ef4…', and the task it is declared against digests to '11d122e1…'"* |

`harness/_episodes.py:347` is the **only** `verify_policy` call in the
repository. `steps`' job spec has no `verify` key at all, so it never reaches
it; `capability.py:255` and `compare.py:201` pass one unconditionally.

### 4d. Why these are derived bundles and not a script revision

The plan preferred authoring from `mechanisms/mg-legs/script.py`, since
ADR-131 made `command_limits_degrees` a first-class kwarg and 004's blocker is
gone. It attached a condition: **the MJCF must come out byte-identical to
clamp25's `80eaa18f…`; verify it rather than assume it.**

Verified, by rebuilding `rollout/script-clamp25.py` with its policy block
removed. **It does not.** The script-built bundle differs in exactly three
places:

| | |
|---|---|
| `actions[*].source` | `angle_limits_degrees` → `command_limits_degrees` — inert, every action *number* identical |
| `label` | set by the caller |
| `model.sha256` | **`80eaa18f…` → `203f746e…`**, bytes 14179 → 14169 |

The model difference is one attribute of one line — the pelvis inertial x,
`5.10066e-11` → `0`. That is **ADR-133's inertial snap**, which landed *after*
clamp25 was authored, and the quantity is mathematically zero by symmetry.
Numerically negligible; still disqualifying, because **the control is already
paid for** and was trained against `80eaa18f…`. Authoring from source would
move the physics in the same step as the reward, and no result could then be
attributed to either — CLAUDE.md invariant 2's reasoning, applied to a model
change rather than a MuJoCo version.

**What that costs, plainly: criterion 5 is not met by this bundle.** It is a
claim about committed bytes. The route that would meet it exists and costs one
extra 5 GPU-hour run — re-derive the arm *and* the control on `203f746e…` —
and that is a budget decision, not a tooling one.

---

## 5. Gate — Phase A, on CPU, in minutes

`feasibility.py` is **not** re-run: the MJCF is byte-identical to the one 004
and 005 gated, and 005 sets that precedent. What was run instead is the
instrument and the pre-flight, all on CPU, none of it contending with the card.

**The instrument fires on purpose before it is believed.**
`tools/fire_jitter_guard.py` plays three synthetic controllers through the same
model and bundle:

| controller | steps | mean Δ deg | rev/s | Σ\|q̇\| deg/s |
|---|---|---|---|---|
| `hold` — the nominal pose | 300 | **0.0000** | **0.00** | 4.60 |
| `chatter` — corners, alternating every step | 28 | 48.00 | **48.15** | 2767.69 |
| `sweep` — a slow full-amplitude triangle | 8 | 0.96 | **0.00** | 978.16 |

`sweep` is the case that matters: **full amplitude and near-zero reversals**, so
a driver reading magnitude alone would call smooth tracking "jitter". The
ceiling is `1/interval_s` = **50 /s**, not 25 — a full oscillation carries two
direction changes, and the first draft printed 25 and then measured 38.6
against it, which is the kind of impossible reading that means the yardstick is
wrong rather than the machine.

Plus 21 pure-half tests under cdx-rl's own interpreter with no mujoco
(`mechanisms/mg-legs/drivers/test_jitter.py`).

### 5a. What Phase A measured — the four gates, stated before they were run

**Gate 1 — is the jitter in the joints at all?** *If settled `Σ|q̇|` is not
materially above the zero-action servo's, arm Q is dropped.*

| | settled Σ\|q̇\| mean | mean Δ cmd | rev/s |
|---|---|---|---|
| zero-action servo | **99.20** deg/s | 0.000 | 0.00 |
| `stand12` s2 (control) | **1522.33** | 17.535 | 36.45 |
| `stand12` s1 (control) | **1573.88** | 19.043 | 37.82 |
| `stand13.001800` (extended) | **1856.90** | 21.722 | 37.80 |

**PASSES, by 15–19×, and it replicates across both control seeds.** The raw
command trace shows why: the policy commands a **full-amplitude square wave**
on a ±25° joint — `+23.8 → −24.85 → +22.25 → −24.81 → +22.87 → −24.86` — at
37.8 reversals/s against a 50 /s ceiling. This is not exploration noise;
rollout inference is the deterministic tanh mean.

**Gate 2 — is it the setpoint or the servo?** Pre-registered: *headroom below
30 % means the jitter is the servo and the contacts, and the arm is aimed at
the wrong thing.*

| | closed loop | frozen setpoint | headroom |
|---|---|---|---|
| `stand13.001800` | 1872.61 | **124.95** | **93.3 %** |
| `stand10.001700` | 1538.55 | **143.16** | **90.7 %** |

**PASSES by a wide margin.** Freezing the setpoint at its settled mean —
same seed, identical reset draw and disturbance schedule — collapses Σ|q̇| by
**15×**. The jitter is the policy's setpoint. A control check runs beside it:
`--open-loop replay` reproduces the closed-loop RMS to the digit (1872.61 vs
1872.61), which is the determinism the whole comparison rests on.

**Gate 3 — is the sole slipping?** *If slip while planted exceeds 20 % of foot
length per contact-second, that is a model change and a re-plan, not an arm.*
Foot length is 70 mm in the foot frame, so the threshold is 14 mm/contact-s.
Measured on `stand13.001800`: **42.7 and 33.2 mm per contact-second.**

> **This gate is reported as INCONCLUSIVE rather than fired, and the reason is
> the instrument's own stated caveat.** The `sample` hook fires at the 50 Hz
> control rate, not the solver's 500 Hz, so a tangential speed is integrated as
> if constant across a 20 ms interval. Under a setpoint reversing 37.8 times a
> second, that integral is biased upward by an unknown factor — it is
> integrating a square wave with a sampler synchronous to it. The honest
> reading is that slip **cannot be sized from this instrument at this rate**,
> and since gate 2 already routes the finding to the setpoint, the arm is not
> re-planned on it. **If `quiet` lands and the fidget falls, re-measure slip
> then**: a quiet machine's slip integral is trustworthy where a chattering
> one's is not.

**Gate 4 — is the termination threshold what stops the recovery?** `tipped`
fires at **45.57°** of tilt. At the *current* band's top, ξ = 38.1 mm against a
141 mm CoM, which is a lean of `atan(38.1/141)` = **15.1°** — a third of the
threshold. `final_tipped_mean` at the declared band is **0.0476** against the
0.15 that terminates. **Not binding.** The band is unchanged in this
experiment, so this gate has nothing left to veto.

### 5b. The control, measured at 24 seeds on `tasks/stand-b8-clamp25/`

| | `stand12` s2 | `stand12` s1 |
|---|---|---|
| survived | 18/24 | 18/24 |
| stepped | 20/24 | 20/24 |
| **conjunction** | **15/24** | **15/24** |
| lifts / surviving episode | 5.78 | 8.33 |
| `longest_step_mm` **median** per episode | 28.54 | 31.57 |
| `longest_step_mm` **max** (the published statistic) | 51.14 | 52.03 |
| settled Σ\|q̇\| | 1522.33 | 1573.88 |
| `hazard15` settled mean % of rating | 53.5 % | 51.4 % |
| `hazard15` settled duty > 90 % | **13.88 %** | **13.28 %** |

`stand12` is the **matched** control: 1800 iterations from scratch on
`clamp25`, which is what each arm here will be. `stand13` is a warm-started
continuation of `stand12` s2 (3600 effective iterations) and is the
extended-training reference, not the control.

> **A trap found while doing this, and it cost a table.** `harness steps` keys
> its `results` dict by the policy's **basename**, so scoring two seeds of the
> same label in one invocation silently *sums* them — the two `stand12` seeds
> came back as `survived 36/24`. The `36/24` is the only tell. Score
> same-named checkpoints in **separate invocations**, and read the denominator.

### 5c. And the shuffling itself, quantified

The behaviour that prompted the experiment, at 24 seeds on the same bundle,
both arms at the same 18/24 conjunction:

| | `stand13.001800` (clamped) | `stand10.001700` (003 s2, unclamped) |
|---|---|---|
| lifts / surviving episode | **9.32** | 6.47 |
| step events / surviving episode | **5.53** | 3.68 |
| `longest_step_mm` median per episode | **31.54** | 44.79 |
| `longest_step_mm` max | 50.14 | 127.08 |

**44 % more lifts and 50 % more step events, each 30 % shorter in median.**
"Many small steps instead of one big one" is a measurement, not an impression.
The clamped swing reach is 98.2 mm and the machine achieves 31.5 mm median —
so the ceiling is not the mechanism.

---

## 6. Budget and stopping rule

**Two runs, both this arm, ~10 GPU-hours on sb1x. Serial — there is one card.**

| | |
|---|---|
| arm Q seed 2 | `stand15`, 1800 iterations, ~5.0 h |
| arm Q seed 1 | `stand15`, 1800 iterations, ~5.0 h — **the replication, dispatched unconditionally** |
| control | **already paid for** — `stand12` seeds 1 and 2. No new control run. |

**The second seed is unconditional and that is the design.** 004's fork spent
its second slot on a third treatment and criterion 4 went unmet, so every
number in it was n = 1 in seeds. A gate that could stop after one seed would
reproduce exactly that. If seed 2 produces nothing, seed 1 is still a fresh
seed and is still worth its 5 hours.

Every arm is a **from-scratch** run. `--init-from` cannot be used: the trainer's
`check_policy_fits` compares the task bundle's **whole-file** digest
(`cadex_train.py:307-309`), ADR-134's semantic-equivalence path lives in
`verify_policy` and not in the trainer, and a reward change is a semantic
change in any case.

```
--iterations 1800 --checkpoint-every 50
--envs 2048 --unroll 40 --epochs 5 --hidden 64 64
--learning-rate 3e-4 --discount 0.99 --gae-lambda 0.95
--clip 0.2 --entropy 2e-3 --value-weight 0.5 --initial-std 0.4
--require-device gpu --timeout 25200 --patience 0 --supervise --detach
```

**All fourteen passed explicitly.** `--discount 0.99` and `--gae-lambda 0.95`
are the two `tools/train.py` would otherwise fill from `RUN_200109` as
**0.995** and **0.97** — the silent substitution that killed 005's first
dispatch.

`--require-trainer bb133b64d57d8f2b521c22b1111e182428ef70e4f2088a5e7cee945a0ec71dc2`,
verified in this session by `sha256sum /home/theo/cadex-prs/training/cadex_train.py`
and pasted from that command's output. **Provenance:** 003/004 ran under
`aacfa823…`, 005-ceiling under `4c1f24f8…`, and `main` is now `bb133b64…`,
which differs from `4c1f24f8…` by one docstring line (`ADR-123` → `ADR-131`).
**No bridge run is owed.** The control `stand12` ran under `aacfa823…`, so the
arm-vs-control comparison crosses a trainer change and is reported as measured
behaviour through the paired test — which `method.md` §8b permits as the
reading least exposed to it.

**Stopping rule:** `--patience 0`. The trainer's scalar is not a proxy for what
matters and this bundle makes that structural — `quiet` moves the total weight
to 5.6, so `reward_per_step` is not even comparable to the control's. Watch
**episode length beside reward** (ADR-099).

**Dispatch is [`chain-006-quiet.sh`](chain-006-quiet.sh)**, beside this file
rather than in `jobs/`, because `jobs/` is gitignored in full and the script is
the executable form of §§6–7 — 004's chain script lived there and was never
committed. It refuses to start if the bundle's digest is not the
`5d8dd7c1…` pre-registered above, dispatches both seeds, scores them on the
common bundle, and prints Q1 and Q3 against the control's Phase A literals
without deciding anything.

---

## 7. Pass criteria, written before the runs

| | |
|---|---|
| **Q1** | settled `Σ\|q̇\|` falls **≥ 40 %** against the matched control at the same checkpoint index — i.e. **≤ 913.4 deg/s** against `stand12` s2's 1522.33, and **≤ 944.3** against s1's 1573.88 |
| **Q2** | conjunction on `tasks/stand-b8-clamp25/` at 24 seeds **not significantly worse** than the control's 15/24, paired **McNemar p > 0.05** |
| **Q3** | `hazard15` settled duty > 90 % of rating stays **< 25 %** (004's criterion 2) |
| **Q4** | **both seeds agree in direction on Q1.** One seed answers nothing; this is the criterion 004 failed to meet |

**Q3 is a falsifier, not a decoration.** A quieter machine bought with bracing
fails the experiment — that is the whole point of the clamp holding.

**Q2 is deliberately a non-inferiority test, not a superiority one.** `quiet`
is not expected to improve stepping and might cost it. Every positive kernel
stops paying during a real swing, so `quiet` taxes stepping too, and if the
machine ends up quieter *and* stepping less that is a **real trade-off
finding** about a reward with no velocity term — not a failure. It is reported
as one.

**What would falsify the premise itself:** if settled `Σ|q̇|` is **unchanged**
in both seeds, then a 0.3-weight kernel at 1/e of the fidget cannot move it,
and the finding is that the jitter is worth more to the policy than 0.3 of 5.6
— which is a statement about the other nine terms and is worth writing down.
Do not conclude "the machine cannot stand still".

---
---

## 8. What happened

Both seeds ran to completion on sb1x, 2026-08-06, **9.67 GPU-hours**
(`stand15` s2 4.86 h, s1 4.80 h), both `iter 1799`, both `state done`, both
with all 37 checkpoints. No crash, no salvage. Scored at the matched operating
checkpoint **`001750`** — the same index the control is quoted at — on the
common bundle `tasks/stand-b8-clamp25/`.

### The four criteria

| | seed 2 | seed 1 | |
|---|---|---|---|
| **Q1** settled Σ\|q̇\| | **466.5** vs 1522.3 → **−69.4 %** | **725.1** vs 1573.9 → **−53.9 %** | **PASS ×2** |
| **Q2** conjunction, paired | 14/24 vs 15/24, 5 discordant (2/3), **p = 1.000** | 10/24 vs 15/24, 7 discordant (1/6), **p = 0.125** | **PASS ×2** |
| **Q3** hazard15 duty > 90 % | **0.98 %** | **45.19 %** | **seed 2 PASS, seed 1 FAIL** |
| **Q4** both seeds agree on Q1 | — | — | **PASS** |

### Q1 — the kernel works, and it replicates

| | control | arm Q | |
|---|---|---|---|
| settled Σ\|q̇\| seed 2 | 1522.33 | **466.48** | −69.4 % |
| settled Σ\|q̇\| seed 1 | 1573.88 | **725.10** | −53.9 % |
| command Δ per step, seed 2 | 17.535° | **5.816°** | −66.8 % |
| reversals/s, seed 2 | 36.45 | **29.26** | −19.7 % |

Both seeds clear the pre-registered 40 % bar. The zero-action servo floor is
99.20 deg/s, so seed 2 closed **74 %** of the gap between the control and a
machine with no network in it at all.

### Q3 — and this is the finding

**The bracing criterion does not replicate, and seed 1 fails it decisively.**
At the operating checkpoint, 12 evaluation seeds, nothing pushing:

| | settled mean % of rating | duty > 90 % |
|---|---|---|
| control s2 | 53.5 % | 13.88 % |
| control s1 | 51.4 % | 13.28 % |
| **arm Q seed 2** | **38.6 %** | **0.98 %** |
| **arm Q seed 1** | **73.4 %** | **45.19 %** |

Seed 2 got quiet *and* got **less** braced than the control on both statistics.
Seed 1 got quiet by **bracing harder than the control** — 73.4 % of rating
against 51.4 %, and a duty of 45.19 % against a 25 % ceiling.

**It is not an artefact of the summary statistic, which was the first thing
checked.** The chain reader takes the max over the whole checkpoint series, so
the obvious suspicion was that it had caught an early-training excursion. It
had not: seed 1's **worst checkpoint is 1750 itself**, and the duty is
*climbing monotonically* into the end of the run —

```
seed 1 duty:  1500  5.4 %   1550  9.6 %   1600 11.0 %
              1650 14.1 %   1700 28.2 %   1750 43.0 %      (6 eval seeds)
seed 2 duty:  1500  6.9 %   1550  6.9 %   1600  1.3 %
              1650 10.6 %   1700  5.3 %   1750  2.1 %
```

Seed 2's series max is 21.36 % at iteration **1400**, an excursion it recovers
from. Seed 1's is its own endpoint. The two runs are doing different things.

The Σ\|q̇\| series says the same thing in the quantity the kernel actually
optimises — seed 1 is still *rising* where seed 2 has flattened:

```
seed 2 Σ|q̇|:  250 166  500 338  750 347  1000 449  1250 460  1500 511  1750 466
seed 1 Σ|q̇|:  250 156  500 135  750 331  1000 338  1250 412  1500 696  1750 725
```

### Q2 — passes, and the point estimate is worth reading anyway

Seed 1's conjunction is **10/24 against the control's 15/24**. The paired test
gives **p = 0.125** on 7 discordant seeds split 1/6, so it clears the
pre-registered `p > 0.05` bar — but the bar is a non-inferiority bar and the
point estimate fell by a third. **The honest statement is that Q2 did not
detect a loss at n = 24, not that there was none.**

> **Which checkpoint the conjunction is quoted at changes the number, and §2
> and §7 do not agree.** §2 specifies *"the best late checkpoint"*; Q1 in §7
> specifies *"the same checkpoint index"* as the control. The table above uses
> the **matched index 001750** throughout, so that every criterion is read off
> one policy. Scored instead as best-over-series the conjunctions are **15/24
> (seed 2)** and **13/24 (seed 1)** — both better, neither changing a verdict.
> Reported both ways rather than picking the flattering one.

Seed 1 also shows the "fewer and bigger" shape 006 was originally chasing, and
in the arm that was not trying to buy it:

| | control s1 | arm Q seed 1 |
|---|---|---|
| lifts per surviving episode | 8.33 | **3.72** |
| median longest step | 31.57 mm | **37.18 mm** |

Seed 2 is flat on both (5.74 vs 5.78 lifts; 30.72 vs 28.54 mm).

---

## 9. What it means, and what it does not mean

### What it means

**A velocity kernel reliably reduces the fidget.** Q1 replicates at two seeds
by 69.4 % and 53.9 %, on a quantity no existing reward term could see, sized by
`swirl_scale.py`'s median rule off the control's own settled distribution. That
part of the premise is confirmed: the ten `*_v` channels were already declared,
the term cost one line, and it moved the thing it was pointed at.

**How the policy pays for that quiet does NOT replicate, and that is the
result.** Seed 2 paid with better posture — less bracing than the control on
both statistics. Seed 1 paid with **more** bracing than the control. Same
bundle, same digest, same hyperparameters, same trainer, same 1800 iterations;
only the seed differs.

That is mechanically unsurprising in hindsight and should have been
pre-registered as a likely outcome: **`quiet` does not say how to be still.**
Holding a joint still by not commanding motion and holding it still by
stiffening against the servo are the same reading on `Σ|q̇|` and opposite
readings on `actuator_force`. The kernel is degenerate with respect to the
thing hazard 15 measures, and PPO found both basins from different seeds.

**Q3 is a pre-registered falsifier and it fired.** §7 says *"a quieter machine
bought with bracing fails the experiment — that is the whole point of the clamp
holding."* One of two seeds did exactly that. **`quiet` as written is not
adoptable into the reward.**

### What it does not mean

* **It does not mean the kernel is wrong** — it means it is *underspecified*.
  A `quiet` term paired with something that sees torque (the existing `effort`
  kernel is the obvious candidate, at 0.2 weight against `quiet`'s 0.3) may
  well separate the two basins. That is a next experiment, not a conclusion.
* **It does not mean seed 2's result is the real one and seed 1 is noise.**
  With n = 2 there is no basis for preferring either. 002's lesson is exactly
  this and it was the reason the second seed was made unconditional.
* **It does not license quoting −69.4 %.** The two-seed range is **−69.4 % to
  −53.9 %**, and the seed that fidgeted less is the seed that braced more.
* **It says nothing about stepping**, which was 006's original subject before
  Phase A vetoed the band arm. Seed 1's 8.33 → 3.72 lifts at a longer median
  step is suggestive of exactly the fewer-and-bigger behaviour arm S was meant
  to buy, but it arrives confounded with a bracing failure in a single seed and
  should not be reported as a stepping finding.
* **The control comparison crosses a trainer change** (`aacfa823…` →
  `bb133b64…`, one docstring line), which is why every comparison above is
  read through the paired test rather than a difference of point estimates.

### Two defects in this experiment's own instruments, found while reading it

1. **The chain's criteria reader printed a fall as `+69.4 %`.** It computes
   `100 · (1 − q/base)`, which is positive for an improvement, and labels it
   `->`. Read literally it says the jitter went *up*. No conclusion was drawn
   from it — the sign was checked against the raw numbers — but a reader
   arriving at the log alone would get it backwards.
2. **It reads Q3 as the max over the whole checkpoint series** where §7 means
   the operating point. On seed 2 those differ by 20 pp (21.36 % over the
   series against 0.98 % at 1750) and the verdict happens to survive; on any
   run with a mid-training excursion it would not. **Q3's verdict here is
   unaffected** — seed 1's worst checkpoint is its operating checkpoint — but
   the reader agrees with the protocol by luck rather than by construction.

Both are fixed in `chain-006-quiet.sh` after the fact, and the fix is recorded
here rather than silently applied, because the numbers in §8 were read off the
**hand measurements at 12 seeds**, not off the chain's log.
