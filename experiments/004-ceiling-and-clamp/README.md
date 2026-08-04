# Experiment 004 — the ceiling, and whether a buildable policy can step

**Written before dispatch.** ADR-097: a metric chosen after looking at the
curve is a metric chosen by looking at the answer. Nothing below is edited
after the fact; where a run departs from it, §8 says so and says why.

## 1. Question

Experiment 003 at four seeds left two findings pointing in opposite
directions, and this experiment runs one arm at each.

**A — the ceiling.** All three 1800-iteration seeds peaked at iteration
**1700–1750**, the last two checkpoints written, and the pooled paired test
put the late-beats-early direction at **8 discordant to 1, p = 0.0391**. The
runs stop mid-climb. *Where does improvement actually stop?*

**B — buildability.** The same policies hold the worst motor at **73–91 % of
the 86 N·mm rating on average, with nothing pushing**, above 90 % of it for
**49–77 % of frames**. Hazard 15 did not dissolve; it moved from the network's
output to the servo's. *Can a policy that cannot saturate its motors still
step?*

B is the one that matters. A better policy that cannot be built is not
progress, and A alone would produce exactly that.

## 2. Metric

`harness steps`, the conjunction — **stepped ≥ 10 mm AND survived 300/300** —
at 12 evaluation seeds, escalating to 24 for ties, read through **McNemar over
the discordant seeds** and not the point estimate. Plus
`mechanisms/mg-legs/drivers/hazard15.py` for the resting torque, which is the
instrument 003 lacked.

## 3. The two runs

Both **2500 iterations, checkpoint every 50** — 50 periodic checkpoints, at
the measured **9.95 s/iteration** on `sb1x`, so **~6.9 h each, ~13.8 h
serial**. There is one card now: `sb9x` was retired 2026-08-04.

| | task | bundle sha256 | seed |
|---|---|---|---|
| **A ceiling** | `tasks/stand-b8/` unchanged | `5572adf265aa51cb…` | 2 |
| **B clamp15** | `tasks/stand-b8-clamp15/` | `50e2d08bcb6b0785…` | 2 |

**Seed 2 for both.** It was 003's best seed (`001700` at 18/24), so A's first
1800 iterations can be read against a run whose shape is already known — 002
measured same-seed runs at **r = +0.9885** in shape while being **0 of 1500
bitwise identical**, so this is a consistency check on shape, never on a
value. And B shares the seed with its own control, which is what makes the
comparison paired at the level that matters.

**A is the control for B.** Running them at the same length, same seed, same
everything-but-the-action-range is the whole point; 003's lesson was that
moving four things at once makes attribution an argument rather than a
measurement.

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

## 5. Provenance of the clamped bundle, and its one weakness

`tasks/stand-b8-clamp15/` is derived from `tasks/stand-b8/` by
`make_clamp_bundle.py` in this directory — the cap applied programmatically,
never by hand.

**It is not reproducible from `script.py`, and that is a real cost.** This
project's premise is that the script is the source of truth. Two things
prevent it here: the ranges come from `angle_limits_degrees` in the mechanism,
so capping them properly is a script edit; and **`sb1x` cannot build
`script.py` at all** — its pinned Cadex `06d1374b` rejects
`centre_of_mass_velocity`, and only `mmini` (Cadex `560935bd`) accepts the
whole observation set. Authoring the clamp on the laptop and carrying the
bundle back is the correct fix and is deferred, not forgotten.

So 004-B is a claim about **committed bytes**, exactly as 001 and 002 are for
`stand-b2`. It is dispatchable and comparable; it is not yet re-derivable.

## 6. Budget and stopping rule

~13.8 h across the two, serial, on `sb1x`. `--timeout 32400` (9 h) per run —
2500 iterations needs ~6.9 h and the cap must not bite before the question is
answered; 003's 25200 would truncate at ~iteration 2530, which is close enough
to the target to be worth avoiding.

`--patience 0`. Reward patience stays off — 001 found no reward rule would
have found the right checkpoint, and 003's four seeds put the trainer's
reward-best at 4 of 4 never better than the run's best.

All fourteen hyperparameters passed explicitly, equal to 003's:
`envs 2048, unroll 40, epochs 5, hidden [64,64], lr 3e-4, discount 0.99,
gae_lambda 0.95, clip 0.2, entropy 2e-3, value_weight 0.5, initial_std 0.4`.
Note these differ from `train.py`'s `RUN_200109` defaults in two —
`discount` and `gae_lambda` — which is the silent-substitution trap.

## 7. Pass criteria — written before the runs

1. **A finds the ceiling, or shows there isn't one yet.** The best checkpoint
   in (1800, 2500] versus the best in [0, 1800], by McNemar at 24 seeds. If
   late still wins, 2500 is *also* too short and that is the finding.
2. **B steps at all.** Its best checkpoint scores ≥ 6/12 on the conjunction —
   B6's baseline, the number 003 was measured against. Below that, the clamp
   is too tight to control the machine.
3. **B is measurably less braced.** Resting duty cycle above 90 % of rating
   **below 25 %**, against A's 49–77 %. This is the point of the experiment;
   a B that steps as well as A while still saturating has taught us nothing.
4. **The comparison is paired and reported as such.** A-vs-B on the
   conjunction through McNemar, not through the point estimates, and the tied
   set named.

**Stated in advance: the likely outcome is a trade.** B steps worse and braces
less. If so the deliverable is the *curve* between them, not a winner, and the
next question is where on it a buildable machine sits — which is an actuator
sizing question, not a policy one.

## 8. What happened

*(to be written after the runs)*
