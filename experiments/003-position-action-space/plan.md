# The B8 plan, as written before dispatch

**This file is evidence, not documentation.** It is the plan for what this
repository calls experiment 003, reproduced verbatim from the working
directory `~/cdx-mjc` where B8 was run on 2026-08-03. It is committed
unedited — including the parts that turned out to be wrong — because
`method.md`'s central rule is *decide the success metric before you
dispatch* (ADR-097), and the only way to show that a metric was chosen before
the answer was seen is to keep the document that chose it.

Where the run departed from this plan, `README.md` §8 says so and says why.
Nothing in this file has been corrected after the fact.

The research note that motivated it is [`research-note.md`](research-note.md),
written the same day and also unedited.

---

# B8 / `stand10` — the literature's action space, reward sign, and control rate

## Context

Nine runs (m9a/b/c, a1c, B1–B7) have moved the **disturbance** five times and
the **reward** three times. Not one has moved the action space, the control
rate, or the sign convention of the reward — and those are the three things
the legged-RL literature is most consistent about. B7 was the run that made
this impossible to ignore: it came back a measured regression (B6's policy
beats every B7 checkpoint on B7's own task, 7/12 stepped against 3/12), after
two probes that failed for reasons the plan could not have reasoned its way
to.

The research note is `~/cdx-mjc/RESEARCH-why-this-is-hard.md`, written today
with sources. Its finding, in one line: **our zero action falls over and
theirs stands.**

Our ten actuators are `kind="motor"` (`script.py:1138`), so the network
outputs joint torque; the trainer maps a network output of 0 to the midpoint
of the action range (`cadex_train.py:449`), which for a ±86 N·mm motor is
zero torque; and `feasibility.py` check 5 measures zero torque as *"falls at
0.976 s"*. So PPO must discover continuous gravity compensation for ten
joints before it can learn anything about balance. In the literature the
action is a position offset from a nominal pose held by a PD servo, so the
untrained policy already stands and the policy learns only the deviation.

Three supporting gaps, all confirmed in our code today: no reward clipping
anywhere in `cadex_train.py` (we measured the "suicide policy" it prevents,
twice); 100 Hz control, which doubles every credit-assignment horizon
measured in steps; and no warm-start flag, which is the real reason ADR-100
concluded a curriculum was unavailable.

**B8 changes four things and every one is script-side.** The engine already
supports PD position actuators (`CadexDynamics.py:3285`, `set_to_position`)
and the expression vocabulary already has `exp`.

---

## 1. The four changes

### 1a. Position actuators — the action becomes a pose, not a torque

Replace the `motor()` helper in `mg-legs.cadex/script.py` with:

```python
def servo(joint, limit, label):
    return assembly.actuator(joint, kind="position", control_deg="0",
                             stiffness_nmm_per_deg=KP_NMM_PER_DEG,
                             damping_nmms_per_deg=KD_NMMS_PER_DEG,
                             torque_limit_nmm=limit, label=label)
```

`torque_limit_nmm` stays 86 — the MG90S judgment is unchanged; what changes
is who computes the torque.

### 1b. A crouched nominal pose, and limits symmetric about it

**The key fact, verified today:** the assembly is *drawn* at the posed
configuration (`leg_turns` → `posed_place` → `revolute`, `script.py:687`),
so the joint's zero **is** whatever pose the `params(...)` defaults describe.
Observed joint angles at the reset keyframe are ~0 today for exactly this
reason. So:

- the crouch is set by changing the **param defaults**, not the geometry;
- the limits only need to be **symmetric about zero** to centre the action
  range on the nominal pose.

Crouch (see §2 Q1): hip pitch 15°, knee 30°, ankle 15° dorsiflexion, both
legs, roll joints at 0. Limits become:

| constant | now | B8 | absolute range about the crouch |
|---|---|---|---|
| `HIP_ROLL_LIMITS` | [−30, 45] | **[−30, 30]** | unchanged, ±30 |
| `HIP_PITCH_LIMITS` | [−60, 90] | **[−45, 45]** | 15° ± 45 |
| `KNEE_LIMITS` | [−5, 130] | **[−30, 30]** | 0°–60° flexion, never hyperextends |
| `ANKLE_LIMITS` | [−40, 40] | **[−25, 25]** | inside the ±40 mechanical stop |
| `ANKLE_ROLL_LIMITS` | [−20, 20] | unchanged | already symmetric |

The `num(...)` slider ranges move with them.

**This is the one step that can silently produce a broken machine**, so it
is verified before anything else: `feasibility.py` check 4 prints every
collision geom's bottom z, and all four sole geoms must sit at z ≈ 0. A sign
error in the crouch chain stands the machine on an edge and everything
downstream would be measuring a different robot.

Re-measure with `measure.py` and update, in the one block that holds them:
`Z0`, `X0`, `Y0`, `FX0/FY0` (foot centroid), `OMEGA0 = sqrt(g/Z0)`. The
`collapsed` termination is written `0.5 * Z0` and follows automatically.
`posture` and `splay` stay as written — joint zero is the nominal pose.

### 1c. All-positive reward — the suicide mode made structurally impossible

Every shaping term becomes a bounded positive kernel `w · exp(−(e/σ)²)`, so
the total is always ≥ 0 and terminating is *always* worse than continuing.
No clip, no trainer change, no possibility of the B7 failure recurring.

| term | w | error `e` | σ |
|---|---|---|---|
| `alive` | 0.2 | — | constant, breaks the tie in dead states |
| `upright` | 1.0 | `pel_qx² + pel_qy²` | 0.02 (already quadratic — not squared again) |
| `capture` | 1.5 | `XI` | 50 mm |
| `over_feet` | 1.0 | `OVER` | 40 mm |
| `height` | 0.5 | `com_z − Z0` | 30 mm |
| `arrest` | 0.3 | `abs(cv_x)+abs(cv_y)` | 300 mm/s |
| `swirl` | 0.3 | `abs(cam_x)+abs(cam_y)` | K, from `swirl_scale.py` |
| `posture` | 0.3 | Σ\|joint\| | measure; ~60° over ten joints |
| `effort` | 0.2 | Σ\|τ\| | measure |

Total 5.3 standing, → 0.2 when everything is wrong. `XI` and `OVER` keep
B6's expressions and B6's measured offsets, re-measured at the crouch.

**Why this specifically fixes B7's failure.** `arrest` and `swirl` as
*costs* taxed the swing phase of the recovery they were bought to buy —
lifts unchanged 35 → 33, steps 13 → 2. As positive kernels they simply stop
paying during the swing and resume when the foot lands. A step can never be
made negative by adding a term.

`drift` is **dropped**: it is an unbounded linear anti-wander term with no
natural kernel form, and `over_feet` + `capture` already reference the feet.
Note it as the first thing to reinstate if the machine walks away.

**Saturation caveat for later:** `exp(−(e/σ)²)` is flat for e ≫ σ, the same
defect that split `capture` in two in B7. At B6's band ξ tops at ~39 mm
against σ = 50, so it is well-matched *here*. When the band goes up, capture
needs two kernels again.

### 1d. 50 Hz control

`control_hz=50`, episode still 6.0 s → 300 steps. 0.02 s still divides the
0.002 s solver step exactly (ten substeps, nothing rounded), so M9's original
reason for 100 Hz is preserved. Every horizon measured in steps halves.

### And the band does *not* move

**0.30–0.80 N, B6's exact band.** With four simultaneous structural changes
the task has to be the one we hold a 6/12 baseline on, or the run answers
nothing. Reset variation, both shove windows and the wind stay as B6 had
them. Ambition goes in B9 (§5), which is where it can be attributed.

---

## 2. The three open questions, answered — revisit these first if B8 disappoints

**Q1 — how shallow the crouch? → hip 15°, knee 30°, ankle 15°.**
Chosen so symmetric limits are *mechanically* meaningful: at a straight leg
a symmetric knee range would require hyperextension, and at 30° of nominal
flexion `[−30, 30]` spans 0–60° and never hyperextends. It is also the
shallowest crouch that does that. Ankle 15° keeps `[−25, 25]` inside the ±40
mechanical stop. **To verify, not assume:** `feasibility.py` check 2 gives
the holding torque at the crouch in seconds — if it is a large fraction of
86 N·mm the machine is being asked to hold a squat all day and θ should come
down.

**Q2 — what PD gains? → kp 5.24 N·mm/deg, kd 0.175 N·mm·s/deg
(= 0.3 N·m/rad and 0.01 N·m·s/rad), re-swept at the crouch.**
Check 6 already sweeps six pairs and reports kp 0.3 N·m/rad / kd 0.01 as
standing. Converted to the surface's units that is **5.24 N·mm/deg** and
**0.175 N·mm·s/deg**. The reason to take the *softest* standing pair rather
than the stiffest: at kp 1.0 N·m/rad the 86 N·mm limit saturates at 4.9° of
error, which would make all but the innermost 5° of a ±30° action range
meaningless. At 5.24 N·mm/deg it saturates at 16.4°, so the middle two-thirds
of the range is proportional. **Re-sweep check 6 at the crouch** and take the
softest pair that stands with ≥2× margin on peak effort.

**Q3 — keep the ten `*_tau` channels? → yes, keep. 58 channels, unchanged.**
Under torque control `actuator_force` was nearly a copy of the action the
policy had just emitted. Under position control it is the *servo's load* —
genuinely new information about contact and effort, and more useful than it
was. Dropping channels also costs comparability for no gain.

---

## 3. Implementation order

Everything is in `~/cdx-mjc/mg-legs.cadex/script.py`, edited **through
`rebuild.py`** — never in place, which breaks `cadex params`.

1. Crouch: param defaults + the five limit constants + slider ranges.
   Rebuild. **Stop and run `feasibility.py`** — check 4 (soles at z ≈ 0) and
   check 2 (holding torque) decide whether the pose is right at all.
2. `measure.py` → new `Z0`, `X0`, `Y0`, foot centroid, `OMEGA0`. Rebuild.
3. Check 6 sweep at the crouch → `KP_NMM_PER_DEG`, `KD_NMMS_PER_DEG`.
   Swap `motor()` for `servo()`. Rebuild. **Re-run the whole gate** — a
   position actuator changes what check 2 and check 6 even mean.
4. Reward → the nine kernels. Measure σ for `posture` and `effort` off the
   bundle; re-run `swirl_scale.py` for K at the crouch.
5. `control_hz=50`.
6. Verify before dispatch:
   - standing-pose reward ≈ **5.3** and every term at its maximum (the
     positive-kernel analogue of the hazard-9 check — a term that is *not*
     ~w at the nominal pose is mis-scaled);
   - `reward_decompose.py` on `stand8`'s states: the total must be
     comfortably positive and, unlike every previous run, it **cannot** be
     negative anywhere;
   - `feasibility.py` green.

## 4. Dispatch — sb1x

`~/cdx-mjc/dispatch_b7.sh` generalises to B8 by renaming `NAME` to
`stand10` and pointing `RUN_DIR` at `runs/b8/`. It already refuses to launch
onto a busy card and copies the bundle *and* MJCF beside the checkpoints
(ADR-099 §5).

```
--envs 2048 --unroll 40 --epochs 5      unroll 40 at 50 Hz is 0.8 s per segment
--discount 0.99        2.0 s at 50 Hz — B6's horizon in seconds
--gae-lambda 0.95      credit chain 17.4 steps = 0.35 s
--initial-std 0.4 --entropy 2e-3 --hidden 64 64
--learning-rate 3e-4 --clip 0.2 --value-weight 0.5
--iterations 1200 --checkpoint-every 50 --seed 0
```

**1200, not 2400.** At 50 Hz each control step runs ten solver substeps
instead of five, so an iteration costs ~2× the physics (~8–9 s/iter against
B7's 4.36). 1200 iterations is ~2.9 h **and the same simulated experience as
B6** — 1200 × 2048 × 40 × 0.02 s ≈ 546 h of robot time. 24 checkpoints, B6's
scoring cadence.

`--initial-std 0.4` carries over from B6 numerically but means something
completely different now: it is ±0.4 in normalised action units, so ~±12° of
joint jitter rather than torque noise. Flagged as a B9 lever.

**Probe 100 iterations first** (~15 min), and the bar is sharper than
before: with a PD action space **iteration 0 should already show long
episodes**. B7's iteration 0 was 85 steps at 100 Hz — i.e. 0.85 s. If B8's
first iterations are not dramatically better in *seconds*, the action-space
change did not take, and nothing else matters until it does. Also check the
usual: witness margin ≥ 100×, `episode_steps` not pinned at envs × unroll,
reward and loss finite, and that the machine is not simply standing rigidly
(mean `com_z` near the new `Z0`).

## 5. Scoring, and what B9 is

`steps.py` at 12 seeds across all 24 checkpoints, **on the conjunction**:
stepped >10 mm AND survived 300/300. Then the controlled comparison that
B7 taught us to run — `stand8` and the best B8 checkpoint **on one task**,
using today's header-channel fix in `compare.play` and `steps.trace`.
**B6's 6/12 is the number to beat.** Install on the conjunction, never on
reward, and install the script with the policy.

If B8 clears it, **B9 is the curriculum**: add `--init-from <policy>` to
`cadex_train.py` — a contained change, and the single highest-value trainer
feature for this project — and walk the band 0.8 → 1.2 → 1.8 → 2.5 N across
four short warm-started runs. That is the schedule ADR-100 called
unavailable, unavailable only because the flag does not exist.

## What would make me stop and rethink

- **Soles not flat at z ≈ 0 after the crouch** — the sign convention in the
  crouch chain is wrong; fix before anything else, everything downstream is
  measuring a different robot.
- **Holding torque at the crouch is a large fraction of 86 N·mm** — θ is too
  deep; come down toward 10°.
- **Probe iteration 0 is not dramatically longer than B7's** — the action
  space did not change in the way intended. Check the exported MJCF actually
  carries `<position>` actuators with the expected gainprm/biasprm before
  looking anywhere else.
- **The machine stands rigidly and never steps** — with positive kernels the
  risk inverts: `upright`/`height` may now be *so* rewarding that a
  motionless stance is a local optimum. The lever is raising `capture`
  relative to them, not adding a cost.
