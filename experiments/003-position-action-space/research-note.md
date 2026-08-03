# Why standing is going badly, and what the literature actually does

Working note for `~/cdx-mjc`. **Lives here, not in the cadex repo**
(`docs/MUJOCO.md` §7). Written 2026-08-03 after B7 came back a measured
regression, in answer to a fair question: *this is the most-studied problem
in legged robot learning — why are we bad at it?*

The short answer is that we have been solving a harder problem than the one
the literature solves, in four specific ways, and three of the four are
**pure script changes** that need no engine or trainer work at all.

This note is evidence, not opinion. Everything asserted about our own setup
below was read out of the code today and the line is cited. Everything
asserted about the literature has a link.

---

## 1. What we have been doing, and where it came from

The genealogy matters, because none of the choices below were mistakes when
they were made — they were made for reasons that were good at the time and
then never revisited.

| choice | where it came from | still justified? |
|---|---|---|
| **raw torque actuators** | M4 built `motor` actuators first because a servo needs a gain nobody had measured (`cadex_assembly_api.py`: *"there is no defensible default"*). The biped inherited it. | **No.** See §3.1. |
| **reward = `alive` +1 minus a pile of costs** | M9b/ADR-087, priced arithmetically against a stumble | **No.** See §3.2. |
| **100 Hz control** | M9: "100 Hz divides the 0.002 s solver step exactly — five steps to an action, with nothing rounded" | Defensible but costly. See §3.3. |
| **no curriculum** | ADR-100: the trainer has no scheduler, so "curriculum inside the distribution" was the only option available | True, and it is a gap worth closing. See §3.4. |

Nine runs — m9a/b/c, a1c, B1–B7 — have moved the *disturbance* five times
and the *reward* three times, and never once moved the action space, the
control rate, or the sign convention of the reward. Those are the four
things the literature is most consistent about.

---

## 2. What the literature actually does

Sources, all checked today:

- **Rudin, Hoeller, Reist, Hutter, *Learning to Walk in Minutes Using
  Massively Parallel Deep RL*** (CoRL 2022) — the paper `legged_gym` comes
  from. https://arxiv.org/abs/2109.11978
- **`leggedrobotics/legged_gym`** — the reference implementation.
  https://github.com/leggedrobotics/legged_gym
- **Booster Gym: *An End-to-End RL Framework for Humanoid Robot
  Locomotion*** (2025) — a current, complete, humanoid-specific recipe.
  https://arxiv.org/html/2506.15132v1
- **Reactive Stepping for Humanoid Robots using RL: Application to Standing
  Push Recovery on the Exoskeleton Atalante** —
  literally our task. https://arxiv.org/pdf/2203.01148
- **DecAP: Decaying Action Priors for ... Torque-Based Legged Locomotion**
  — exists *because* torque-based policies are hard to train, and its whole
  contribution is a scaffold to make them trainable.
  https://arxiv.org/html/2310.05714

The consistent recipe:

1. **The policy outputs joint position offsets from a nominal pose**, not
   torques. Booster Gym: `q_des = q_0 + a_t`, with a PD controller turning
   that into torque at a higher rate than the policy runs, "improving
   stability of the policy on the real robot compared to making the policy
   directly output torque."
2. **The per-step reward is clipped at zero.** Booster Gym: *"The total
   reward for each frame is clipped to zero to avoid incentivizing early
   termination with negative rewards."* `legged_gym` ships the same thing as
   the `only_positive_rewards` flag (`torch.clip(self.rew_buf[:], min=0.)`).
3. **The policy runs at ~50 Hz**, with the PD loop underneath it faster.
4. **Shaping rewards are positive bounded kernels** — `exp(-e²/σ)` — and the
   penalties are small regularisers on top (torque ~2e-4, action rate ~1.0,
   joint limits ~1.0).

---

## 3. The four gaps, ranked, with our own code cited

### 3.1 The action space — and the fact that our zero action falls over

**This is the big one.**

Our ten actuators are all `kind="motor"` (`mg-legs.cadex/script.py:1138`):
the network's output *is* joint torque in N·mm. And the trainer maps a
network output of 0 to the **midpoint** of the action range
(`cadex_train.py:449-450`: `output_bias = (high + low) / 2`), which for a
±86 N·mm motor is **zero torque**.

`feasibility.py` check 5 measures what zero torque does to this machine:

> from the reset pose, zero torque for 3 s → **falls at 0.976 s**

So an untrained policy is a machine that falls over, and the *first* thing
PPO must discover — before anything about balance — is the gravity
compensation torque for ten joints simultaneously, continuously, forever.
Nothing in the reward tells it what those torques are.

In the literature the same untrained policy **stands**, because zero action
means "hold the nominal pose" and a PD servo holds it. The action space
carries the statics; the policy only learns the *deviation*. The reason is
stated plainly in the survey work: the nominal-offset parameterisation means
"the zero action naturally corresponds to the robot standing at its
nominal/default pose", which is why it is a good initial policy.

That is not a small efficiency difference. It is the difference between
searching for a controller and *adjusting* one.

**And our engine already supports it.** `CadexDynamics.py:3285` handles
`kind == "position"` with `set_to_position(kp, kv=kd)` — a real PD loop in
the compiled model, with a phase-0-verified gain/bias check at line 4642.
`feasibility.py` check 6 already proves a PD holds this machine (kp 1.0,
kd 0.02, **3.3 N·mm** of peak effort against 86 available).

**The catch, and it is the one design problem worth thinking about.** A
position actuator's action range is *its joint's own limits*
(`CadexDynamics.py:_ACTION_SOURCES`, `("position","angular") → "angle_limits_degrees"`),
and zero action is the midpoint of that range. Our limits are asymmetric
about the straight-legged standing pose:

| joint | limits | midpoint = zero action |
|---|---|---|
| hip_roll | [−30, 45] | 7.5° |
| hip_pitch | [−60, 90] | 15° |
| knee | [−5, 130] | **62.5°** |
| ankle | [−40, 40] | 0° |
| ankle_roll | [−20, 20] | 0° |

Zero action would command a 62.5° knee bend. So switching actuator kind
alone is not enough: **the nominal pose and the joint limits have to be made
symmetric about each other.**

The literature's answer is the one to copy: humanoids in this field do not
stand straight-legged. Their nominal pose is a **slight crouch** — knees
bent — which is also what gives the knee and ankle authority to absorb.
A kinematically consistent crouch is hip θ, knee 2θ, ankle θ.

This is a genuine change to the drawing, so `Z0`, `X0`, `Y0`, the foot
centroid and `ω₀` all have to be re-measured (hazard 9). `measure.py`
exists for exactly that.

### 3.2 The reward sign convention — the failure we already measured twice

Our reward is `alive` +1.0 minus eleven-to-thirteen costs, and the trainer
sums them raw: there is no clip, no floor, and no `only_positive_rewards`
equivalent anywhere in `cadex_train.py`.

We then measured, with `reward_decompose.py`, exactly the failure that flag
exists to prevent. On the states B6's own policy visits:

| objective | per-step reward |
|---|---|
| B6 | **+0.0103** |
| B7 as first written | **−0.2060** |

A negative per-step reward with a termination available means **ending the
episode beats continuing it**, so the optimal policy is to fall over
immediately — the "suicide policy". B7's first probe found it in 150
iterations: mean episode length fell monotonically 23.5 → 8.0 steps while
critic loss converged to 2.46. It did not fail to learn. It learned what it
was asked.

And note what the B6 column says: **B6 succeeded on one per cent of
headroom.** Every run of this project has been balanced on a knife edge
nobody had measured, which is why adding three terms tipped it over.

The literature's fix is structural rather than careful: make the sum
non-negative *by construction*. Positive bounded kernels — `w · exp(−e²/σ²)`
— are in [0, w], so the total is always ≥ 0, termination is always bad, and
no amount of adding terms can ever recreate the suicide mode.

**Our expression vocabulary already has `exp`** (`docs/XSCRIPT.md`: *"may
call abs, asin/arcsin, arctan, cos, sin, exp, sqrt and tanh"*), so this is
a script change and needs no trainer work.

### 3.3 Control rate

100 Hz was chosen so the control interval divides the solver step exactly.
That is a real virtue and it is not free: at 100 Hz a 2 s recovery is **200
control steps** of credit assignment, where the field's standard 50 Hz makes
it 100. The GAE credit chain `1/(1−γλ)` and the discount horizon are both
counted in *steps*, so every temporal-credit problem this project has had —
and ADR-112 is entirely about one — is twice as hard as it needs to be.

50 Hz still divides 0.002 s exactly (ten solver steps to an action), so
nothing about the "nothing rounded" argument is lost.

### 3.4 No curriculum, and no way to build one

ADR-100 is right that the trainer has no scheduler, and B7 established the
consequence quantitatively: a band wide enough to demand stepping
(0.4–2.5 N) left **~4% of episodes winnable**, and the unwinnable 96% were
not neutral — they trained giving up.

Two ways to get a curriculum without a scheduler:

- **Warm starting.** Train at an easy band, then continue *from those
  weights* at a harder one. `cadex_train.py` has **no** resume or
  init-from-policy option (checked: nothing but `--initial-std`). Adding one
  is a contained change and it is the single highest-value trainer feature
  for this project.
- **Reward-driven implicit curriculum**, which is what positive kernels give
  for free: a machine that survives 0.4 N scores well immediately and the
  gradient toward harder pushes is smooth rather than a cliff.

---

## 4. What we are *not* doing wrong

Worth stating, so effort does not go here:

- **Observation normalisation is present and correct** — running mean and
  variance, `cadex_train.py:773, 791`.
- **Truncation vs. termination bootstrapping is correct**, which is a thing
  most hand-rolled PPO gets wrong: *"GAE, with a timeout bootstrapped and a
  failure cut"* (`cadex_train.py:901`). `terminal` cuts the bootstrap,
  `done` cuts the carry. This is right.
- **Sample budget is not obviously short.** 2048 envs × 40 unroll × 2400
  iterations ≈ **197 M** environment steps, which is in the normal range for
  this class of task.
- **The mechanism is sound.** `feasibility.py` passes every gate: it can be
  held up, it falls when it is not, a PD stands it on 3.3 N·mm of 86, and a
  step reaches the worst declared shove with 1.34–1.76× of margin.
- **The engine's dynamics are verified** — MJX and MuJoCo agree to float32
  on every observation kind, box-on-box contact drift is measured and
  bounded, and ADR-116's new channel was checked in both simulators before
  it was used.

The physics and the plumbing are not the problem. The **problem
specification** is.

---

## 5. The plan: B8

One run, four changes, all of them script-side. Ordered so that each is
independently attributable if it goes wrong.

### B8a — the rebuild (no training)

1. **Nominal crouched stance.** Hip θ, knee 2θ, ankle θ, with θ chosen so
   the crouch is shallow — 15–20° at the hip. Re-measure `Z0`, `X0`, `Y0`,
   the foot centroid and `ω₀` with `measure.py`. Re-run `feasibility.py`:
   gravity compensation and the PD hold both have to pass at the new pose.
2. **Joint limits symmetric about that pose**, so the trainer's zero action
   *is* the nominal stance. Knee becomes roughly [−5, 75] rather than
   [−5, 130] — giving up hyperflexion the machine has never used.
3. **Position actuators.** `kind="position"`, `control_deg="0"`,
   `stiffness_nmm_per_deg` and `damping_nmms_per_deg` from `feasibility.py`
   check 6's measured PD (kp 1.0 N·m/rad ≈ 17.5 N·mm/deg, kd 0.02 N·m·s/rad
   ≈ 0.35 N·mm·s/deg), torque limit unchanged at 86 N·mm.
4. **All-positive reward.** Every shaping term becomes `w · exp(−(e/σ)²)`,
   bounded in [0, w]:

   | term | weight | error |
   |---|---|---|
   | `upright` | 1.0 | `pel_qx² + pel_qy²`, σ 0.15 |
   | `over_feet` | 1.0 | foot-referenced CoM offset, σ 40 mm |
   | `capture` | 1.5 | ξ, σ 50 mm |
   | `height` | 0.5 | `com_z − Z0`, σ 30 mm |
   | `arrest` | 0.3 | true CoM speed, σ 300 mm/s |
   | `swirl` | 0.3 | \|cam_xy\|, σ from `swirl_scale.py` |
   | `posture` | 0.3 | joint deviation from nominal, σ 20° |
   | `effort` | 0.2 | actuator force, σ from measurement |

   Total 5.1 standing, 0 when everything is wrong. **No term can ever make
   a step negative**, which is the specific defect that killed B7:
   `arrest` and `swirl` as *costs* taxed the swing phase of the very
   recovery they were added to buy. As positive kernels they simply stop
   paying during the swing and resume when it lands.

   Verify with `reward_decompose.py` before dispatch, on `stand8`'s states.

5. **50 Hz control**, episode 6 s = 300 steps. `--discount 0.99` (2.0 s),
   `--gae-lambda 0.95` — B6's *horizon in seconds*, at half the step count.
6. **Band 0.3–0.8 N**, B6's exact band. Not a retreat: with three
   simultaneous structural changes, the task has to be the one we have a
   6/12 baseline on, or the run answers nothing.

### B8b — the run

2,400 iterations (~3 h at 4.36 s/iter, likely faster at 50 Hz since an
episode is half the steps). Probe 150 first; the bar is episode length
climbing rather than collapsing, and with a PD action space the *first*
iteration should already show long episodes — if iteration 0 is not
dramatically better than B7's 85 steps, the action-space change did not take
and everything else should stop until it does.

**Score against `stand8` on one task**, using the `steps.py` /
`compare.py` header-channel fix made today, on the conjunction: stepped
>10 mm **and** survived. B6's 6/12 is the number to beat.

### If B8 works, B9 is the curriculum

Add `--init-from <policy.cxpolicy>` to `cadex_train.py` and walk the band up
0.8 → 1.2 → 1.8 → 2.5 N across four short runs. That is the shape ADR-100
said was unavailable, and it is unavailable only because the flag does not
exist yet.

---

## 6. Open questions worth an answer before B8a

1. **How shallow can the crouch be?** Deep enough that the limits centre
   sensibly, shallow enough that `height` does not fight it and the servos
   are not holding a squat all day. `feasibility.py` check 2 gives the
   holding torque at any candidate pose in seconds — sweep it.
2. **What PD gains?** Check 6 sweeps six pairs already; kp 1.0 / kd 0.02 is
   the measured winner but it was measured at the *straight-legged* pose and
   should be re-swept at the crouch.
3. **Does the observation set still make sense?** With a PD action space the
   policy no longer needs `*_tau` to infer what it just did — ten channels
   might be better spent on something else, or simply freed.
