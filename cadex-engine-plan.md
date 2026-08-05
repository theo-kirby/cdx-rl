# What cdx-rl needs from the Cadex engine

**Drafted 2026-08-04 for hand-off. Rewritten 2026-08-05 as PR specs, because
cdx-rl now submits the PRs itself.**

Work happens in the PR clone at `/home/theo/cadex-prs`, against
`origin/main` — which is the right base: `standing-policy` is fully merged
(`git rev-list --count origin/main..origin/standing-policy` is **0**). Never
push to `theo-kirby/cadex`; never touch the operator's tree at
`/home/theo/cadex`.

`cadex-wishlist.md` is the full list — now fifteen entries — with what each
gap cost. This document is the subset that was **costing research time**,
scoped so the work can be started without re-deriving the problem.

**Order matters.** §2 (the command range) goes first: it is smaller, touches
no trainer, and is independently useful. §1 (`--init-from`) changes
`cadex_train.py`'s sha256 and so invalidates every downstream trainer pin
including `remote_train.sh`'s own check — acceptable now, but not something to
have in flight while another PR is being reviewed against a moving trunk.

**Item 3 is resolved and kept for the record**, at the bottom. It asked for a
pin that could build the mechanism it trains; getting current answered it, and
no PR was needed.

Nothing here is a request to change training's *place*. ADR-084 stands: the
engine ships without a line of JAX and there is no train verb. §1 touches
`training/cadex_train.py`, which is a different thing from putting a GPU
dependency in the CLI.

**Line numbers below are against `b169a092`** and were re-derived there on
2026-08-05 — the numbers in the original draft were taken against `06d1374b`
and are all shifted by the 15-commit engine delta. The *trainer* line numbers
did not move, because `training/` is byte-identical across that range.

---

## 1. `--init-from`: warm-start training from an existing policy

**Wishlist #11. Costs ~5 GPU-hours every time a run needs to go further.**

`cadex_train.py` has no resume and no initialise-from-policy option, so every
run starts from a fresh network.

### What it cost this week, measured

Experiment 003's three seeds all peaked at iteration **1700–1750** of 1800 —
the last two checkpoints written — and the pooled paired test put
late-beats-early at **8 discordant to 1, p = 0.0391**. The runs stop
mid-climb, and the obvious next question is where improvement actually stops.

Answering it without a warm start means re-running from zero at a greater
length: **6.9 h to reach 700 new iterations**, with 5 h spent recomputing a
curve already on disk. That arm was dispatched on 2026-08-04 and **cancelled
30 minutes in** once the accounting was written down. With `--init-from`, the
same question is a **1.9 h continuation** from `stand10.001700`.

It also blocks experiment **B9** outright — a warm-start curriculum walking
the disturbance band 0.8 → 1.2 → 1.8 → 2.5 N across four short runs, each
initialised from the last. ADR-100 concluded a curriculum could not be
scheduled; it was right about the capability, and the capability is one flag.

### Wanted

```
--init-from <policy.cxpolicy>
```

Load the weights out of the container into the initial network; leave the
optimiser state **fresh**. Fresh optimiser state is the conservative choice
and probably the right one — each leg of a curriculum is a different task, so
a carried-over Adam moment describes a gradient landscape that no longer
exists.

Two things that make this smaller than it looks:

* The container already round-trips. `policy_forward` reads header and weights
  out of a `.cxpolicy` today; the loader exists.
* The engine already refuses a shape mismatch by name
  (`policy_observation_mismatch`), so `--init-from` against an incompatible
  bundle has a correct failure mode without new work.

### What landing it costs downstream

This section used to read *"why we can't do it here"* — `cadex_train.py` was
inside a read-only tree, and patching a copy would have meant a different
update rule with every comparison to 001, 002 and 003 needing to be
re-argued. The first half is obsolete. **The second half is not, and it is
the thing to get right in the PR body.**

Landing `--init-from` moves `cadex_train.py`'s sha256. That invalidates every
recorded trainer pin, and `remote_train.sh:199`'s own check along with them.
Under the new policy that is acceptable rather than disqualifying, but it is
not free:

* **State it in the PR.** The digest is part of the interface for anyone
  comparing runs across it.
* **Pay for a bridge run.** One seed of an existing arm, retrained under the
  new trainer and scored against where the old one landed on the same metric
  and the same evaluation seeds. `method.md` §8b has the protocol and the
  warning that it reports on *shape*, never on a value — this card gives 0 of
  1500 iterations bitwise identical at the same seed and the same digest.
* **The unused flag must be a no-op.**
  `test_a_second_run_at_the_same_seed_writes_the_same_policy` (`:446`) is the
  regression that proves it; if it stays green, the digest moved but the
  algorithm did not, and the bridge run should confirm that cheaply.

---

## 2. A position actuator's command range, separate from its joint limits

**Wishlist #12. Currently forces an experiment to be un-reproducible.**

`CadexDynamics._ACTION_SOURCES` maps `("position", "angular")` to
`angle_limits_degrees`, so a position servo's action range **is** the joint's
declared physical range. The rationale in `_action_bound` — *"a setpoint
outside them is a command the joint cannot obey"* — proves only that the
command range must not **exceed** the joint range. It does not follow that the
two must be **equal**.

### What it cost, measured

Experiment 003's policies brace by commanding position errors far past the
servo's saturation point. The servo saturates at **16.4° of error**
(86 N·mm ÷ 5.236 N·mm/deg); the trained policies command up to **44°** on a
±45° joint. With nothing pushing at all, the worst motor sits at **73–91 % of
its rating** and is above 90 % of it for **49–77 % of frames**, replicated 3
of 3 seeds. That is a policy family describing a machine nobody can build.

The natural experiment — cap what the policy may *ask for*, leave the
machine's range of motion alone — **cannot be expressed**. Capping
`angle_limits_degrees` in the script narrows the joint too (the exported
MJCF's joint ranges are the same ten numbers as the action table's), which
removes reachable configurations for reasons unrelated to torque and
confounds the result.

So experiment 004 edits a **derived bundle** instead
(`make_clamp_bundle.py`: cap the action table, copy the MJCF unchanged — it
works because `ctrllimited="false"` on every actuator means nothing downstream
re-clamps). It runs, but the result is a claim about committed bytes rather
than something re-derivable from source, which is the one property this
project most wants to keep.

### Wanted

```python
assembly.actuator(..., command_limits_degrees=[-15, 15])
```

Optional; defaults to the joint's range so nothing changes for anyone who does
not ask; **refused if it exceeds** the joint's limits, which preserves the
original rationale exactly. It needs to reach the exported action table and
should leave the MJCF joint range alone. Linear twin: `command_limits_mm`.

#### The path a value takes

Located against **`b169a092`** on 2026-08-05. Every row was opened and read.

| step | where |
|---|---|
| kwarg on the surface | `cadex_assembly_api.py:1838` — `Assembly.actuator`, already keyword-only |
| validation | same file; `_unit_pair:543` is the deg/mm pair helper, `_limits:397` the two-endpoint precedent |
| into `properties` | `cadex_assembly_api.py`, `actuator`'s properties dict — **a kwarg missing here never reaches the worker**, and it fails silently |
| worker passthrough | `cadex_assembly_worker.py`, the `actuators` maker (`:1934-1935`) |
| engine record | `CadexDynamics.actuator_records:2737` → its `declared` sub-dict |
| **the actual narrowing** | `CadexDynamics._action_bound:5497`, the `else` (position) branch — intersect with `joint_limits["declared"]` at **`:5596-5597`**, immediately before `low, high = declared_pair[0], declared_pair[1]` |
| into the task JSON | `CadexDynamics.task_records:6305` |

#### Semantics to decide and state in the PR body

`experiments/004-ceiling-and-clamp/make_clamp_bundle.py` takes `abs()` of both
bounds and writes a **symmetric** ±cap. **The PR should clamp each bound
independently** — that is the more general behaviour and the one an asymmetric
joint needs. Say so explicitly: it is a no-op for `stand-b8`, whose joint
ranges are already symmetric, so the two agree on the only bundle anyone can
currently diff.

#### Note for the PR body: policies carry the narrowed numbers

`_POLICY_ACTION_FIELDS` (`CadexDynamics.py:7732`) makes `verify_policy`
(`:7752`) compare `low`/`high` exactly, so a policy trained under a narrowed
range records the narrowed bounds and **will not verify against an unclamped
bundle**. That is correct — it is a different action space — but it is
surprising, and it is better said out loud than discovered.

#### Tests

* `test_dynamics_task_model.py` —
  `test_a_position_action_range_is_the_joints_own_declared_limits` asserts
  `action["source"] == "angle_limits_degrees"`; it needs a sibling for the
  narrowed case.
* `test_dynamics_actuators_api.py` — the refusals: exceeding the joint's
  limits, one endpoint only, wrong unit for the motion type.
* **Prose in the same commit**, which is the Cadex convention:
  `CadexScriptedRuntime.py` and `docs/XSCRIPT.md`.

#### Verification

`pixi run test-engine` in the PR clone — **1105 tests, headless, no build
needed** (`conftest.py` stubs FreeCAD), so it runs without the ~5-minute
`build-engine`. Then regenerate `tasks/stand-b8-clamp25` from `script.py` and
diff its action table against `make_clamp_bundle.py`'s output. **A
byte-identical action table retires the workaround.**

### Why it matters beyond one experiment

A software command limit narrower than the mechanical range is ordinary
practice on real machines. It is also one of the very few levers that acts on
this failure **without going through the reward** — and a reward term is
already ruled out (hazard 16). Most importantly it separates two findings that
call for completely different next steps: *the policy chooses to saturate* and
*the dynamics force it to*.

---

## 3. ~~A pin that can build the mechanism it trains~~ — RESOLVED 2026-08-05

**Wishlist #13. No PR was needed: the kinds were already upstream and this
clone could not see them.** Kept below as written, because the reasoning about
what an engine bump costs is still the reasoning, and because the way this was
misdiagnosed is worth remembering.

`centroidal_angular_momentum` / `mjSENS_SUBTREEANGMOM` landed in **`593f64e6`
(ADR-116, 2026-08-03)**, `centre_of_mass_velocity` before it, and both are in
`origin/main`. `git log --all -S"centroidal_angular_momentum"` came back empty
against the old checkout not because nobody had written it but because
`06d1374b` was the newest object that clone held. **"Searched and not found"
was a claim about one clone.**

Getting current cost a clone, a `pixi run build-engine` (~5 min), and two
environment variables. The bump was safe on exactly the grounds this section
argued it would be: `training/` is byte-identical across all 15 commits, so
the engine moved and the trainer did not.

What it did *not* fix, and what became wishlist **#14** and **#15**: the
worker's default address-space cap refuses the build anyway (500× slowdown,
`SIGXCPU`), and the rebuilt bundle's digest differs from the laptop's by one
floating-point ULP, which makes `assembly.policy` refuse a valid policy. Items
2 and 3 of "wanted" below — a build-time capability check, and `cadexd`
reporting its observation vocabulary — are **still open and still worth PRs**.

---

*Original text follows.*

**Wishlist #13. Makes the mechanism editable only on a machine with no GPU.**

`mechanisms/mg-legs/script.py` declares two observation kinds:

```python
assembly.observation(pelvis_c, "centre_of_mass_velocity",     name="cv"),
assembly.observation(pelvis_c, "centroidal_angular_momentum", name="cam"),
```

Neither string appears anywhere in the pinned checkout. `06d1374b`'s
observation table stops at the eighth kind:

```
actuator_force, armature, centre_of_mass, component_angular_velocity,
component_linear_velocity, component_orientation, component_position,
damping, friction_loss, mass, position, velocity
```

`sb1x` (`06d1374b`) refuses `centre_of_mass_velocity`. `sb9x` (`ae8da6a6`)
got past that and refused `centroidal_angular_momentum`. Only the laptop
(`560935bd`) builds it — and the laptop has no GPU and no training venv.

The practical effect: recovering `script.py` was supposed to make the
mechanism *changeable rather than a dead end*, and it is — on the one machine
that cannot train. Every mechanism edit has to be authored on the laptop,
exported, and carried to `sb1x` as bytes, which is the situation the recovery
was meant to end.

### This is now the binding constraint on the research, not the GPU

Filed 2026-08-04, after experiment 004. This item was previously "makes the
mechanism awkward to edit". It is now **what the next experiment is waiting
on**, and that is a different priority.

004 established that the bracing is a policy choice: capping the command range
cuts resting duty above 90 % of rating from 51.8 % to 13.5 % at ±25°, for a
stepping cost the paired test cannot distinguish from zero. The obvious
counterfactual — *stop restricting the policy and size the motor for the
~230 N·mm it wants* (44° of commanded error × 5.236 N·mm/deg) — **is not
available, and `script.py` §1232–1276 already says why**:

* The centre of pressure cannot leave the sole, which reaches 45.5 mm ahead of
  the ankle, so past **2.581 N × 45.5 mm = 117 N·mm the foot rolls instead of
  pushing** (ADR-082).
* 86 N·mm was chosen partly *for this*. At MG90S stall (216 N·mm) a single
  ankle out-torques the footprint by **1.8×**, which means the machine could
  **tip itself** by over-torquing one ankle. At 86, one ankle sits below 117
  and cannot roll the foot.
* 86 is ~40 % of stall — an engineering judgment, stated as one, not a
  datasheet number (ADR-086).

So the torque budget is **not raisable without a bigger foot**, and the foot
is a `script.py` edit. **The next mechanism experiment is foot geometry, and
this item is what blocks it.** Every other lever this project has — reward
terms, disturbance schedule, action space, command range — has now been moved,
and the remaining one is on the far side of a build that only the laptop can
do.

Note that item 2 (the command range) and this item compound rather than stack:
even a clamp expressible in the script could not be built into a bundle here.
Together they are why `concept.md`'s success criterion 5 — *"a result that
came out of here can be built"* — is recorded as **currently failing**.

### Wanted, in preference order

1. **The earliest revision that has both kinds**, stated, so cdx-rl can move
   its engine pin deliberately instead of discovering the gap per-box. The
   ten commits between `06d1374b` and `ae8da6a6` were **all** engine-side with
   `training/` byte-identical — the trainer sha256 is unchanged across them —
   so an engine-only bump is available without touching comparability, if
   somebody states where that boundary holds.
2. **A build-time capability error that names the gap**:
   `assembly.observation` with an unknown kind should report the kind, the
   revision, and the supported set. Today it surfaces partway through a long
   script and reads like a modelling error.
3. **`cadexd` should report its observation vocabulary** so a driver can
   refuse before spending a build.

---

## Also open, but not currently blocking

Filed with reproductions in `cadex-wishlist.md`; none of these is costing GPU
time today.

| # | | Cost |
|---|---|---|
| 4 | Re-stage the engine payload, or make a stale one fail loudly | The stale payload at `build/engine/…` predates the whole MuJoCo surface and fails as *"assembly.mjcf is not defined"* — an hour, once, for anyone who exports `CADEX_ENGINE_ROOT` |
| 1 | An artifact-path resolver on the CLI surface | Every driver reimplements it |
| 6 | `progress.json` should carry the run's identity and the series | Forces a sidecar |
| 7 | Print the termination mix by default | ADR-106: collected since M9, never printed, cost three runs |
| 8 | `put_asset` moves the revision, and the next `write_script` is refused | Ordering trap |
| 9 | `policy_forward` takes the header and weights, not the container | Awkward call site |
