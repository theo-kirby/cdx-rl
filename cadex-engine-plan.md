# What cdx-rl needs from the Cadex engine

**Drafted 2026-08-04, for hand-off to the Cadex development repository.**

`cadex-wishlist.md` is the full list — thirteen entries, written as they were
hit. This document is the subset that is **currently costing research time**,
ordered by what it costs, with the diff sketched so the work can be scoped
without re-deriving the problem.

Nothing here is a request to change training. ADR-084 stands: the engine ships
without a line of JAX and there is no train verb. Items 1 and 2 touch
`training/cadex_train.py` and the mechanism vocabulary respectively; item 3 is
a version-pin question, not a feature.

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

### Why we can't do it here

`cadex_train.py` is inside the read-only tree. Copying it out to patch it
would break the one property that makes runs comparable: `tools/train.py` pins
the trainer by sha256 (`--require-trainer`) precisely because ADR-104
established that the same seed and the same hyperparameters mean nothing if
the update rule differs. A patched copy is a different update rule, and every
comparison to 001, 002 and 003 would have to be re-argued.

**Please treat the sha256 as part of the interface.** If `--init-from` lands,
we will re-pin deliberately and re-baseline; what we cannot absorb is the
digest moving for an unrelated reason.

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
should leave the MJCF joint range alone.

### Why it matters beyond one experiment

A software command limit narrower than the mechanical range is ordinary
practice on real machines. It is also one of the very few levers that acts on
this failure **without going through the reward** — and a reward term is
already ruled out (hazard 16). Most importantly it separates two findings that
call for completely different next steps: *the policy chooses to saturate* and
*the dynamics force it to*.

---

## 3. A pin that can build the mechanism it trains

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
