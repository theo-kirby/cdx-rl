# cadex-wishlist.md — things cdx-rl wants from Cadex

**Captured, not acted on.** `/home/theo/cadex` is read-only from this
repository: no commits, no file edits, no branch changes, no builds. When
cdx-rl finds it wants something from Cadex, it is written down here and acted
on somewhere else, by somebody with that repository open.

Each entry says what we hit, why the workaround is unsatisfying, and — where
it matters — why the current behaviour might be *right* and this wish wrong.
Cadex has 106 ADRs behind its current shape; several things that look like
gaps are decisions.

Status is one of: **open**, **worked around**, **withdrawn**, **filed**.

---

## 1. An artifact-path resolver on the CLI surface

**Status: worked around** (`tools/cadexd_client.py::accepted_artifacts`).

`cadex --json` names every output and its `artifact_kind` and does not say
where any file is. `cadex export` writes STEP/STL/BREP, which are BREP-domain
formats — so for `assembly_mjcf_xml` and `assembly_training_task_json`, the
two artifacts a training pipeline exists to consume, **the CLI has no path at
all.**

The information is right there: `script.json` carries
`accepted_attempt.staging`, and `result.json` in that directory lists every
output with an `artifact_path`. `inspect scope="output"` already reads
exactly that report — it just returns names and kinds rather than paths.

**Wanted:** either `inspect scope="output"` including the resolved absolute
`artifact_path` per output, or a `cadex artifacts --json` that prints the
accepted revision's outputs with their paths. Roughly twenty lines against
`CadexPinResolution.accepted_attempt_dir`, which is already public for this
reason (ADR-043).

**Why it matters more than it looks:** every consumer that wants an MJCF has
to re-derive this, and the derivation has a trap in it (see #2). Three
consumers means three chances to get it wrong differently.

## 2. Say out loud that the artifact directory is not named by `accepted_revision`

**Status: open.**

The staging directory is named with a *pre-run* revision computed over the
stored spec cache (`CadexScriptedRuntime`, ~line 1110); `validate_project_result`
then recomputes the revision from the specs the worker actually collected and
records **that** as `accepted_revision`. The two disagree for any script that
declares a parameter.

Measured on sb1x with a two-parameter box script: accepted revision
`104826f0…`, artifacts under `script_artifacts/93a118d4…/`.

The code comment at the construction site is accurate. Nothing at the
*consumer* end says it, and the obvious wrong implementation —
`root/"script_artifacts"/state["accepted_revision"]` — looks right, passes on
a script with no parameters, and fails on every real one.

**Wanted:** one sentence in `INTEGRATION.md`'s store description, or in
`CadexScriptStore`'s class docstring, saying the directory name is not the
accepted revision and `accepted_attempt.staging` is the only correct route.
Solving #1 would make this moot for most callers.

## 3. Trainer-side early stopping

**Status: open. cdx-rl is building the workaround (a supervisor).**

`cadex_train.py` has `--iterations` and nothing else. There is no
`--early-stop`, no patience, no `--max-iterations-without-improvement`. It
computes and writes `best_iteration` and `best_reward_per_step` into
`progress.json` on **every iteration** — and then keeps going regardless.

The bill, from `/home/theo/cadex-jobs/stand-task-20260802-200109`:

| | |
|---|---|
| best iteration | 598 (reward/step 0.337) |
| final iteration | 2499 (reward/step 0.146) |
| wall time | 14 050 s ≈ 3.9 GPU-hours |

**About 76 % of that run was spent regressing**, and the file said so the
whole time. `MUJOCO.md` §7 step 8 already names the problem — *"the reward
peaked at iteration 1200 of 2000 the last time and nobody could see it"* —
and answers it with `watch`, i.e. with a human.

**Wanted:** `--patience N` (stop after N iterations with no improvement in
`best_reward_per_step`), and an exit that writes a normal terminal
`progress.json` with `state: "done"` and a reason field, so a supervisor can
tell "stopped early" from "crashed".

**The honest counter-argument, and why the wish is narrower than it looks:**
ADR-099 established that the trainer's reward and measured survival can be
*anti-correlated* — in the M9 run, survival was 12/12 where the trainer
reported its worst numbers and 0/12 exactly where it reported its best. So
stopping at the reward peak is a **compute** decision, not a **selection**
decision, and a `--patience` flag must not be mistaken for one. It should
stop the burn; it must not be allowed to choose the checkpoint. Selection is
by measured survival, which is `compare`'s job and belongs outside the
trainer.

Until then, cdx-rl polls `progress.json` and sends `SIGTERM`. That works, and
it is strictly worse than the trainer knowing: a supervisor cannot flush a
final checkpoint, and it races the checkpoint writer.

## 4. Re-stage the engine payload, or make a stale one fail loudly

**Status: open. Blocking-ish; worked around by driving the checkout.**

`build/engine/cadex-engine-0.0.0-linux-x64` on sb1x was assembled 2026-07-31
and predates the entire MuJoCo surface. Its `Mod/cadex` has no
`CadexDynamics.py`, and its assembly domain exports stop at `exploded_view` —
no `dynamics`, `mjcf`, `task`, `policy`, `rollout`, `body`, `collision`,
`joint_dynamics`, `actuator`, `observation`, `reward`, `termination`,
`randomise`, `reset_variation` or `disturbance`.

A client pointed at it does not get a clear error. It gets *"assembly.mjcf is
not defined"* from inside a script, which reads like a modelling bug and
sends you to the wrong file. Worse, `CADEX_ENGINE_ROOT` exported in a shell
silently downgrades the CLI too — `./cadex` with it set drives the stale
payload; with it unset it falls back to the development tree and works.

cdx-rl works around this by driving the checkout directly
(`CADEX_ENGINE_DEV_TREE`) and by asserting the surface at startup
(`CadexdClient.require_dynamics()`).

**Wanted, in rough order of preference:**

1. Re-stage the payload on this box. One command in the Cadex repo; not ours
   to run.
2. `cadex-engine.json` carrying an **API surface version or a domain
   fingerprint**, so a client can refuse a payload that cannot do what it
   needs before spawning it. The manifest already checks the *protocol*
   version for exactly this reason (ADR-020); the domain surface is the other
   half of the same contract, and it is the half that moves.
3. Failing both, a line in `CLI.md` saying that a payload older than the
   feature you want will fail as a script error.

## 5. A `run` op, or documented cadexd reuse across projects

**Status: open, low priority.**

One `cadexd` per open project, spawned per client. Startup on a warm page
cache is fast (0.04 s to the ready banner, measured), so this is not a
performance complaint. It is a *sweep* complaint: a parameter sweep across
twenty variants is twenty processes, and `open_project` on a fresh root plus
`write_script` is two round trips before any work happens.

**Wanted:** either confirmation that reusing one client across sequential
`open_project` calls is supported (the protocol table does not say a second
`open_project` is legal, and the CLI never does it), or a note that it is
not. Either answer is fine; not knowing means every driver reopens
defensively.

## 6. `progress.json` should carry the run's identity, and the series

**Status: open, small.**

Two gaps, both about `progress.json` being *the* file a supervisor and a
graph node read while not being self-describing.

**Identity.** `progress.json` has everything about the run's *state* and
nothing about what was run: no bundle path, no `task_sha256`, no
`model_sha256`, no seed, no hyperparameters. All of that **is** emitted —
into the terminal JSON line of `train.log`, which carries `task_sha256`,
`model_sha256`, `out`, `parameters`, `witness_error`, `witness_samples`,
`witness_tolerance` and more. So the information exists; it is just in the
log rather than in the artifact.

**Wanted:** that same block written into `progress.json` at *startup* rather
than only into the log at exit. A supervisor needs it while the run is alive
— to refuse a bundle that is not the one it thinks it is dispatching — and a
run that dies never writes the terminal line at all.

**The series — this half was wrong, and is corrected here.**

What this section used to say: *"the curve exists only as prose in
`train.log`, so reading it requires regex over a 210 KB log"*, and it asked
for an append-only `progress.jsonl`.

**The first clause is false.** Every `.cxpolicy` carries the complete
per-iteration series in its header, as structured JSON:

```python
header["training"]["reward_curve"]      # 2 500 entries for 200109's final policy
# {"iteration": 0, "reward_per_step": -0.9412031, "episode_steps": 85.33,
#  "action_std": 0.40019885, "loss": 134.64029}
```

— along with `hyperparameters`, `seed`, `device`, `trainer_sha256`,
`wall_time_s`, `versions` and `randomisation`. 395 KB of the 451 KB file, on
a constant 33 KB of weights: **the header is what grows, not the network.**
The container is `CXPOLICY1\n`, a little-endian `uint64` header length, the
JSON, then the weights.

So the curve is already a first-class artifact, in every checkpoint, and
`progress.jsonl` would duplicate it. This was found while planning experiment
002, which needs 200109's `reward_curve[0:1500]` to check a replication
against — and got it out of the header in three lines.

**What is still true, and is what remains wanted:** the curve is only in a
*written checkpoint*. While a run is alive, the newest curve is as old as the
last `--checkpoint-every`, and a run that dies before its first checkpoint
has no curve anywhere but the log. `progress.json` still holds one sample.

**Narrowed wish:** the identity block above, written into `progress.json` at
startup; and the series available *during* the run — either the same
`reward_curve` array in `progress.json`, or the admission that a supervisor
should read the newest `.cxpolicy` rather than the log.

**Why this matters more than it sounds:** the episode-length column is the
one that revealed that `stand-task-20260802-200109`'s reward peak at
iteration 598 (episode 277.7 of 600) was *not* its survival peak (468.1 at
iteration 1800). That conclusion changes what a supervisor should do — and it
was reachable from the checkpoint header all along, not only by parsing the
log.

## 7. Print the termination mix by default

**Status: open — and mostly a note to ourselves.**

ADR-106: *"`compare.summarise` had collected the termination mix since M9 and
`main` had never printed it; that omission cost three runs."* The distinction
between dying `collapsed` (upright and sinking — the state a recovery passes
through) and dying `tipped` was available the whole time and invisible.

This one is not really a Cadex wish, because `compare.py` is ours now. It is
here as the general form: **a driver that computes a diagnostic and does not
print it is worse than one that never computed it**, because it creates the
impression the question was asked. Recorded so the principle survives the
specific bug.

## 8. `put_asset` moves the revision, and the next `write_script` is refused

**Status: worked around** (`CadexdClient.refresh_revision()`).

Measured while bringing experiment 000's trained policy home. The sequence is
the obvious one and the only one ADR-084 offers:

```python
client.put_asset("pendulum.cxpolicy")          # ok
client.write_script(rig_source + policy_decl)  # STALE_PROGRAM_REVISION
```

with

```
The project script changed after inspection.  [STALE_PROGRAM_REVISION @ precondition]
observed  {"current_revision": "31daf67d…"}
retry     {"required_changes": [{"inspect": "core.inspect scope=script"}]}
```

**The script had not changed.** Storing an asset is not a script write, but
it moves whatever the engine compares `expected_revision` against — so the
canonical "train elsewhere, bring the weights home, declare them" flow fails
on its second step unless the caller knows to re-inspect in between.

The retry hint is correct and machine-readable, which is why this is a
papercut rather than a trap: `inspect scope="script"` returns
`value.revisions.working_revision` and the next write succeeds. The
workaround is four lines.

**Wanted:** `put_asset`'s reply carries the new `revision`, the way
`write_script`, `set_params` and `rebuild` all do. Every other mutating op
tells the caller what the project is now; this one does not, and it is the op
that stands between a trained policy and a verified one.

Related and smaller: `open_project`'s reply comes back with an **empty**
working revision for a project that already has an accepted script, so a
client that trusts it starts out stale.

## 9. `policy_forward` takes the header and weights, not the container

**Status: papercut, worked around.**

`decode_policy(blob)` returns `{"header": …, "weights": …}`. The natural next
call does not take it:

```python
container = cd.decode_policy(blob)
cd.policy_forward(container, observation)                        # ✗
cd.policy_forward(container["header"], container["weights"], observation)  # ✓
```

`verify_policy(container, task, …)` *does* take the container, so the two
functions on either side of it disagree about what a policy is. Trivial once
known, and one line in every evaluator that ever loads a policy.

**Wanted:** accept the container as the first argument, or return a small
object with a `forward(observation)` on it. Either would make the four-call
sequence — decode, verify, load, play — read as one thing.

## 10. The witness margin is thousands-separated on stderr

**Status: worked around** (`harness/runlog.py::WITNESS_RE`).

The trainer prints its margin as prose:

```
witness agrees to 8.761e-08 (1,141x inside the engine's tolerance)
```

The comma is the problem. A parser written against the only margins this box
had seen — `355x`, `430x` — matched `[\d.]+` and **silently found nothing**
on the first run whose margin exceeded a thousand. `supervise` reported "no
witness margin recorded" for a run eleven times inside the floor.

Our regex is fixed. But the general shape is wishlist #6's shape again: the
margin is a *number* that only exists as formatted text, and hazard 13 says
the number under 100× is the one that matters. **Wanted:** `witness_error`
and `witness_tolerance` are already in the terminal JSON blob — put the
factor there too, or put the margin in `progress.json` where a supervisor can
act on it while the run is still going rather than after it ends.

## 11. `--init-from`: start training from an existing policy

**Status: open.** Blocks experiment **B9** outright.

`cadex_train.py` has no resume and no initialise-from-policy option, so every
run starts from a fresh network. Experiment 003's "what to do next" proposes
B9, a warm-start curriculum that walks the disturbance band
0.8 → 1.2 → 1.8 → 2.5 N across four short runs, each initialised from the
last — and that is the thing ADR-100 concluded was unavailable when it
decided a curriculum could not be scheduled. It was right about the
capability and the capability is one flag.

**cdx-rl cannot add it.** `cadex_train.py` lives at
`/home/theo/cadex/training/cadex_train.py`, inside the read-only tree, so
invariant 1 forbids it. 003's README calls the change "contained", which is
true of the diff and not of who is allowed to make it.

**Wanted:** `--init-from <policy.cxpolicy>`, loading the weights out of the
container and into the initial network, leaving the optimiser state fresh.
Fresh optimiser state is the conservative choice and probably the right one:
each leg of a curriculum is a different task, so a carried-over Adam moment
is describing a gradient landscape that no longer exists.

**Why the workaround is unsatisfying:** copying the trainer out of the tree
to patch it would break the one thing that makes runs comparable. `train.py`
pins the trainer by sha256 (`--require-trainer`) precisely because ADR-104
established that the same seed and the same hyperparameters mean nothing if
the update rule differs. A patched copy is a different update rule, and every
comparison to 001, 002 and 003 would have to be re-argued.

**Note also**, and separately from the flag: 003's README says
`tasks/stand-b8/stand10.001150.cxpolicy` "is committed as that starting
network". **It is not committed** — `tasks/stand-b8/` holds `stand-task.json`,
`model-model.xml` and `README.md` only, and no `.cxpolicy` has ever been
tracked in this repository.

The weights do exist on this box, at
`/home/theo/cadex-jobs/stand-task-20260803-140221/` — B8 seed 0's 25
checkpoints, trained on `sb1x`, on a `stand-task.json` and `model-model.xml`
whose digests are identical to the committed ones. So B9 lacks the flag, not
the network. Note the location is the **read-only** jobs directory of
invariant 3: read from it freely, write to it never.

## 12. Let a position actuator's action range be narrower than its joint

**Status: open.** Forced experiment 004-B to edit a derived bundle.

`_ACTION_SOURCES` maps `("position", "angular")` to `angle_limits_degrees`, so
a position servo's action range *is* the joint's declared physical range. The
rationale in `_action_bound` is sound as far as it goes — *"a setpoint outside
them is a command the joint cannot obey"* — but it proves only that the
command range must not **exceed** the joint range. It does not follow that the
two must be **equal**.

**What we hit.** Experiment 003's policies brace by commanding position errors
far past the servo's saturation point: it saturates at 16.4 deg of error
(86 N·mm / 5.236 N·mm/deg) and they command up to 44 deg on a ±45 deg joint.
The natural experiment is to cap what the policy may *ask for* while leaving
the machine's range of motion alone. That cannot be said in the mechanism
vocabulary. Capping `angle_limits_degrees` narrows the joint too — the
exported MJCF's joint ranges are the same ten numbers as the action table's —
which changes the reachable configuration space and confounds the result.

**Wanted:** an optional action bound on the actuator, separate from the
joint's limits, refused if it exceeds them. Something like
`assembly.actuator(..., command_limits_degrees=[-15, 15])`, defaulting to the
joint's range so nothing changes for anyone who does not ask.

**Why it matters beyond one experiment.** A software command limit narrower
than the mechanical range is ordinary practice on real machines, and it is one
of the few levers that acts on hazard 15 without touching the reward — which
hazard 16 says cannot work. It is also the difference between "the policy
chooses to saturate" and "the dynamics force it", two findings that call for
completely different next steps.

**The workaround** is `experiments/004-ceiling-and-clamp/make_clamp_bundle.py`:
derive the bundle, cap the action table, copy the MJCF unchanged. It works —
the MJCF sets `ctrllimited="false"` on every actuator, so nothing downstream
re-clamps — but the result is not reproducible from the script, which is the
one property this project most wants to keep.

## 13. The pinned engine cannot build the mechanism its own experiments ran on

**Status: open.** Makes `mechanisms/mg-legs/script.py` unbuildable on the only
box that has a GPU.

`script.py` declares two observation kinds its own comments number **ninth**
and **tenth**:

```python
assembly.observation(pelvis_c, "centre_of_mass_velocity",    name="cv"),   # mjSENS_SUBTREELINVEL
assembly.observation(pelvis_c, "centroidal_angular_momentum", name="cam"),
```

Neither string appears anywhere in the pinned checkout. `06d1374b`'s
observation table has twelve entries and stops at the eighth kind:

```
actuator_force, armature, centre_of_mass, component_angular_velocity,
component_linear_velocity, component_orientation, component_position,
damping, friction_loss, mass, position, velocity
```

So the mechanism builds on the **laptop** (`560935bd`) and nowhere else.
`sb1x` at `06d1374b` refuses `centre_of_mass_velocity`; `sb9x` at `ae8da6a6`
got past that one and refused `centroidal_angular_momentum`.

**Why this is worse than it sounds.** CLAUDE.md's headline recovery story is
that finding `script.py` made the mechanism *"changeable rather than a dead
end"*. It is changeable only on a machine with no GPU and no training venv.
Every mechanism edit has to be authored on the laptop, exported, and carried
to `sb1x` as bytes — which is exactly the `stand-b2` situation the recovery
was supposed to end, one step removed.

**Wanted, in preference order:**

1. **A pin that can build what it trains.** Either the two observation kinds
   backported to `06d1374b`, or a statement of the earliest revision that has
   both, so cdx-rl can move its pin deliberately rather than discover the gap
   per-box. Note the constraint: bumping the *trainer* breaks comparability
   with 001–003 (ADR-104), but the ten commits between `06d1374b` and
   `ae8da6a6` were all engine-side with `training/` byte-identical, so an
   engine-only bump is available if the boundary is stated.
2. **A build-time capability check.** `assembly.observation` with an unknown
   kind should name the kind, the revision, and the supported set. Today the
   failure surfaces partway through a long script, which reads like a
   modelling error rather than a version gap.
3. **`cadexd` should report its observation vocabulary**, so a driver can
   refuse before it spends a build.

---

## Withdrawn

### ~~A `cadex train` subcommand~~

**Withdrawn before it was filed.** ADR-084 decided training happens
elsewhere, and there is deliberately no train verb and no train button. The
reasoning holds: the engine ships without a line of JAX,
`test_engine_purity_guardrails` asserts no `jax` or `mjx` reaches the staged
payload, and the pixi environment stays buildable on a box with no CUDA. A
train verb would put a GPU dependency inside a CAD kernel's CLI.

The thing that actually hurts is not the absent verb; it is that the
*supervision* around training has no home either. That home is cdx-rl. See
[`harness/DESIGN.md`](harness/DESIGN.md).
