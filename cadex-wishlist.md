# cadex-wishlist.md — things cdx-rl wants from Cadex

**This file is no longer a queue. It is the record.**

It was written under a policy that ended on 2026-08-05: cdx-rl was read-only
toward Cadex, so when it wanted something it wrote the want down here and
somebody else was supposed to act on it. Nobody did. Thirteen entries
accumulated, and three of them became the binding constraint on the research
— ahead of the GPU, which is the point at which a documentation practice has
started costing more than it buys.

**The policy now: work in the PR clone at `/home/theo/cadex-prs` and submit
pull requests.** Never push to `theo-kirby/cadex`; never touch the operator's
tree at `/home/theo/cadex`. `cadex-engine-plan.md` scopes the blocking three
as PR specs.

**Seventeen entries now, and five have merged** — #11 and #12 as GitHub PRs
[#2](https://github.com/theo-kirby/cadex/pull/2) and
[#1](https://github.com/theo-kirby/cadex/pull/1) on 2026-08-05, then #15, #16
and #17 as [#3](https://github.com/theo-kirby/cadex/pull/3),
[#4](https://github.com/theo-kirby/cadex/pull/4) and
[#5](https://github.com/theo-kirby/cadex/pull/5) the same day. **Mind the two
numbering schemes**: an entry number here is not a GitHub PR number and never
has been.

**#16 and #17 are the first entries this file did not accumulate.** Both were
found and closed inside one session, and both were *created* by the entry
before them — #15's fix orphaned every trained policy, which is #16; #16's
surface could not reach the store, which is #17. A file that only collects
wants would have recorded the first and shipped the other two broken.

**Why keep the file.** Two reasons, and neither is sentiment. Each entry
records *what the gap actually cost* — measured, in hours and in wrong
conclusions — which is the evidence a PR body needs and which nobody will
reconstruct later. And several entries argue that the current behaviour is
**right and the wish is wrong**; Cadex has 106 ADRs behind its shape and
things that look like gaps are frequently decisions. #3 (no trainer-side
early stop) is the clearest: ADR-099 says you do not select the final
iteration, 001 measured survival against reward at −0.34 after the peak, and
a PR "fixing" it would make the tool worse. Read the entry before opening
anything.

**Status is one of:**

| status | means |
|---|---|
| **open** | Still wanted, no PR yet. Says what it costs. |
| **PR #N** | A pull request is up against `theo-kirby/cadex`. Links it. |
| **merged** | Landed upstream. Says at which commit, and what in cdx-rl can now be deleted. |
| **worked around** | cdx-rl solved it locally and the workaround is fine. Not a PR candidate unless the workaround starts costing. |
| **withdrawn** | We were wrong, or the current behaviour is a decision. Says why — this is the most valuable status in the file. |

The old vocabulary had **filed**, which meant "written down here", which
under the old policy meant nothing had happened. It is gone; entries that
carried it are **open**.

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

**Status: withdrawn — the current behaviour is right, and this wish was
wrong.** cdx-rl built the supervisor and it belongs on this side.

The entry stands as written below because the *observation* is accurate: the
trainer computes `best_iteration` every iteration and keeps going regardless.
What the entry got wrong was the conclusion. ADR-099 says you do not select
the final iteration, and experiment 001 measured survival against
`reward_per_step` at **r = +0.06** over a whole run and **−0.34** after its
peak — so a reward-patience stop would reliably stop at the wrong place. 002
confirmed it across three seeds. A trainer-side early stop keyed on the only
scalar the trainer has would make Cadex worse; **do not open this PR.** What
a supervisor must stop on is divergence, device and liveness, which are
`harness/supervise.py`'s business and need nothing from the engine.

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

**Status: MERGED 2026-08-05** — [theo-kirby/cadex#2](https://github.com/theo-kirby/cadex/pull/2),
landed on `main` at `75efe784` (ADR-124).
*(Wishlist #11, GitHub PR #2 — the two numbering schemes are unrelated.)*
**`training/cadex_train.py` is now `4c1f24f8bdf2368a…`, not `aacfa823…`** —
every `--require-trainer aacfa823…` in these docs and in older run records
now refers to a trainer that is one merge behind. `method.md` §8b's bridge
run is the protocol for anything that must compare across it.
**Experiment 005 already ran under `4c1f24f8…`**, which is what `main` now
carries, so 005 needs no bridge.
(`cadex-engine-plan.md` §1; body in `prs/11-train-init-from.md`.) Blocks
experiment **B9** outright, and halves experiment 005: ~9.9 h without it,
~5 h with. Not the first PR out — it changes `cadex_train.py`'s sha256 and so
invalidates every downstream trainer pin, including `remote_train.sh`'s own
check, which is fine under the new policy but wants #12 to land first.

`cadex_train.py` has no resume and no initialise-from-policy option, so every
run starts from a fresh network. Experiment 003's "what to do next" proposes
B9, a warm-start curriculum that walks the disturbance band
0.8 → 1.2 → 1.8 → 2.5 N across four short runs, each initialised from the
last — and that is the thing ADR-100 concluded was unavailable when it
decided a curriculum could not be scheduled. It was right about the
capability and the capability is one flag.

**cdx-rl can now add it, and should.** This paragraph used to read *"cdx-rl
cannot add it — `cadex_train.py` is inside the read-only tree, so invariant 1
forbids it,"* and noted that 003's README calling the change "contained" was
true of the diff and not of who was allowed to make it. That is no longer the
constraint: the work happens in `/home/theo/cadex-prs` and lands as a PR.
`cadex-engine-plan.md` §1 scopes it, including the four things that make it
more than one flag.

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

**Status: MERGED 2026-08-05** — [theo-kirby/cadex#1](https://github.com/theo-kirby/cadex/pull/1),
landed on `main` at `cfa6640e` (ADR-123).
*(Wishlist #12, GitHub PR #1 — the two numbering schemes are unrelated.)*
**What this retires:** `experiments/004-ceiling-and-clamp/make_clamp_bundle.py`
is no longer the only way to express the clamp. A `script.py` variant passing
`command_limits_degrees` reproduces its action table on every number, all ten
actuators; the only differing field is `source`, which now honestly says
`command_limits_degrees`. Keep the script as the reproduction record of what
004 actually ran.
(`cadex-engine-plan.md` §2; body in `prs/12-position-command-range.md`.)
Forced experiment 004-B to edit a derived
bundle. It is ahead of #11 because it is smaller, touches no trainer, and is
independently useful: it turns 004's result from a bundle hack into something
reproducible from `script.py`.

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

**Status: merged upstream — resolved 2026-08-05 by getting current.** No PR
was needed; the kinds were already there and this clone could not see them.

`centroidal_angular_momentum` / `mjSENS_SUBTREEANGMOM` landed in **`593f64e6`
(ADR-116, 2026-08-03)** and `centre_of_mass_velocity` before it. Both are in
`origin/main`. The old checkout could not see either because `06d1374b` was
the newest object it had — which is why `git log --all -S…` came back empty
and why this entry hedged between "not written yet" and "not fetched yet".
**"Searched and not found" was a claim about one clone**, the same lesson the
`script.py` recovery taught about machines.

`mechanisms/mg-legs/script.py` now builds on sb1x. Preference-order item 1 —
*"a statement of the earliest revision that has both"* — is answered: it is
`593f64e6`, and the engine-only bump is safe because `training/` is
byte-identical across the whole 15-commit range.

Items 2 and 3 below (a build-time capability check, and `cadexd` reporting its
observation vocabulary) are **still open and still worth a PR.** They are what
would have turned this week's confusion into one clear line of output.

**Two things the unblock did *not* fix**, both now separately tracked: the
worker's default budgets refuse the build (#14 below), and the script's
`assembly.policy` output fails a task-digest check for reasons that are
cross-platform floating point rather than a real mismatch (#15).

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

## 14. The worker's default address-space cap makes a real assembly 500× slower

**Status: open — the strongest PR candidate in this file**, because it is a
measured pathology rather than a missing feature, and because at the default
budgets Cadex cannot build a mechanism it shipped experiments for.

**Its two `test_dynamics_collision.py` failures are INTERMITTENT, measured
2026-08-05.** Three runs of that file alone, no code change between them:
**2, 1, 2** failures. So a statement of the form "the suite baseline is exactly
two failures" is slightly wrong, and a branch comparison should read that file
as **1–2**. The defect is resource-dependent, which is consistent with it being
an address-space cap rather than a logic error.

`cadex_domain_worker.py::_resource_limits` applies the scripted budgets as
real `setrlimit` calls: `RLIMIT_AS` from `memory_limit_mb`, `RLIMIT_CPU` from
`timeout_seconds`, plus a hard-coded `RLIMIT_NOFILE` of 64. The engine's
defaults are **300 s and 6144 MB**
(`CadexEngineSettings.DEFAULT_SCRIPTED_*`).

**Measured on sb1x, 2026-08-05.** Same script
(`mechanisms/mg-legs/script.py` with its policy output removed), same fresh
project, same engine at `b169a092`. The only variable is `memory_limit_mb`:

| `RLIMIT_AS` | outcome | CPU used | RSS | `memory_exceeded` |
|---|---|---|---|---|
| 6144 MB (default) | `SIGXCPU` — never finishes | 107 s to hit a 120 s cap, and 1787 s to hit an 1800 s cap | 217 MB | `false` |
| 32768 MB | **succeeds** | **8.2 s** | 218 MB | `false` |

Two things make this worth fixing rather than working around:

* **Memory is never the constraint.** RSS is ~218 MB against a 6 GB cap, and
  the engine's own `memory_exceeded` flag stays `false`. What is scarce is
  *address space*, not memory.
* **The cost is system time, not work.** The failing runs spend ~80 % of
  their CPU in the kernel — `user 6m20s / sys 23m40s` on the 1800 s run,
  `user 29s / sys 1m31s` on the 120 s one — which is the signature of an
  allocator retrying failed `mmap`s rather than of geometry being computed.
  The same geometry takes 8 seconds when the cap is lifted, so the honest
  ratio is ~500×.

**Cadex's own test suite demonstrates it.** This is not a cdx-rl-shaped
complaint. `pixi run test-engine` at `b169a092` on sb1x is **1507 passed, 2
failed, 22 skipped**, and both failures are the same cap:

```
src/Mod/cadex/cadex_tests/test_dynamics_collision.py
  test_a_real_concave_part_is_refused_with_the_numbers_in_it   FAILED
  test_the_same_bracket_is_accepted_when_the_script_says_hull  FAILED

error: …/mujoco/plugin/libelasticity.so: failed to map segment from
       shared object
```

`_live_dynamics` (`:736`) calls `open_project` with no budgets, so those tests
run at the 6144 MB default, and MuJoCo's plugin cannot `dlopen` — *"failed to
map segment"* is an `mmap` refusal, the same mechanism as the slowdown above.
Patching that one call to pass `memory_limit_mb: 32768` and re-running:
**11 passed** (the probe was reverted; the clone is clean). Same code, same
box, only the budget differs.

That makes the fix upstream's rather than ours, and gives the PR a
regression it can point at.

**And the failure is unreadable.** It surfaces as `DOMAIN_WORKER_NO_RESULT`
at stage `external_process` with `returncode: -24`, inside an `observed`
blob whose `stdout` is several kilobytes of OCCT `Processing......` progress
bars. Nothing says "CPU limit"; signal 24 is `SIGXCPU` and you have to know
that. It reads like a hang or a crash.

**Wanted, in preference order:**

1. **Do not cap `RLIMIT_AS` from `memory_limit_mb`.** Address space is not
   memory. If the intent is to stop a runaway script eating the box, cap RSS
   (cgroups, or `RLIMIT_DATA`), or raise the default far above any plausible
   working set.
2. **Name the signal.** A worker killed by `SIGXCPU` or `SIGSEGV` should say
   so in `failure_code` — `DOMAIN_WORKER_CPU_EXCEEDED` — rather than
   `NO_RESULT` with the evidence buried in `observed.returncode`.
3. **Defaults that fit a real mechanism.** 300 s is comfortable for the
   assemblies in `smoke.py` and does not fit a ten-joint biped even once the
   address-space pathology is gone.

cdx-rl's side is done and is not a substitute for the fix: `open_project`
sends both budgets (`tools/cadexd_client.py::DEFAULT_WORKER_*`), and
`harness rebuild` exposes `--worker-cpu-seconds` / `--worker-memory-mb`.

## 15. A task digest is not stable across platforms, and it gates policy replay

**Status: MERGED — [PR #3](https://github.com/theo-kirby/cadex/pull/3),
ADR-133, 2026-08-05.** The third option below is what landed: a snap to zero on
inertial coordinates below **one nanometre**, absolute. Cost while it was open:
`script.py`'s own `assembly.policy` output could not be rebuilt on the box that
trained the policy.

`mechanisms/mg-legs/script.py` declares
`assembly.policy(stand, weights="stand10.cxpolicy", sha256=…)`. Rebuilding it
on sb1x is refused:

```
policy output 'balance' was trained on a task bundle whose digest is
'5572adf265aa51cb…', and the task it is declared against digests to
'0b4d160cd436fd16…'.
```

The refusal is **correct in principle** — ADR-level reasoning that a policy is
only meaningful for the task it was trained on, and the correction text says
so well. It is **wrong in this instance**, and the reason is worth recording
because it is not a mechanism change at all.

Diffing the generated bundle against the committed `tasks/stand-b8/`:

* the **task JSON differs in exactly one line**, and that line is the
  embedded MJCF `sha256`;
* the **MJCF differs in exactly one line** — the pelvis inertial `pos`
  x-component, `5.10066e-11` (built on the macOS laptop) against
  `5.10087e-11` (sb1x, Linux). That is **2.1 × 10⁻¹⁵ m**, on a quantity that
  is mathematically **zero**: the machine is symmetric, so the pelvis centre
  of mass sits on the plane, and both numbers are rounding noise around it.

Mass, quaternion and diagonal inertia are bit-identical, as is every other
line of the 14179-byte file. So a cross-platform ULP in a coordinate that
should be zero propagates through two digests and refuses a valid policy.

**Wanted:** not a looser check — the check is right. Either a documented
statement that bundle digests are platform-specific (so a policy travels with
its bundle and `tasks/` is the unit of provenance, which is what cdx-rl
already does), or geometry output that is reproducible across platforms, or a
`snap-to-zero` on inertial coordinates below a tolerance where the value is
symmetry noise. The first is cheap and honest; the third is what a CAD kernel
arguably owes a digest-based contract.

### How it was disposed of

**The third.** ADR-133 snaps centre-of-mass components below one nanometre to
exactly `0.0`, inside `body_inertial` where both publications read the number.
The rule is **absolute**, and that is the part worth carrying: the two readings
differ in their *fifth significant figure*, so no relative tolerance sees them
as equal. Cancellation is why — a symmetric body's x-centroid is a difference
of near-equal sums, so a last-bit disagreement in OCCT's own per-solid readings
arrives amplified by eleven orders of magnitude. `math.fsum` is
correctly-rounded, so no summation order fixes it either.

Measured after, same script, byte-identical engine sources:

| | script build digest | MJCF | task bundle |
|---|---|---|---|
| macOS 26, arm64 | `560a33a4bfce810e…` | `203f746e9bb8a857…`, 14 169 B | `6dc1c580f4bcd01a…` |
| Ubuntu 24.04, x86-64 | `560a33a4bfce810e…` | `203f746e9bb8a857…`, 14 169 B | `6dc1c580f4bcd01a…` |

`cmp` reports both pairs identical.

**Two things the fix did not do**, both stated in the ADR rather than
discovered later:

* it does not snap mass or the inertia tensor. A symmetry-zero product of
  inertia has the same cancellation problem, and a nanometre is not a tolerance
  for kg·m² — that bound would have to be relative, which is a different
  decision. This mechanism does not hit it: both platforms print identical
  `quat` and `diaginertia`.
* it does not make a *simulation* reproducible. The MJCF and the bundle now
  agree byte for byte; the rollout trace does not (`d7cf5c5faa19f171` against
  `d598a51eb615483f`, same 152-frame episode), and the policy receipt differs
  in `witness_error` at the tenth significant figure. That is MuJoCo's
  disclaimed cross-platform reproducibility (hazard 3), it is why the far
  machine rebuilds rather than receiving a store, and no snap can fix it.

**And it moved every model digest**, which orphaned every already-trained
policy — see #16.

---

## 16. A whole-file task digest conflates a different task with a different route

**Status: MERGED — [PR #4](https://github.com/theo-kirby/cadex/pull/4),
ADR-134, 2026-08-05.** Raised and disposed of in the same session as #15,
because #15's fix created it.

`verify_policy` check 1 hashes the whole task bundle. Two things that are *not*
mechanism changes therefore refuse a valid policy:

* **#15's own fix.** Snapping inertial coordinates changes every model digest,
  so every bundle embedding one, so every policy trained before it.
* **ADR-131's honest provenance string.** `tasks/stand-b8-clamp25/` was
  produced by editing the derived bundle by hand, and reports
  `actions[].source` as `angle_limits_degrees` — the joint's limits, which are
  not where ±25° came from. The script that now produces the same arm from
  source reports `command_limits_degrees`. All ten actuators, `low`, `high`,
  `unit` and `scale` are identical; `label` and `source` are not, and they move
  the hash:

```
trained on        3d627ef4b9a509fe…
declared against  bd8071b50360eaab…
```

Cost while it was open: **retrain (4–5 GPU-hours a seed) or revert a
correctness fix.** `mechanisms/mg-legs/rollout/README.md` said the only route
was to retrain, and it was wrong.

**Wanted, and what landed:** `assembly.policy(..., trained_task=)`. The policy
is bound to its own travelling bundle whole-file — *unweakened* — and the
script-built bundle is then proved **equivalent**: every behaviour-deciding
field, plus the two models compared as models rather than as hashes. That is
**stronger** than making check 1 semantic, which was the obvious version and
would have weakened every policy in the system to buy compatibility for a few.

The model comparison is the half that matters. Two bundles can agree on every
number while naming different mechanisms — same joint names, same limits,
different masses — and the action table would match perfectly. A 0.4 mm bracket
plate changes **no field of the task bundle at all** and is caught only there.

It needed **no change to `training/cadex_train.py`**, which is why no trainer
digest moved and no bridge run is owed.

---

## 17. The project store will not hold what a policy travels with

**Status: MERGED — [PR #5](https://github.com/theo-kirby/cadex/pull/5),
ADR-135, 2026-08-05.**

`put_asset` accepted `.cxpolicy`, `.obj`, `.ply` and `.stl`. #16's whole surface
needs a `.json` bundle and a `.xml` model in `assets/`, so **ADR-134 shipped
unusable and all 52 of its unit tests passed.** The first end-to-end replay
refused at step one:

```
ASSET_REJECTED at precondition
'clamp25-task.json' is not one of the formats this project store holds
['.cxpolicy', '.obj', '.ply', '.stl']
```

Every one of those tests exercised a pure function — `task_semantic_digest`,
`task_differences`, `model_differences`, the API's argument validation. Not one
went through `store_project_asset`.

**The lesson is one this repository already had, one level over.** `method.md`
says validate at length, not at three iterations, because every fault it names
is scale-dependent. A surface whose unit tests all pass and whose first real use
fails at step one was tested at three iterations.

**Wanted, and what landed:** a third suffix constant,
`_PROVENANCE_ASSET_SUFFIXES = {".json", ".xml"}`. Not two more members of
`_ASSET_SUFFIXES`, which must stay exactly three because the shell mirrors it by
name (ADR-091) — the same reason `_POLICY_ASSET_SUFFIXES` is already separate.

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
