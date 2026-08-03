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
