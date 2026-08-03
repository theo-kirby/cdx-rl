# 000 — loop validation

**Status: specified. The modelling half is verified; the training half is
not run.**
**No GPU time. CPU only, minutes.**

---

## 1. Question

**Does every link in the chain hold, end to end, on this box alone?**

```
Cadex script → MJCF → task bundle → trainer → .cxpolicy
             → witness verification → local playback → a graph node with artifacts
```

This is not a research question and it is not pretending to be one. It is the
integration test that makes every later result trustworthy, and it exists
because a chain that has never been run end to end has an unknown number of
broken links — and you find them one at a time, each after an expensive run.

Both answers are interesting: if it holds, everything after is cheap; if it
breaks, it breaks somewhere specific and cheap to fix.

## 2. Metric

**Binary pass/fail per link**, plus one number: **the digest is identical
across two rebuilds.**

There is no reward target and no survival target here, deliberately. A
1-DOF pendulum learning to hang still is not a result — it is a receipt. The
temptation to report "reward went up" as a finding is exactly what
`method.md` exists to resist, and it would be a strange place to start
ignoring it.

The one quality bar that *is* a real check: **the trainer's witness margin
must exceed 100×**. Under that, stop and read `MUJOCO.md` hazard 13 rather
than continuing (the witness records what the GPU rounded the network to, not
what the network is).

## 3. Mechanism

A **1-DOF pendulum**: a grounded post with an arm on a single revolute joint.
`rig.py` in this directory, 53 lines.

```
post  40 × 40 × 300 mm, steel (7850 kg/m³), grounded
arm   arm_length × 20 × 20 mm, aluminium (2700 kg/m³), parametric 50–400 mm
hinge revolute, ±180°, damping 0.5 N·mm·s/deg
motor torque, torque_limit_nmm = 2000
obs   hinge position, hinge velocity
```

**Actuator limit models the mechanism, not hardware.** There is no bench
part; 2000 N·mm is chosen to be comfortably able to lift the arm, because
this experiment is about plumbing and an under-powered actuator would turn a
plumbing test into a control problem.

### Verified on 2026-08-02, sb1x, Cadex `06d1374b`

Authored through `cadexd` with no display anywhere:

```
digest b77a1f5a39523b8d839c790ddb4a67367c04b084f496a48332acdb4db3ff7575

brep                          post_solid   output-000.brep       2 594 B
brep                          arm_solid    output-001.brep       2 594 B
assembly_mjcf_xml             model        model-model.xml       1 259 B
assembly_training_task_json   task         task-task.json        2 648 B
```

The MJCF is self-contained, in radians and SI, with mass and inertia from the
solids (`mass="3.768" diaginertia="0.0287624 0.0287624 0.0010048"` for the
post), a `<keyframe name="solved">`, `<sensor>` elements for both declared
observations, and `forcerange="-2 2"` derived from `torque_limit_nmm=2000`.
The bundle is `cadex-training-task-v1` and records
`"mujoco_version": "3.10.0"` — the trainer venv's exact pin.

**So links 1 and 2 are proven.** Links 3 onward are not.

## 4. Task

```
episode_seconds  2.0
control_hz       50            → 100 actions per episode
actions          [the hinge motor]
reward           -abs(hinge)   weight 1.0     hang down
                 1.0           weight 1.0     alive
termination      none
reset_variation  none          ← see below
disturbance      none          ← see below
```

**Reset variation and disturbance are deliberately absent, and this is the
one place in the repository where that is allowed.**

`method.md` step 6 (hazard 16) says a task whose word for success is
"balance", "hold" or "stand" *must* declare both, because a task in which
nothing ever changes cannot tell balancing from bracing. This task's word for
success is **"the pipeline ran"**. There is no posture to brace into and no
recovery to fake. Adding variation here would add a source of failure to a
test whose purpose is to isolate failures.

**Every subsequent experiment declares both.** If this exemption is ever
copied into an experiment with a real question, that is a mistake and this
paragraph is the thing to point at.

Capture-point arithmetic (`method.md` §6b) does not apply: nothing is shoved,
and the mechanism has no support polygon — it is bolted to the world.

## 5. Gate

`feasibility`'s six checks are **not run**, and this is the second and last
exemption.

Four of the six are about a floating base standing on a floor: gravity
compensation, shove rejection in place, contact sanity, and the zero-torque
drop test. A grounded pendulum has no floating base and touches no floor.
Running them would produce four green rows that measured nothing — which is
hazard 18 exactly, and worse than not running them.

What *is* checked instead, and is a genuine gate:

* `tools/smoke.py` passes (engine, dynamics surface, digest stability,
  trainer venv, GPU, checkout commit).
* The MJCF loads in stock MuJoCo from the trainer venv, and reports 1 dof and
  1 actuator.
* At zero torque from the `solved` keyframe, the arm **falls** and settles
  hanging. A pendulum that does not fall under gravity is a pendulum with a
  units bug, and this is the cheapest possible detector for one.

## 6. Budget and stopping rule

| | |
|---|---|
| device | **CPU** — `--allow-cpu`, or simply no CUDA visible |
| iterations | 200 |
| environments | 256 |
| checkpoint-every | 20 |
| expected wall | single-digit minutes |
| **GPU-hours** | **zero** |

Stopping rule: 200 iterations, or `supervise` stopping earlier. There is no
patience threshold worth setting on a run this short — but the supervisor
runs anyway, because **this experiment is also the supervisor's integration
test**.

## 7. Pass criteria

Written before the run. Each is a link.

| # | Link | Passes when | |
|---|---|---|---|
| 1 | script → engine | `rig.py` accepted; digest recorded | ✅ `b77a1f5a…` |
| 2 | engine → artifacts | `assembly_mjcf_xml` and `assembly_training_task_json` resolve to files that exist | ✅ 4 artifacts |
| 2b | determinism | two rebuilds produce **identical digests** | ✅ `rebuild --verify` |
| 3 | MJCF → MuJoCo | the XML loads in stock MuJoCo 3.10.0 and reports 1 dof, 1 actuator, 2 sensors | ✅ nv=1 nu=1 nsensor=2 |
| 4 | bundle → trainer | `cadex_train.py` accepts the bundle and completes 200 iterations with no error | ✅ exit 0, 48.3 s, CPU |
| 5 | trainer → policy | a `.cxpolicy` is written, and **every witness margin printed is > 100×** | ✅ 8.761e-08, **1 141×** |
| 6 | policy → verification | `assembly.policy(task, weights=…, sha256=…)` is accepted on rebuild — the engine re-checks the bundle digest, the model, the observation channels in order, the action table, and re-evaluates the witness in float64 | ✅ receipt published |
| 7 | policy → playback | `assembly.rollout` produces a trace locally; frame count and hinge angle over the episode are reported | ✅ 52 frames, 0.05° |
| 8 | supervisor | `supervise` produces a terminal report with peak vs final, the checkpoint inventory with sha256s, and the witness margins | ✅ live + post-mortem |
| 9 | evaluator sanity | **the same checkpoint played twice produces the same row** (ADR-103 §9) | ✅ 24 episodes, 1 worker |
| 10 | record | a Flywheel node under the cdx-rl root, carrying `progress.json`, the chosen `.cxpolicy`, the MJCF and the bundle as artifacts | ✅ see §9 |

**Explicitly not a pass criterion: the reward going up.** If the reward
climbs, fine. If it does not, links 1–10 can still all pass and the
experiment still succeeds. A 1-DOF pendulum with a hang-down reward is not a
learning problem worth grading.

Criterion 9 is the one most likely to fail quietly, and it is the two-second
test that found the ADR-103 bug. Do not skip it because the model has no
domain randomisation — the point is that the evaluator reloads.

## 8. What happened

**All ten links pass. 2026-08-02/03, sb1x, cdx-rl `b1e90af8`+, Cadex
`06d1374b` (clean). Zero GPU-hours.**

Raw output in [`results/`](results/): the gate checks, the `supervise`
post-mortem, the `compare` table, and the `bring_home` receipt.

### Links 1–2b — the engine half (re-verified, not assumed)

`rebuild --verify` reproduced digest
`b77a1f5a39523b8d839c790ddb4a67367c04b084f496a48332acdb4db3ff7575`
**twice** and resolved four artifacts at the sizes recorded in §3.

### Link 3 and the gate — the physics is real

`experiments/000-loop-validation/checks.py`, under the trainer interpreter:

```
ok  mjcf_loads_1dof_1actuator_2sensors  nq=1 nv=1 nu=1 nsensor=2 nbody=3 njnt=1
ok  bundle_names_this_model             e6378deb… == e6378deb…
ok  falls_under_gravity_at_zero_torque  swept 97.664° in 2.0 s, settling at 90°
```

**The drop test earned its place.** From the `solved` keyframe at zero
torque the arm swings 97.7° and hangs. A pendulum that did not fall would
have been a units bug, and every number after it would have been a number
about the wrong mechanism.

### Links 4–5 — the trainer, on CPU

`tools/train.py` dispatched `cadex_train.py` with `JAX_PLATFORMS=cpu` into
`jobs/pendulum-20260803-001327/`, the bundle and its model copied in beside
it:

| | |
|---|---|
| device | **cpu** |
| iterations | 200 of 200, exit 0 |
| wall | **48.3 s** (62.0 s including startup) |
| reward/step | −46.66 at iteration 0 → **−0.050** at 194 |
| checkpoints | 19 records over 10 files, plus the final |
| **witness** | **8.761e-08, 1 141× inside tolerance** |

`supervise --watch` attached live and printed the curve as it ran. It was
given `--patience 0` deliberately: experiment 001 Phase A found reward
patience stops runs that are working, and this run has nothing to stop.

**The witness margin exposed a bug in `supervise`.** The trainer
thousands-separates the factor, so a very good margin prints as `1,141x` —
and `WITNESS_RE`'s `[\d.]+` matched only margins under a thousand. The first
version of this parser reported *"no witness margin recorded"* for a run
whose margin was eleven times the floor. A check that quietly finds nothing
is worse than no check, and this one was caught only because the run it was
pointed at happened to be unusually clean.

### Link 6 — the round trip, which nothing had tested

`experiments/000-loop-validation/bring_home.py`: `put_asset` the
`.cxpolicy`, then a second script revision declaring
`assembly.policy(task, weights="pendulum.cxpolicy", sha256="e0fc4d42…")`
and `assembly.rollout(policy, frames_per_second=25, seed=0)`.

**Accepted.** Digest `9a265fb7…`, eleven outputs, and a
`assembly_policy_receipt_json` carrying the engine's own re-verification:

```
schema                cadex-policy-receipt-v1
task_sha256           a0999c23…      (matches the bundle)
model_sha256          e6378deb…      (matches the MJCF)
observation_channels  ["hinge", "hinge_rate"]     in order
action_count          1
parameters            4417
witness_error         8.761484508568174e-08
witness_tolerance     0.0001
```

The engine re-evaluated the trainer's witness **in float64** and got the same
number the trainer reported in float32. That is the link the whole experiment
existed to test, and it holds.

**One real bug in the spine, found here.** `put_asset` moves the project
revision without being a script write, so the very next `write_script` was
refused with `STALE_PROGRAM_REVISION` — *"The project script changed after
inspection"* — even though the script had not changed. `CadexdClient` now has
`refresh_revision()` and calls it after `put_asset`, and `open_project` calls
it when the reply comes back without a working revision (which, measured, it
does for a project with an accepted script).

### Link 7 — playback

`assembly.rollout` produced `assembly-simulation-trace.json`: 52 frames (one
`input` plus 51 `solver_output` at 25 fps over 2.0 s), 100 control steps,
`truncated: true`, total reward 71.60.

The hinge angle over the episode:

| t (s) | hinge | command (N·mm) |
|---|---|---|
| 0.00 | 0.00° | — |
| 0.20 | 0.30° | 1 555 |
| 0.40 | 0.12° | 1 362 |
| 1.20 | 0.05° | 1 426 |
| 2.00 | 0.05° | 1 429 |

**The policy learned to hold the arm horizontal**, which is exactly what
`-abs(hinge)` asks for: hinge = 0 is the reward maximum, and holding a 200 mm
aluminium arm out against gravity takes ~1 428 N·mm of the 2 000 available.
The drop test says the same mechanism falls 97.7° when that torque is
removed. The two measurements agree, which is the point of having both.

### Links 8–9 — the instruments, on a mechanism they were not written against

`compare` ran over all 11 checkpoints × 12 seeds in **1.0 s**. The
same-file-twice test passed. Every episode survives — this task declares no
termination at all — so survival is uninformative here, and the driver says
so rather than picking a winner: *"THE WINNER IS NOT SEPARATED FROM THE
RUNNER-UP … indistinguishable at this seed count"*, listing all eleven.

That is the right answer, and getting it required no special-casing. Two
things did need fixing, both found by pointing the driver at a task unlike
the one it was built for:

* `tilt` and `drift` printed as `nan` for a task that declares neither a
  `tipped` termination nor a `drift` reward term. They now print `—` with a
  line saying the task does not define them — an empty cell must not read as
  a measurement that went wrong.
* The footnote hard-coded `it terminates above 0.15`, which was the biped's
  threshold quoted at a pendulum. It now reads the bundle's own bound, or is
  omitted.

**And the torque columns did their job on a mechanism with no hazard-15
story.** `pendulum.best` holds the hinge at a **mean of 1 566 N·mm — 78 % of
its 2 000 N·mm limit — on 47 % of frames above 90 %**, and `compare` flagged
it. That is not a fault here: §3 says this actuator limit models the
*mechanism* rather than hardware, and holding an arm out horizontally is
supposed to cost most of it. But the flag is correct, it fired unprompted on
a mechanism the driver had never seen, and had this been a bench part it
would have been the finding.

## 9. What it means

**The chain holds end to end on this box alone, and it costs 62 seconds.**

That is the entire result, and it is what every later number rests on. Before
this run, six of the ten links had never been executed in sequence anywhere,
and the honest state of the repository was "an unknown number of broken
links, each of which will be found after an expensive run".

What the run also established, none of which was the question:

1. **CPU training is free and real.** There is no trainer-side CPU guard —
   `--allow-cpu` belongs to `remote_train.sh`, not to `cadex_train.py` — so
   `JAX_PLATFORMS=cpu` is the whole of it. A 200-iteration, 256-environment
   run of a 1-DOF mechanism takes 48 seconds. Any future integration test can
   be a real training run rather than a mock.
2. **Three defects were found by running, not by reading**, and all three
   were in code that had already been reviewed: the witness regex missing
   any margin over 999, `put_asset` invalidating the revision, and
   `compare` printing `nan` for undeclared columns. None would have been
   caught by a test that only exercised the biped.
3. **The engine's re-verification is not a formality.** It re-ran the witness
   in float64 against the trainer's float32 record and matched to 8.8e-08.
   A policy whose weights survived the trip but whose architecture the engine
   read differently would have been refused here, with a receipt naming
   which of the six claims failed.

### What it does not mean

* **Not that the reward went up — and that was never a criterion.** It did
  (−46.66 → −0.050), and it is a receipt, not a result. A 1-DOF pendulum
  learning to hold still is not a learning problem worth grading, and §7 said
  so before the run.
* **Not that the drivers are correct in general.** They are correct on two
  mechanisms now instead of one. The three defects above are the measure of
  how much that second mechanism was worth.
* **Not that the exemptions in §4 and §5 generalise.** This task declares no
  reset variation, no disturbance and no termination, and that is legitimate
  *only* because its word for success is "the pipeline ran". Experiment 001
  is what a real question looks like.

---

## Running it

```bash
set -a; . ./config/env; set +a
uv run python tools/smoke.py                    # the floor

# link 1–2 (verified, reproducible):
uv run python - <<'PY'
import sys; sys.path.insert(0, "tools")
from cadexd_client import CadexdClient, Engine, accepted_artifacts
proj = "projects/000-loop-validation"
with CadexdClient(Engine.resolve()) as c:
    c.open_project(proj); c.require_dynamics()
    r = c.write_script(open("experiments/000-loop-validation/rig.py").read())
    print("digest", r["digest"])
    for a in accepted_artifacts(proj):
        print(f"{a.kind:32} {a.output:12} {a.path}")
PY
```

Links 3–10 need `harness/` and are the follow-up plan's work.
