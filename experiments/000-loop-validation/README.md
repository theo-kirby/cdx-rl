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

| # | Link | Passes when |
|---|---|---|
| 1 | script → engine | `rig.py` accepted; digest recorded | ✅ verified |
| 2 | engine → artifacts | `assembly_mjcf_xml` and `assembly_training_task_json` resolve to files that exist | ✅ verified |
| 2b | determinism | two rebuilds produce **identical digests** | ✅ verified for the smoke script; **to confirm for `rig.py`** |
| 3 | MJCF → MuJoCo | the XML loads in stock MuJoCo 3.10.0 and reports 1 dof, 1 actuator, 2 sensors | ☐ |
| 4 | bundle → trainer | `cadex_train.py` accepts the bundle and completes 200 iterations with no error | ☐ |
| 5 | trainer → policy | a `.cxpolicy` is written, and **every witness margin printed is > 100×** | ☐ |
| 6 | policy → verification | `assembly.policy(task, file=…, sha256=…)` is accepted on rebuild — the engine re-checks the bundle digest, the model, the observation channels in order, the action table, and re-evaluates the witness in float64 | ☐ |
| 7 | policy → playback | `assembly.rollout` produces a trace locally; frame count and hinge angle over the episode are reported | ☐ |
| 8 | supervisor | `supervise` produces a terminal report with peak vs final, the checkpoint inventory with sha256s, and the witness margins | ☐ |
| 9 | evaluator sanity | **the same checkpoint played twice produces the same row** (ADR-103 §9) | ☐ |
| 10 | record | a Flywheel node under the cdx-rl root, carrying `progress.json`, the chosen `.cxpolicy`, the MJCF and the bundle as artifacts | ☐ |

**Explicitly not a pass criterion: the reward going up.** If the reward
climbs, fine. If it does not, links 1–10 can still all pass and the
experiment still succeeds. A 1-DOF pendulum with a hang-down reward is not a
learning problem worth grading.

Criterion 9 is the one most likely to fail quietly, and it is the two-second
test that found the ADR-103 bug. Do not skip it because the model has no
domain randomisation — the point is that the evaluator reloads.

## 8. What happened

*(Links 1, 2 and the MJCF/bundle contents verified 2026-08-02 — see §3. The
rest not yet run.)*

## 9. What it means

*(Pending.)*

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
