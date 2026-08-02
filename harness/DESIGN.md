# harness/DESIGN.md — the drivers, specified

**Nothing in this directory is built yet.** This file is the specification
the follow-up plan starts from, so that plan begins with interfaces rather
than a blank page.

## Why these exist here

`MUJOCO.md` §7 — the canonical 13-step method — leans on five scripts:
`rebuild.py`, `measure.py`, `feasibility.py`, `capability.py` and
`compare.py`. **ADR-088 §6 keeps them out of the cadex repository on
purpose**: they are ~100-line per-project drivers that speak `cadexd` over
NDJSON, and what is reproducible is the *method*, not a model file.

They do not exist on this box at all. cdx-rl is their natural home,
**generalized**: parameterized by project and task rather than copied and
edited per project. Plus a sixth thing §7 assumes a human provides by
watching — a training supervisor.

## Shared substrate

Everything sits on `tools/cadexd_client.py`:

```python
from cadexd_client import (
    Engine, CadexdClient, CadexdError, ScriptRefused,
    accepted_artifacts, accepted_outputs_dir, artifacts_of_kind,
)
```

### Conventions every driver follows

1. **`--json` emits an envelope; prose otherwise.** Every driver has both.
   The envelope is what a graph node's artifact is made of.
2. **Exit codes mirror Cadex's.** `0` fine, `1` infrastructure, `2` usage,
   `3` the engine refused the script. `ScriptRefused` → 3, `CadexdError` → 1.
   A driver that collapses these retries a script that will never build.
3. **`require_dynamics()` immediately after `open_project`.** One round trip
   that turns a stale engine into a loud failure instead of a confusing one.
4. **Every envelope carries provenance**: engine identity
   (`Engine.describe()`), the Cadex checkout commit, the project digest, the
   bundle sha256 where one is involved, and a UTC timestamp. A number without
   these is a number about nothing in particular.
5. **A computed diagnostic is always printed.** No `--verbose` gate on
   anything that could change a conclusion. ADR-106's termination mix cost
   three runs by being collected and not shown.
6. **Every check must be able to fail**, and must have been made to fail once
   on purpose. Hazard 18: a green light from a check that computes nothing is
   worse than no check.
7. **Reload the model per episode.** ADR-103 §9 — `evaluate_episode`
   multiplies domain randomisation **in place** and keeps no baseline, so an
   evaluator looping over one loaded model compounds every draw it has made.
   Four milliseconds against a two-second episode. **Every evaluator ships
   with a test that plays the same file twice and asserts the rows match.**
8. **MuJoCo comes from the trainer venv, by subprocess.** cdx-rl's own
   `.venv` deliberately has no `mujoco` pin — a second one would be a second
   answer to "which simulator produced this number". Evaluators shell out to
   `$CADEX_TRAIN_VENV/bin/python`.

---

## 1. `rebuild` — get a project to a known state

The thinnest driver, and the one the others call.

```
rebuild --project DIR [--script FILE] [--set k=v ...] [--json]
```

* Opens (creating if absent), optionally writes a script or sets parameters,
  rebuilds, and resolves the accepted attempt.
* Emits: `digest`, `accepted_revision`, `attempt_dir`, and every artifact as
  `{output, kind, path, sha256, bytes}`.
* **Asserts digest stability** when `--verify` is passed: rebuild twice,
  compare. That assertion is the floor everything else stands on.
* On refusal, emits the failure envelope verbatim — `failure_code`,
  `failure_stage`, `observed`, `retry`, `candidates` — because that is what a
  next attempt is written from.

## 2. `measure` — the numbers everything else is sized from

```
measure --project DIR [--model OUTPUT] [--json]
```

Reads `model_evidence.inertials` from the worker report, so **mass and
inertia come from OCCT** — not from a tessellation and not from a guess.

Reports:

* total mass, and per-component mass;
* the standing centre of mass, **measured at the exported keyframe**, not
  read off the drawing;
* each joint's height, and the mass hung below it;
* the support polygon, forward / back / lateral, from the collision geometry
  — **asymmetry is normal and is a finding** (`mg-legs`: 45.5 mm forward,
  24.5 back, because the toe reaches and the heel does not);
* leg length and available swing, for the one-step reach;
* whether the mechanism has any **lateral authority** at all — without ankle
  roll or hip yaw, sideways is hip roll plus a weight shift and the effective
  polygon is well under the geometric half-width.

And it computes the **capture point** table (ADR-100), because that is what
the numbers are *for*:

```
ω₀ = √(g/h)     Δv = F·t/m     ξ = Δv/ω₀
```

printed as ξ per candidate shove magnitude, annotated *in place* / *one step*
/ *beyond reach*. A `--sustained F` argument subtracts the CoP offset
`F·h/W`, because a sustained force shrinks the polygon rather than adding to
the shove.

## 3. `feasibility` — the gate

```
feasibility --project DIR --task OUTPUT [--json]
```

Six checks, none of which learn anything:

| # | Check | Must |
|---|---|---|
| 1 | static arithmetic | **advisory** since ADR-099 — report, never gate |
| 2 | gravity compensation by `mj_inverse` | be within limits |
| 3 | reject the **worst declared shove** in place | be within limits |
| 4 | contact sanity | no interpenetration at the reset pose, at the widest declared tilt and smallest declared lift, **at sixteen azimuths** |
| 5 | drop test at zero torque | **fall** — and report *when* |
| 6 | hand-written PD | **hold** |

If a PD can stand it, PPO can.

**Output is a table with a reason per row, not a boolean.** Red is a
diagnosis, not a verdict: hazard 14 is the gate being wrong, hazard 18 is the
gate measuring nothing, hazard 10 is the gate being right and the machine
being unable to do the task. `--explain` prints which of the three each red
row looks like.

> **Check 3 has a known way of being fake.** `mj_inverse` with an external
> force applied returns leg torques **bit-identical** to the undisturbed
> case, because inverse dynamics on a floating base solves for the force
> needed at every dof including the six unactuated ones — the push is
> absorbed by the free joint's own residual. The tell is all joints reporting
> the same worst-case. **The implementation must assert the disturbed and
> undisturbed results differ**, and fail the check itself if they do not.

Check 4 must not be arithmetic anyone did by hand: `mg-legs` was written with
a 3 mm lift for a 6° tilt on plausible reasoning, and the measured answer was
5.13 mm.

## 4. `capability` — what was the task even asking

```
capability --policy FILE --task BUNDLE [--scales …] [--seeds N] [--json]
```

The driver ADR-106 is named after, and **the one that must run before anyone
writes "the policy failed to learn X"**.

Sweeps a scale factor over the task's declared shove magnitudes and prints,
per scale:

* survival — episodes surviving over episodes shoved — **split by azimuth**;
* mean and median steps of the episode budget;
* the **termination mix** (`collapsed` / `tipped` / `timeout` / …), always,
  never behind a flag;
* **how far into its own disturbance schedule each death got** — this is how
  ADR-106 discovered that the second shove window had never once been
  exercised, so half the episode's design had never run.

Default scales `[0.0, 0.15, 0.30, 0.50, 0.75, 1.00]`, plus a
reset-variation-only row and a no-variation row, because those three
establish whether the machine can stand at all before anything is asked of
it.

**It must say when it measured nothing.** A curve flat across the whole sweep
— 0/12 everywhere, or 12/12 everywhere — is not a result. A 50-iteration
sanity run reads 0/12 at every scale, and the file has to say so out loud
rather than let it read as a finding.

## 5. `compare` — choose a checkpoint by what it did

```
compare --dir RUNDIR --task BUNDLE [--seeds N] [--json]
```

Plays **every** `.cxpolicy` in a directory locally against several seeds —
stock MuJoCo, no GPU, seconds — and prints one table.

Columns, per checkpoint:

| | |
|---|---|
| survival | episodes held / episodes run |
| episode length | mean steps, and the budget |
| final tilt | degrees, against the termination threshold |
| drift | mm, evaluated through **the task's own `drift` expression** via the engine's evaluator, not a re-derived one |
| **peak torque per motor** | N·mm |
| **mean torque per motor** | N·mm |
| **% of frames above 90 % of limit, per motor** | the hazard-15 column |
| termination mix | how it died |

The torque columns are not optional and not a flag. Hazard 15 is a policy
that plays as a clean stand while holding three of eight motors above 95 % of
an MG90S's *stall* rating on 100 % of frames — and 216 N·mm is a momentary
number that no real servo holds for six seconds. Nothing in the trajectory
shows this; these columns are what catch it without a rebuild.

Three implementation requirements, each learned by being caught:

* **`--task PATH` is mandatory, and it means the run's own bundle.** A
  rebuild is keyed by script digest and replaces `script_artifacts/`, so a
  finished run's task can vanish locally while its checkpoints sit beside
  you. Never fall back to "the project's current task".
* **Reload the model per episode** (ADR-103 §9), with the same-file-twice
  test in CI.
* **Flag `<out>.best.cxpolicy` when it looks untrained.** Best-by-reward
  early in a run can be the initial network, which scores well by standing
  still before the disturbance distribution has bitten. A policy commanding
  1–2 N·mm of 86 is not balancing, it is doing nothing — and the table should
  say so on the row rather than leave it to be noticed.

Watching two policies animate at once is not available and must not be faked:
ADR-077 is exactly one simulation per script and the shell has one timeline.
Numbers compare side by side; animations do not.

---

## 6. `supervise` — the training supervisor

The piece `MUJOCO.md` §7 assumes a human provides, and the clearest first
contribution this repository can make.

```
supervise --run DIR [--patience N] [--min-iterations M]
          [--stop-on-witness-margin X] [--json]
```

Wraps a dispatch, or attaches to a running one. It polls `progress.json` —
which the trainer rewrites every iteration and which nothing on this box
reads — and acts.

### What it watches

| Signal | In `progress.json` | Act on |
|---|---|---|
| reward peak | `best_iteration`, `best_reward_per_step` | iterations since best > `--patience` |
| **episode length** | `episode_steps` | **falling while reward rises → hazard 19** |
| divergence | `loss`, `action_std` | NaN, or σ collapse |
| device | `device` | not `"gpu"` on a GPU dispatch → abort immediately |
| liveness | `wall_time_s`, `iteration` | no progress for k× the observed iteration time |
| terminal | `state`, `error` | `done` / an error string |

Witness margin is on the trainer's stderr rather than in `progress.json`, so
`train.log` is tailed for it: **under 100× means stop and read hazard 13**.

### The bill it exists to prevent

`/home/theo/cadex-jobs/stand-task-20260802-200109` — best at iteration **598**
(0.337 reward/step), ran to **2499** (0.146), 14 050 s ≈ 3.9 GPU-hours.
**~76 % of the run was spent regressing**, and `progress.json` said so every
iteration.

### The line it must not cross

**Stopping the burn is not choosing the checkpoint.**

ADR-099 measured the trainer's reward and real survival as *anti-correlated
across a whole run*: 12/12 survival where the trainer reported its worst
numbers, 0/12 exactly where it reported its best. The checkpoint labelled
`best` fell in 43 steps of 600, from every seed and every direction, before
the first shove window opened.

So `supervise` **stops** a run and **never installs** one. Selection is
`compare`'s job, by measured survival. The supervisor's report ends with a
recommendation to *run compare over the retained checkpoints*, and nothing
stronger.

### Mechanics

* **`SIGTERM`, and a grace period.** There is no trainer-side early stop
  (wishlist #3), so stopping means signalling. Time it away from the
  checkpoint writer — read the `checkpoints` list and avoid the moment after
  a multiple of `--checkpoint-every`. Losing a half-written `.cxpolicy` is
  cheap; losing the run's last complete one is not.
* **`--min-iterations`** guards against stopping during the initial plateau,
  where an untrained network can look stable.
* **A terminal report**, always, whether it stopped the run or watched it
  finish: the curve, peak vs final, episode length beside reward, wall time,
  the checkpoint inventory with sha256s, and the witness margins seen. That
  report is the artifact that attaches to the graph node.
* **Attachable after the fact.** Pointed at a finished run directory it
  produces the same report from `progress.json` and `train.log`, which is how
  experiment 001 reads the existing runs.

---

## Ordering for the follow-up plan

1. `rebuild` — everything else needs it, and it is nearly done inside
   `cadexd_client`.
2. `supervise` in report-only mode against the finished runs in
   `/home/theo/cadex-jobs`. No dispatch, no risk, and it produces experiment
   001's first evidence.
3. `compare` — the thing that makes any checkpoint claim meaningful.
4. `measure` — needed before any new task is sized.
5. `feasibility`, then `capability`.
6. `supervise` with an active dispatch.

Experiment 000 is the integration test for 1–3.
