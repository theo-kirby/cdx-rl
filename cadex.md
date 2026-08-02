# cadex.md — Cadex, for an agent with no prior context

Everything below was executed on `sb1x` on 2026-08-02 against Cadex at
`06d1374b`. Where a command is quoted, its real output is quoted with it.
Where a claim contradicts a reasonable assumption, it says so.

**cdx-rl is read-only toward `/home/theo/cadex`.** No commits, no edits, no
branch changes, no builds. Things we wish Cadex did go in
[`cadex-wishlist.md`](cadex-wishlist.md).

---

## 1. What Cadex is

A parametric CAD system whose source of truth is **one Python script per
project**. The script runs, publishes named outputs, and the engine records a
content **digest** of the model it produced. Same script, same parameters,
same digest, on any machine.

An AI turn writes the *script*; everything after that is cheap and
deterministic. That cost asymmetry is the whole design: an expensive model
call authors a parametric mechanism once, and a loop sweeps its parameters
with no model in the loop at all.

For cdx-rl the important part is what the script can *also* declare: a
dynamics layer. Joints get inertia and damping, components get collision
shapes and actuators, and the same script that draws the machine exports a
**MuJoCo MJCF model** and a **training task bundle**. The mechanism and the
learning problem come out of one source.

## 2. The three-way split

| Piece | Where | What it is | Interpreter |
|---|---|---|---|
| **Engine** | `/home/theo/cadex/src/Mod/cadex` + a built `FreeCADCmd` | `cadexd`, an NDJSON service over stdio. Does the modelling. | pixi env, Python 3.11 |
| **Shell** | `/home/theo/cadex/shell` | The Blender-based GUI. macOS. Sliders, timeline, Policy Outputs panel. | — |
| **Trainer** | `/home/theo/cadex/training/cadex_train.py` | PPO on MJX. Reads a task bundle, writes a `.cxpolicy`. | `/home/theo/cadex-train-venv`, Python 3.12.3 |

They share files, not processes. The engine never imports JAX; the trainer
never imports FreeCAD. `cdx-rl` drives the first and the third, on this box,
and sends hero results to the second for visual verification.

### Two environments, and why they must stay two

* The **engine** environment is pixi (`pixi.toml`), Python 3.11. It builds
  and ships without a line of JAX.
* The **trainer** environment is a plain venv at `/home/theo/cadex-train-venv`,
  Python 3.12.3, with `mujoco==3.10.0`, `mujoco-mjx==3.10.0`, `jax==0.7.2`
  (cuda12 wheel), `numpy==2.5.1`.

`training/requirements.txt` says why, and it is not a style preference:

> MuJoCo's own `VERSIONING.md` disclaims cross-version numerical
> reproducibility, and a training run is only reproducible if the thing it
> ran against is. **NONE of this enters `pixi.toml`.**
> `test_engine_purity_guardrails` asserts that no `jax` and no `mjx` reach
> the staged payload.

The task bundle records `mujoco_version` (we measured `3.10.0`), and a
mismatch is a run whose numbers cannot be compared with the engine's. **Never
rebuild the trainer venv.** cdx-rl references it by path
(`CADEX_TRAIN_VENV`) and never recreates it.

## 3. The CLI

The entry point is a **bash shim** at `/home/theo/cadex/cadex`. It picks an
interpreter — the pixi env's Python if present, else any `python3` — sets
`PYTHONPATH=<repo>/cli`, and `exec`s `python -m cadex_cli`.

> **Trap: Cadex is not pip-installable.** There is no `pip install cadex`,
> and `python -m cadex` does not work. The module is `cadex_cli`, and it is
> only importable with `cli/` on the path. Run `/home/theo/cadex/cadex`.

### The whole surface

```
$ ./cadex --help
usage: cadex [-h] [--project PROJECT] [--out OUT] [--format FORMAT]
             [--engine ENGINE] [--json] [--wait] [-p PROMPT] [--resume]
             [--model MODEL] [--claude CLAUDE]
             {params,export,script} ...
```

Four things it can do. `-p/--prompt` is the default action (an AI turn), plus
three subcommands:

| Command | Does | Tokens |
|---|---|---|
| `cadex -p "<prompt>"` | One AI turn: the model writes or edits the script | **yes** |
| `cadex params --set k=v` | Set declared parameters and rebuild | no |
| `cadex script` / `script --set FILE` | Print or replace the script | no |
| `cadex export` | Rebuild the accepted script and write STEP/STL/BREP | no |

> **Trap: there is no `train` subcommand, and there is no train button.**
> This is a product decision (ADR-084), not a gap. Training is reached only
> by running `training/cadex_train.py` against a task bundle. Do not go
> looking for a CLI verb that will never exist; do not add one to Cadex.

### Exit codes — the branch every pipeline needs

```
0  fine
1  the engine or the agent failed
2  the command was wrong
3  the engine refused the script
```

> **Trap: 3 is not 1.** A refused script is a *modelling* problem to feed
> back into the next attempt. A failed engine is an *infrastructure* problem
> to retry or abort on. A driver that collapses them retries a script that
> will never build. Measured:

```
$ printf "result = {'plate': part.box(0.0, 0.0, 0.0)}\n" > broken.py
$ ./cadex script --project ./demo --set broken.py --json ; echo "exit=$?"
{ ...
  "error": "api.box: invalid length: must be greater than 0. Received 0.0.",
  "ok": false,
  ... }
exit=3
```

`tools/cadexd_client.py` reproduces this split as `ScriptRefused` versus
`CadexdError`.

### Streams

Progress → **stderr**. Report → **stdout**. `--json` is always safe to pipe.
`cadex script` with no `--set` prints the script and nothing else, so
`cadex script > model.py` works.

### The `--json` envelope, and what it will not tell you

Real output from `cadex export --project ./demo --out ./out --json`:

```json
{
  "schema": "cadex-cli-v1", "ok": true,
  "project_root": "/home/theo/cdx-rl/projects/cli-demo",
  "revision": "cb56fea2…", "accepted_revision": "cb56fea2…",
  "digest": "65112ef8…",
  "params": {"thickness": 6.0, "width": 42.0},
  "outputs": [
    {"name": "plate", "kind": "brep",
     "files": {"step": "…/out/plate.step", "stl": "…/out/plate.stl"}}
  ],
  "out_dir": "…/out",
  "engine": {"source": "payload", "freecadcmd": "…", "module_dir": "…"}
}
```

Two things to read carefully:

* **`outputs` is empty without `--out`.** `cadex script --set` and
  `cadex params --set` both returned `"outputs": []` on a project that has
  one. They rebuilt it; they just wrote nothing. `files` appears only for
  outputs `export` actually wrote.
* **`--json` gives no path to a non-BREP artifact, ever.** `cadex export`
  writes `step`, `stl` and `brep` — BREP-domain formats. The MJCF and the
  task bundle have no export path on the CLI at all. Locating them is
  §5, and it is something cdx-rl had to implement.

**Compare `digest`, never the files.** STEP writes a wall-clock timestamp
into `FILE_NAME`, so two exports of an identical model differ byte for byte
across a second boundary.

## 4. Which engine — the trap that costs a day

`cadexd` is resolved from a **staged engine payload** (a directory holding
`cadex-engine.json`) or from a **built checkout** (`src/Mod/cadex` plus a
`FreeCADCmd`). The CLI prefers `--engine`, then `CADEX_ENGINE_ROOT`, then the
development tree.

> **On sb1x the staged payload is stale and cdx-rl must not use it.**

`build/engine/cadex-engine-0.0.0-linux-x64` was assembled 2026-07-31. Its
`Mod/cadex` has no `CadexDynamics.py`, `CadexNets.py`, `CadexSolder.py` or
`CadexTerminals.py`. Asked for its assembly API it answers:

```
assembly component connector joint solve motion simulation exploded_view
```

The checkout's `src/Mod/cadex` answers with those **plus**:

```
dynamics mjcf task policy rollout body collision joint_dynamics actuator
observation reward termination randomise reset_variation disturbance
```

Every one of those is load-bearing here. Point a driver at the payload and it
does not get a clear error — it gets *"assembly.mjcf is not defined"* from
inside a script, which reads like a script bug.

**So: set `CADEX_ENGINE_DEV_TREE=/home/theo/cadex` and leave
`CADEX_ENGINE_ROOT` unset.** The same applies to the CLI: `./cadex` with
`CADEX_ENGINE_ROOT` exported silently drives the stale payload; with it unset
it falls back to the development tree and works.

`CadexdClient.require_dynamics()` asserts the surface in one round trip.
Call it right after `open_project`, in every driver.

Re-staging the payload would fix this properly and is **not cdx-rl's to do** —
it writes into the Cadex checkout. It is on the wishlist.

## 5. The project store, and finding an artifact

A project root looks like this:

```
<project>/
  script.py                      THE script — the sole source of truth
  script.json                    params, revisions, accepted digest, accepted_attempt
  script_artifacts/<revision>/attempt-<stamp>-<uuid>/
      result.json                the worker report — per-output artifact_path
      request.json               the source this attempt ran
      outputs/                   the files
  script_history/                the last 25 accepted sources (text only)
  assets/                        put_asset lands here (.stl/.obj/.ply/.cxpolicy)
```

What lands in `outputs/`, by declaration:

| Script call | File | `artifact_kind` |
|---|---|---|
| `part.*`, `partdesign.*` | `output-NNN.brep` | `brep` |
| `mesh.*` | a triangle mesh | `mesh` |
| `assembly.mjcf(...)` | `<output>-model.xml` | `assembly_mjcf_xml` |
| `assembly.task(...)` | `<output>-task.json` | `assembly_training_task_json` |
| `assembly.policy(...)` | `<output>-policy.json` | `assembly_policy_receipt_json` |
| `assembly.simulation` / `dynamics` / `rollout` | `assembly-simulation-trace.json` | `assembly_simulation_json` |

**`artifact_kind` is an open set.** Select on the kinds you know and *ignore*
one you have never heard of. Do not fail on it.

### The resolver, and the trap inside it

`script.json` carries `accepted_attempt.staging`, a project-root-relative
path pinned against the pruner (which otherwise keeps only the last three
attempts). `result.json` in that directory lists every output with an
`artifact_path` relative to the same directory. That is the whole lookup, and
`tools/cadexd_client.py` implements it:

```python
from cadexd_client import accepted_artifacts, accepted_outputs_dir
```

> **Trap: do not build the path from `accepted_revision`.** The staging
> directory is named with a *pre-run* revision computed over the stored spec
> cache; `validate_project_result` then recomputes the revision from the
> specs the worker actually collected and records **that** as the durable
> one. Measured on this box, for a two-parameter box script: accepted
> revision `104826f0…`, artifacts under `script_artifacts/93a118d4…/`. They
> disagree for any script that declares a parameter — which is every script
> worth writing.

### `inspect` is a bounded reader

`inspect scope="output"` answers from the pinned attempt's `result.json`.
Replies cap at 32 KiB, containers page (`offset`/`limit`, `limit <= 50`), and
any single value over 1 KiB is replaced by a marker naming the JSON Pointer
that reaches it. `value` is a *view*, never a promise of the whole. Read to
the end of the pages, or accept a sample. Real reply:

```json
{"output_count": 1,
 "outputs": [{"name": "cube", "type": "solid", "domain": "part",
              "artifact_kind": "brep"}],
 "revision": "104826f0…"}
```

## 6. The `cadexd` protocol

`cadex-cadexd-v1`. Newline-delimited JSON over stdio, 8 MB frame cap, one
`FreeCADCmd` child per open project. Request `{schema, id, op, args}`,
response `{id, ok, ...payload}`, progress `{id, event}` interleaved. A `ready`
banner event on startup. Binary is referenced by path, never inlined.

Ops cdx-rl uses: `open_project`, `describe_api`, `write_script`, `set_params`,
`rebuild`, `inspect`, `put_asset`, `cancel`, `shutdown`.

Things worth knowing before writing to it:

* `write_script` and `set_params` require **`expected_revision`**. There is
  one writer per project, so the client tracks it (`CadexdClient.revision`).
  `STALE_PROGRAM_REVISION` is the guard when it is wrong.
* `write_script` replaces **the whole** script. Dropping an output the
  accepted revision declares needs `replace=True` (ADR-045), because "add a
  part" is an easy way to ask for that by accident.
* `put_asset` takes a **path, not bytes** — the asset budget is 128 MB
  against an 8 MB frame cap. It accepts `.stl`/`.obj`/`.ply` **and
  `.cxpolicy`**: that is how a trained policy comes home.
* Server failure codes are `CADEXD_PROTOCOL_ERROR`, `CADEXD_BUSY`,
  `CADEXD_NOT_OPEN`, `CADEXD_CRASHED`, `CADEXD_RESTORE_FAILED`. Anything
  else in a failure envelope is a *tool* failure — the exit-3 case.
* FreeCADCmd prints on stdout before `cadexd` hijacks the fds. A non-JSON
  line before the banner is expected; skip it.

## 7. The authoring surface, in pipeline order

`describe_api` is the ground truth and it moves; this is the shape, not a
substitute for asking. Five domains are staged as globals: `sketcher`,
`part`, `partdesign`, `mesh`, `assembly`, plus `params`/`num`.

```
                geometry          part.box(...) / partdesign / sketcher
                    │
   assembly.component(solid, grounded=…, placement=…)
   assembly.connector(component, offset={position, axis, angle_degrees})
   assembly.joint(kind, first, second, angle_limits_degrees=…)
                    │
   assembly.assembly(components, joints) → assembly.solve(asm)
                    │
   assembly.body(component, density_kg_m3=…, collision=assembly.collision(…))
   assembly.joint_dynamics(joint, damping_nmms_per_deg=…, armature_kgmm2=…)
   assembly.actuator(joint, kind="motor", control_nmm=…, torque_limit_nmm=…)
   assembly.observation(target, kind, name=…)
                    │
   assembly.mjcf(asm, bodies, actuators=…, joint_dynamics=…, observations=…)
                    │            → outputs/<name>-model.xml
   assembly.reward(expression, weight=…)
   assembly.termination(expression, above=…, below=…)
   assembly.randomise(target, property_name, scale=[lo, hi])
   assembly.reset_variation(target, tilt_degrees=…, height_mm=…, …)
   assembly.disturbance(target, newtons=[lo, hi], azimuth_degrees=…, at_seconds=…)
                    │
   assembly.task(model, actions=…, reward=…, episode_seconds=…, control_hz=…,
                 termination=…, randomisation=…, reset_variation=…,
                 disturbance=…)      → outputs/<name>-task.json
                    │
                 [ the trainer ]     → <label>.cxpolicy
                    │
   assembly.policy(task, file=…, sha256=…)   → outputs/<name>-policy.json
   assembly.rollout(...)                     → a simulation trace
```

Every value must also be returned in the script's `result` dict exactly once.

Observation kinds and the channel names they create:

| `kind` | channels |
|---|---|
| `position`, `velocity`, `actuator_force` | `<name>` |
| `component_position`, `component_linear_velocity`, `component_angular_velocity`, `centre_of_mass` | `<name>_x/_y/_z` |
| `component_orientation` | `<name>_qw/_qx/_qy/_qz` |

Those names are what a `reward` or `termination` expression writes.

**Units are suffixed and the suffix is checked.** `control_deg` vs
`control_mm`, `stiffness_nmm_per_deg` vs `stiffness_n_per_mm`,
`armature_kgmm2` vs `armature_kg`. Passing the wrong one is a refusal, not a
factor of 5.5 million (MUJOCO.md hazard 1). Observations carry a per-channel
`scale` so the trainer *multiplies* rather than converts — the one shape of
the operation that cannot be performed backwards. `angle_degrees` is the
dangerous one: every other conversion on that boundary is a power of ten and
*looks* wrong when reversed; 57.29578 looks like a mechanism.

**Action ranges come from the mechanism, not the trainer.** A `motor` is
bounded by `torque_limit_nmm`; a `position` servo by its joint's own limits,
*both* endpoints declared; a `velocity` actuator has no derivable range and
`assembly.task` refuses it.

### Proven headless, on this box

The claim that matters most to cdx-rl: **the dynamics domain evaluates with
no display and no shell.** A 1-DOF pendulum, authored through `cadexd` on
sb1x, rebuilt, and its artifacts resolved:

```
digest b77a1f5a39523b8d839c790ddb4a67367c04b084f496a48332acdb4db3ff7575
brep                          post_solid   output-000.brep       2594 bytes
brep                          arm_solid    output-001.brep       2594 bytes
assembly_mjcf_xml             model        model-model.xml       1259 bytes
assembly_training_task_json   task         task-task.json        2648 bytes
```

The MJCF is self-contained, in **radians and SI**, with masses and inertias
computed from the solids (`mass="3.768" diaginertia="0.0287624 …"`), a
`<keyframe name="solved">`, `<sensor>` elements for the declared
observations, and `forcerange="-2 2"` derived from `torque_limit_nmm=2000`.
The bundle is `cadex-training-task-v1` and records
`"mujoco_version": "3.10.0"` — the trainer venv's pin.

> **The file opens where the assembly was solved.** MuJoCo's reference
> configuration is where each joint's connector frames coincide, which is not
> the solved pose. Reset to the `solved` keyframe. And the export carries
> **collision geometry only**: a mechanism with no `assembly.collision`
> shapes opens *invisible* in MuJoCo's viewer.

The "export is BREP-only" limitation is real but narrow: it constrains
`cadex export`'s STEP/STL writing, and does not block MJCF or task
production. That is what makes a self-contained loop on this box possible.

## 8. The trainer

```
$ /home/theo/cadex-train-venv/bin/python \
    /home/theo/cadex/training/cadex_train.py --help
usage: cadex_train.py [-h] --out OUT [--seed SEED] [--iterations ITERATIONS]
                      [--envs ENVS] [--unroll UNROLL] [--epochs EPOCHS]
                      [--hidden HIDDEN [HIDDEN ...]] [--learning-rate LR]
                      [--discount DISCOUNT] [--gae-lambda GAE_LAMBDA]
                      [--clip CLIP] [--entropy ENTROPY]
                      [--value-weight VALUE_WEIGHT] [--initial-std STD]
                      [--label LABEL] [--quiet] [--checkpoint-every N]
                      [--progress PATH]
                      bundle
```

`bundle` is `<attempt>/outputs/<name>-task.json`. `--out` is where the
`.cxpolicy` goes. `--checkpoint-every N` writes a complete `.cxpolicy` every
N iterations *plus* `<out>.best` — it costs about one iteration each.
`--progress PATH` names the progress file; it defaults to `progress.json`
beside `--out`, and it is what `remote_train.sh watch` and the shell's
Training panel read.

**There is no early stop and no `--max-iterations-without-improvement`.**
That absence is why the training supervisor is cdx-rl's first real
contribution, and it is on the wishlist.

The trainer proves its own **witness** before writing each file and prints
the margin. *If that margin is under 100×, stop and read MUJOCO.md hazard 13
rather than continuing.* It also records `jax.default_backend()` into the
policy, so a run that silently fell back to CPU is visible in the artifact
rather than only in how long it took.

`progress.json` (schema `cadex-training-progress-v1`) carries, every
iteration: `iteration`, `total`, `reward_per_step`, **`best_iteration`**,
**`best_reward_per_step`**, `episode_steps`, `loss`, `action_std`, `device`,
`eta_s`, `wall_time_s`, `state`, `error`, and a `checkpoints` list with per
entry `{iteration, path, reward_per_step, sha256, bytes, tag}`.

Nothing on this box acts on `best_iteration`. See
[`method.md`](method.md) §"Peak versus final".

## 9. Where to read further, in `/home/theo/cadex/docs/`

| File | What it is | Read when |
|---|---|---|
| **`MUJOCO.md`** (162 KB) | The MuJoCo work end to end. **§7 is the canonical 13-step method**; §5 is 19 ranked hazards; §6 the open questions. | Before designing any experiment. §7 and hazards 15/16/19 are mandatory. |
| **`DECISIONS.md`** (610 KB) | ADR-001…106, the rationale log. | When you want to know *why*. Grep for the ADR number a doc cites; do not read it front to back. |
| `INTEGRATION.md` | The `cadexd` protocol, op by op, with response shapes. | Writing a driver. |
| `CLI.md` | The CLI, verified against source 2026-08-01. | Scripting the CLI. |
| `ARCHITECTURE.md` | How the engine is put together. | Rarely. |
| `training/README.md`, `training/SETUP.md` | The trainer and its environment. | Setting up a training box. |

ADRs cdx-rl leans on directly:

| ADR | What it settles |
|---|---|
| **084** | Training happens elsewhere; a policy is a file we can check. Why there is no `train` verb. |
| **085** | The policy comes home — `put_asset`, `assembly.policy`, digest required. |
| **088 §6** | The per-project drivers stay **out** of the cadex repo. This is cdx-rl's charter. |
| **096** | The Policy Outputs panel — commands, not only trajectory. |
| **097** | Reset variation and disturbance; decide the metric before dispatch. |
| **099** | The feasibility gate re-specified; the reward curve is not the result. |
| **103** | The two simulators agree; the instrument did not. Reload the model per episode. |
| **104** | Refuse to dispatch to a box running a different trainer. |
| **106** | The task was out of range, and the band that replaces it. |

## 10. The trap list, in one place

1. **No `train` subcommand, no train button.** ADR-084. Not a gap.
2. **Not pip-installable.** Run the shim; the module is `cadex_cli`.
3. **Exit 3 ≠ exit 1.** Refused script vs failed engine.
4. **`--json` carries no artifact paths**, and `outputs` is empty without
   `--out`.
5. **The staged payload on sb1x has no dynamics domain.** Use the checkout.
6. **`accepted_revision` does not name the artifacts directory.** Read
   `accepted_attempt.staging`.
7. **`inspect` is bounded** — 32 KiB, paged, >1 KiB values replaced by
   markers.
8. **Compare digests, never files.** STEP embeds a timestamp.
9. **`MUJOCO_GL` is unset box-wide** and there is no precedent for setting
   it: the trainer is headless MJX and never opens a renderer. If video
   rollout is ever added, set `MUJOCO_GL=egl` at that call site.
10. **The MJCF opens at MuJoCo's reference pose, not the solved one.** Reset
    to the `solved` keyframe.
