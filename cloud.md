# cloud.md — compute topology, and when to leave this box

**Scope of this file:** where cdx-rl's compute physically happens, how work
gets onto it, how to budget a 32 GB card, and the one case where renting
somebody else's GPU is the right call.

It is *not* about deploying Cadex, and not about Flywheel's hosted side
beyond the compute it can rent on our behalf. If that is what was wanted, say
so and this gets re-aimed.

---

## 1. The topology

```
   MacBook (macOS)                          sb1x  (this box)
   ┌──────────────────┐                     ┌──────────────────────────────┐
   │ Cadex shell      │                     │ engine  (FreeCADCmd, cadexd) │
   │  Blender GUI     │   ssh + rsync       │ trainer (MJX on RTX 5090)    │
   │  sliders         │ ──────────────────► │ cdx-rl  (drivers, graph glue)│
   │  timeline        │   remote_train.sh   │ /home/theo/cadex-jobs/       │
   │  Policy Outputs  │ ◄────────────────── │ /home/theo/cdx-rl/jobs/      │
   └──────────────────┘   checkpoints back  └──────────────────────────────┘
                                                        │
                                                        │ (rare)
                                                        ▼
                                            Flywheel managed compute
                                            nebius / primeintellect / vastai
```

**sb1x is both ends at once**, and that is the change this repository is
built around. It has always been the remote training target driven from the
laptop. What exploration established is that it is also a complete authoring
host: the dynamics domain evaluates headlessly, so the mechanism, the MJCF,
the task bundle, the training and the local playback all happen here, with no
display in the loop (verified — see [`cadex.md`](cadex.md) §7).

The macOS shell keeps one job that nothing here replaces: **watching a policy
move, and the Policy Outputs panel.** That panel is the only thing that
catches hazard 15 — a policy that plays as a clean stand while holding three
motors at 98 % of stall — in one glance. It is a **promotion step for hero
results**, not a link in the chain.

### The box

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, **32 607 MiB**, driver 580.159.03 |
| CPU | Ryzen 9 9950X, 32 threads |
| RAM | 60 GB |
| Disk | 2.4 TB free |
| OS | Ubuntu 24.04, kernel 6.17 |
| JAX | 0.7.2 + cuda12, `default_backend() == "gpu"`, `cuda:0` |

### The second box: `sb9x`

Everything above described the only box there was. There are now two, and
they are not interchangeable — the differences are large enough to change
both what a run costs and whether it survives at all.

| | `sb1x` | `sb9x` |
|---|---|---|
| GPU | RTX 5090, 32 607 MiB | **RTX 4070, 12 282 MiB** |
| driver | 580.159.03 | **595.84** |
| CPU | 9950X, 32 threads | **16 threads** |
| RAM | 60 GB | **15 GB** |
| s/iteration @ 2048, steady | 4.29–5.62 | **8.93** |
| iteration 0 (XLA compile) | — | **65 s** |
| a checkpoint | ~1 iteration | **~106 s** |
| 1500 iterations, 15 checkpoints | 2.34 h | **~4.2 h** |

Measured on 2026-08-03 against `tasks/stand-b2/`, trainer `aacfa823…`, the
same fourteen hyperparameters, in **one 40-iteration run** rather than by
combining numbers across runs. **sb9x is ~1.6–2.1x slower per steady
iteration**, so every wall cap written for sb1x is wrong here — see the
projection guard in §4.

Two of those rows are worth more than the headline. **Iteration 0 costs 65 s**,
seven times a steady one, so any rate averaged from the start of a run reads
high — which is why the projection measures a slope between two later
samples. And **a checkpoint costs ~106 s here, not ~1 iteration** as §4 says
for sb1x: at `--checkpoint-every 100` that is 1 590 s of a 4.2 h seed, about
10 %, and it is the single largest correction to a naive `iterations x
s/iteration` estimate.

Both boxes carry the same `cadex-train-venv` pins, which `smoke.py` checks.
sb9x's Cadex checkout arrived **unbuilt** (no `.pixi`), so `engine_resolved`
failed and the eight `cadexd` checks never ran — training was unaffected
(`tools/train.py` never touches the engine) but nothing that authors a
mechanism worked. It was built on **2026-08-03** and **smoke now passes
13/13 there**, including `dynamics_surface` at 23 assembly exports and
`digest_stable` across two rebuilds. The recipe is below, and it is not the
obvious command.

The build leaves the checkout **clean** — 5.6 GB in `.pixi` and 225 MB in
`build/`, both gitignored, no tracked file touched, still at `ae8da6a6`.

#### Building the engine on a headless box — not `pixi run build-engine`

sb9x arrived with no `.pixi`, so `smoke.py` failed at `engine_resolved` and
the eight `cadexd` checks never ran. Getting from there to a built engine has
one trap in it, and the obvious command is the wrong one.

`build-engine` depends on `configure-release`, which depends on
`initialize` — and `initialize` is `git submodule update --init
**--recursive**`, which pulls `shell/lib/<platform>`: **1.3 GB over git-lfs,
for an application this box will never run**, and `pixi.toml`'s own comment
says it hard-fails when git-lfs is absent. It is absent here.

ADR-060's engine-only path avoids it, checking out just the two submodules
the engine actually compiles:

```bash
curl -fsSL https://pixi.sh/install.sh | sh        # ~/.pixi/bin, appends to .bashrc
cd /home/theo/cadex
~/.pixi/bin/pixi run setup-engine                 # OndselSolver + GSL only
~/.pixi/bin/pixi run -- env CFLAGS= CXXFLAGS= DEBUG_CFLAGS= DEBUG_CXXFLAGS= \
    cmake --preset conda-linux-release            # NOT `pixi run configure-release`
~/.pixi/bin/pixi run -- cmake --build build/release --parallel 6
~/.pixi/bin/pixi run -- cmake --install build/release
```

Two details are load-bearing. The empty `CFLAGS`/`CXXFLAGS` are what the
`configure-release` task sets, and they matter — conda exports aggressive
defaults that the preset expects to override. And `--parallel 6`, not the 16
threads available: FreeCAD translation units run over 1 GB each and this box
has **15 GB of RAM**, which a `-j16` build will exhaust. Peak observed at
`-j6` was ~5.7 GB used with ~9.7 GB free.

#### Three runtime settings, one real fix and two that only move the fault

At 2048 environments the trainer **segfaults on sb9x** with stock settings.
`tools/train.py` sets all three of these by default and records them in
`runtime.json` and `sweep.json`:

| setting | flag to undo | effect |
|---|---|---|
| stack 256 MiB | `--stack-mb` | **fixes** SIGSEGV during tracing, before iteration 0 — no traceback, `progress.json` still `state: starting` |
| `XLA_PYTHON_CLIENT_PREALLOCATE=false` | `--xla-preallocate` | moves the later fault; does not remove it |
| cyclic GC off | `--child-gc` | buys a correct exit code on runs that do finish; does not remove it |

**The stack one is a genuine fix and is understood.** MJX's `make_constraint`
fans out over contact pairs through nested `vmap` and overflows an 8 MiB
stack while tracing. `ulimit -s unlimited` does *not* help: glibc reads
`RLIMIT_STACK` when it creates a thread and substitutes its own 8 MiB default
for `RLIM_INFINITY`, and JAX traces on threads. The limit must be large and
**finite**, which is why this is a `preexec_fn` and not a shell line.

**The other two are not fixes, and it took a 40-iteration run to see it.**
The remaining fault is a `SIGSEGV` that lands *after a checkpoint write*, and
the settings only decide which one:

| configuration | died at | left behind |
|---|---|---|
| prealloc off, GC off | the **final** checkpoint, as `train()` returned | all checkpoints, no final policy |
| `XLA_PYTHON_CLIENT_ALLOCATOR=platform` | the **first** checkpoint, iteration 20 | one checkpoint, nothing else |

#### What this is *not*

Two hypotheses are ruled out by measurement, and both were stated as fact
here before they were checked — which is the mistake this file should not
repeat:

* **Not the card's size.** Peak device usage with preallocation off is
  **777 MiB of 12 282** — six per cent. The 9 131 MiB seen with preallocation
  *on* is JAX's 75 % pool, not demand. §4's "32 GB is not the binding
  constraint at these sizes" holds here too; 12 GB is not binding either, and
  the earlier claim that it was is withdrawn.
* **Not a GC leak.** RSS is flat to the byte between checkpoints (above).

What remains is a fault in the compile-and-free path around checkpointing,
on jax 0.7.2 with driver 595.84 on Ada — the two things that actually differ
from sb1x, neither of which is memory. The next diagnostic is a
`faulthandler` traceback taken at the checkpoint boundary rather than at exit.

**sb9x can train and cannot exit cleanly** — which is not the same as cannot
finish. What a crashed run leaves is measured below, and it is enough.

#### Necessary and **not yet sufficient**

Measured 2026-08-03, and the reason no seed has been dispatched on sb9x.

With all three settings on, 2048 environments completes cleanly at **1 and 3
iterations** — exit 0, witness 32 975x, final `.cxpolicy` written. At **40
iterations it segfaults**: `trainer exited -11`, `state: training`, every
checkpoint present, **no final `.cxpolicy`**. The earlier "clean" result was
a run too short to provoke it. **Validate at length; a 1-iteration run
reports success for every fault on this page.**

`faulthandler` puts that one at `cadex_train.py:1661` — the line
`trained = train(...)` — with no Python frames beneath it, so it is a C-level
fault as `train()` returns. It lands *before* the final policy is written,
which is why `os._exit()` in the launcher would not save it.

**The memory profile.** RSS across the failing run, sampled every 10 s
against the iteration counter:

```
iterations  1 → 18    5 690 MB, flat, not one byte of growth
iteration      19     6 442 MB   ← the checkpoint at 20
iterations 19 → 38    6 442 MB, flat
iteration      39     6 975 MB   ← the final checkpoint
```

So: **no per-iteration leak** — turning the collector off does not accumulate
anything across training, which was the open worry about it. Growth is
per-*compile*, ~500–750 MB a time, and the host still had 12.4 GB free when
the run died.

`XLA_PYTHON_CLIENT_ALLOCATOR=platform` has been tried, and it is **worse**:
routing frees through `cudaFree` instead of the BFC pool moved the crash
forward to the *first* checkpoint. It is not in the defaults.

Reducing `--envs` would shrink the working set, but `envs` is one of the
fourteen that define the run — that trades the crash for a different
experiment, and the point of these seeds is to replicate a specific one.

#### The crash is survivable, and that was measured too

"The run crashed" and "the run is lost" are different statements, and on a
40-iteration run that died this way the second is false:

* every `.cxpolicy` on disk parsed as a **complete** policy — header,
  weights, `network`, `normaliser`, and the model/task/trainer digests;
* `checked_policy` runs the witness **before** writing, so the crash can
  leave a file *missing* but never leave one *bad*;
* the `.best` header carried the whole `reward_curve`; and
* **`compare` consumed them end to end** — played both checkpoints over 6
  seeds and returned per-motor torque, the hazard-15 column and a selection
  verdict with its binomial bound.

The `.best` file was rewritten ~20 times in that run, each rewrite running a
fresh witness pass, without faulting — so **the checkpoint path is not what
breaks; `train()`'s return is.** What a crash costs is `<label>.cxpolicy`,
the final iteration's network, which ADR-099 says is not what you select
anyway.

`tools/train.py` therefore returns **`EXIT_SALVAGEABLE` (4)** for this state
and prints the checkpoint inventory and the `compare` line to run next. A run
with *no* checkpoints stays `EXIT_INFRASTRUCTURE` — the distinction is the
point, and a sweep's verdict ranks by `SEVERITY`, not by the raw number.

So sb9x **is** usable for a seed that must produce a witnessed policy, with
the caveat that the 1500-iteration case is inferred from a 40-iteration run:
check the checkpoint count on completion.

## 2. How the laptop drives it

`/home/theo/cadex/training/remote_train.sh`:

```
usage: remote_train.sh {check|train|watch|pull|stop|shell|config} [args...]
       train <bundle.json> <out.cxpolicy> [--allow-cpu] [--detach] [-- trainer args]
       watch <run-id> [destination]     poll progress, pull checkpoints
       pull  <run-id> [destination]     bring everything home once
       stop  <run-id>                   TERM a detached run
```

**ssh and rsync. No daemon, no service, no network I/O inside the product.**
Cadex itself never opens a socket; the transport is entirely in this script.
That is worth preserving — it is why a training run has no failure mode that
lives inside the engine.

Configuration is `training/.remote.env` (from `remote.env.example`), and two
of its variables matter here:

* `CADEX_TRAIN_REPO` — the checkout on the box. `remote_train.sh` runs
  **that** copy of `training/cadex_train.py`, so the trainer that runs is
  whatever revision the box is at. The policy records `trainer_sha256`, so
  which one it was is recoverable afterwards — but only afterwards.
* `CADEX_TRAIN_VENV` — **checked, never created.** `check` exits non-zero
  naming the path if it is absent rather than building one. *A venv this
  script silently created is a venv nobody knows the contents of, and the
  whole point of the exact pins is knowing.* cdx-rl inherits that stance
  exactly: [`config/env.example`](config/env.example) references the venv and
  never recreates it.

There is no password variable and there will not be one. `ssh` has no
non-interactive password path without `sshpass`, and a plaintext secret on
disk is a worse thing to own than a path to a key file.

> **ADR-104's guard is not optional.** Before dispatch:
> `ssh <box> "cd <repo> && git log --oneline -1"`. A trainer that predates a
> surface addition silently ignores the new fields while recording the new
> algorithm string in the policy header, and nothing fails loudly.
> `tools/smoke.py` records the commit on every run; on sb1x it is
> `06d1374b`.

## 3. Run directories

`/home/theo/cadex-jobs/<label>-<YYYYMMDD>-<HHMMSS>/` is the existing
convention on this box and cdx-rl copies it under
`CDXRL_JOBS=/home/theo/cdx-rl/jobs/` (gitignored). One directory per
dispatch, holding everything needed to interpret the run without the project
that produced it:

```
stand-task-20260802-200109/
  stand-task.json          the task bundle, as dispatched          30 KB
  model-model.xml          the MJCF it references                  13 KB
  progress.json            rewritten every iteration               15 KB
  train.log                the trainer's own stderr               210 KB
  train.pid
  stand8.000050.cxpolicy … stand8.002450.cxpolicy   (every 50)  88–444 KB
  stand8.best.cxpolicy     best-by-reward so far                  170 KB
  stand8.cxpolicy          the final network                      451 KB
```

Two properties of that layout are load-bearing:

* **The bundle travels with the run.** A rebuild is keyed by script digest
  and replaces `script_artifacts/`, so a finished run's task can vanish
  locally while its checkpoints sit beside you. The copy in the run directory
  is the one that survives, and it is what `compare --task` must be pointed
  at. Play a run against **its own** bundle, never the newest one.
* **Checkpoints are cheap and small.** 13 MB for 52 checkpoints of a
  four-hour run. There is no reason to economise on `--checkpoint-every`, and
  every reason not to: each one is a complete policy you can pull mid-run and
  play, and the ADR-099 table exists only because somebody had checkpoint 500
  to play.

`/home/theo/cadex-jobs` is **read-only input to cdx-rl.** Experiment 001
reads from it. Do not write there.

## 4. Budgeting the 5090

Two measured runs, both on this class of card:

| run | iters | envs | wall | s/iter |
|---|---|---|---|---|
| `mg-legs` (ADR-095) | 2000 | 4096 | 1 h 16 m | 2.28 |
| `stand-task-20260802-200109` | 2500 | — | 3 h 54 m | 5.62 |

So the planning number is **2–6 s per iteration at a few thousand
environments**, and a "normal" run is **1–4 GPU-hours**. Rules of thumb:

* **32 GB is not the binding constraint at these sizes.** MJX batches the
  whole population into one XLA program, so memory scales with
  `envs × unroll × model size`, and these mechanisms are 2–8 bodies. If a
  bigger mechanism ever does hit the ceiling, reduce `--unroll` before
  `--envs`: the batch statistics are what the learning depends on.
* **Environments are nearly free until they are not.** Doubling `--envs`
  costs well under double the wall time up to the point where the GPU
  saturates, and it is the cheapest variance reduction available. Find the
  knee once per mechanism and write it in the experiment README.
* **`--checkpoint-every 100` costs about one iteration each** — a rollout for
  the witness observations and 32 forward passes. At 25 checkpoints over 2500
  iterations that is 1 % of the run. Always on.
* **One run at a time on this card.** MJX is not polite about memory and two
  concurrent runs will make each other's timings meaningless. The 32 threads
  are the parallel resource here, and they belong to `compare` — which is
  stock MuJoCo on CPU and embarrassingly parallel across seeds.
* **The wasted-compute number to beat**: the reference run spent ~76 % of 3.9
  GPU-hours after its reward peak. Fixing that is worth more than any
  hardware.

`MUJOCO_GL` is unset box-wide and there is no precedent for setting it — the
trainer is headless MJX and never opens a renderer. If video rollout is ever
added, set `MUJOCO_GL=egl` at that call site, not in the environment.

## 5. Bursting off-box

Flywheel can rent compute on our behalf (`flywheel_compute_list_options`,
`flywheel_compute_acquire`, `flywheel_launch_execution`, …). Live prices from
this account on 2026-08-02, cheapest first:

| Provider | SKU | ¢/hr |
|---|---|---|
| primeintellect | CPU 4v/16G spot (datacrunch) | 1 |
| vastai | 1× L4 24 GB | 3 |
| vastai | 1× Tesla V100 32 GB | 3 |
| vastai | 1× RTX 3090 24 GB | 8 |
| vastai | 1× RTX 5070 12 GB | 11 |
| vastai | 1× RTX 4090 24 GB | 14 |
| vastai | 1× RTX 5080 16 GB | 16 |
| vastai | 2× RTX 3090 24 GB | 17 |
| nebius | GB300 NVL72, 4 GPU | (0 — policy-priced, not a real quote) |

Note what is *not* in that list: an RTX 5090, or anything that beats one for
a single MJX job. The fastest single card on offer is a 4090 at 14 ¢/hr.

### The decision

**For a single run: essentially never.** A 4-hour run at 14 ¢/hr is 56 ¢ of
GPU — trivially affordable, and completely beside the point. The costs are:

* an rsync of the bundle and the model each way, plus checkpoint pulls;
* an unverified host (`vastai` rows are marked *unverified*) whose driver,
  CUDA and clock you do not control, on a workload whose entire premise is
  numerical reproducibility;
* a second trainer checkout to keep in sync, which is exactly the failure
  ADR-104 exists to refuse; and
* a slower card than the one already idle in this room.

The 5090 is here, it is pinned, and it is the machine every recorded number
came from. Comparability is worth more than 56 ¢.

**For a parallel sweep: this is the real case.** The things cdx-rl actually
wants many of are *independent*, and that is where a fleet earns its keep:

* a seed sweep — the same task at 8–12 seeds, because a single seed is an
  anecdote;
* a shove-band sweep across the capture-point range, which is how ADR-106's
  finding would have arrived in one afternoon instead of three runs;
* an actuator-limit sweep (the hardware limit vs the mechanism limit, as two
  populations rather than two arguments);
* a reward-weight sweep, once there is a metric that is not the reward.

Twelve 4090s for 4 hours is **$6.72**. On this box that is two days. When the
question is genuinely a sweep, rent.

**Never rent for `compare` or `capability`.** Both are stock MuJoCo on CPU
and take seconds; the 9950X's 32 threads are more than enough, and every
extra machine is another place the "reload the model per episode" bug
(ADR-103 §9) can be got wrong differently.

### Preconditions before any burst

1. The trainer venv on the rented box is built from
   `training/requirements.txt` with the **exact** pins — `mujoco==3.10.0`,
   `mujoco-mjx==3.10.0`, `jax==0.7.2` (cuda12 wheel), `numpy==2.5.1`. The
   bundle records `mujoco_version`; a mismatch is a run whose numbers cannot
   be compared with anything here.
2. The trainer checkout is at a **recorded commit**, and it is the same one
   for every member of the sweep. ADR-104.
3. The policy header's `jax.default_backend()` is checked on the way back. A
   run that silently fell back to CPU is visible in the artifact rather than
   only in how long it took.
4. Each run's bundle travels with it and comes home with it.
5. The graph node says where it ran, on what, at which commit — and carries
   the `execution_id` if it went through Flywheel. Without one there is
   nothing structural to say it, so it has to be in the content.

## 6. What lives where, in one table

| | Path | Tracked? |
|---|---|---|
| Cadex checkout (engine, trainer, docs) | `/home/theo/cadex` | **read-only to cdx-rl** |
| Trainer venv | `/home/theo/cadex-train-venv` | referenced, never rebuilt |
| Existing training runs | `/home/theo/cadex-jobs` | read-only input |
| cdx-rl repo | `/home/theo/cdx-rl` | git, `origin` on GitHub |
| cdx-rl tooling venv | `cdx-rl/.venv` | gitignored, `uv sync` rebuilds it |
| cdx-rl projects (`.cadex` stores) | `cdx-rl/projects` | gitignored — rebuildable from the script |
| cdx-rl training runs | `cdx-rl/jobs` | gitignored — the graph carries what matters |
| The record | Flywheel, root `rapid-bar-6214` | the durable one |
