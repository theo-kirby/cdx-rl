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
