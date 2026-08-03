# 002 — pre-flight

Every line below was executed before the GPU started, and its real output
pasted. `method.md`, and the checklist in the README's §5.

Recorded **2026-08-03**, on `sb1x`.

---

## `uv run python tools/smoke.py` — PASS, 13/13

```
ok    engine_resolved        {"protocol": "cadex-cadexd-v1", "root": "/home/theo/cadex", "version": "dev-tree"}
ok    cadexd_handshake       {"ready_seconds": 0.03}
ok    open_project           {"revision": ""}
ok    describe_api           {"domains": ["assembly", "mesh", "part", "partdesign", "sketcher"], "engine": "xscript"}
ok    dynamics_surface       {"assembly_exports": 23}
ok    write_script           {"digest": "913815cd…", "outputs": ["cube"]}
ok    resolve_outputs_dir    {"artifact_count": 1, "kinds": ["brep"]}
ok    inspect_output         {"output_count": 1, "revision": "104826f0…"}
ok    digest_stable          {"first": "913815cd…", "second": "913815cd…"}
ok    trainer_pins           {"jax": "0.7.2", "mujoco": "3.10.0", "mujoco-mjx": "3.10.0", "python": "3.12.3"}
ok    trainer_gpu            {"backend": "gpu", "devices": ["cuda:0"]}
ok    nvidia_smi             NVIDIA GeForce RTX 5090, 32607 MiB, 580.159.03
ok    cadex_checkout         {"dirty": false, "head": "06d1374b …"}

PASS
```

`version: "dev-tree"` is the setting that matters — the Cadex *checkout*, not
the stale staged payload (`CLAUDE.md` §4). `dynamics_surface` asserting 23
assembly exports is what proves it: the payload's exports stop at
`exploded_view`.

## Cadex checkout — `06d1374b`, clean

```
$ git -C /home/theo/cadex log --oneline -1
06d1374b Refuse to dispatch to a box running a different trainer (ADR-104)

$ git -C /home/theo/cadex status --porcelain
(no output)
```

ADR-104 §8b's expected commit. Read-only throughout; verified again after the
run in §8.

## Trainer — byte-identical to the one that produced 200109

```
$ sha256sum /home/theo/cadex/training/cadex_train.py
aacfa82318e4e2399f65cf2ffe234504a288b586a178fc1dbe32e539a1fe7b24
```

200109's `.cxpolicy` header records `training.trainer_sha256` as **the same
digest**. This is what makes §7 criterion 6 — seed 0 as a replication control
— a real control rather than a hopeful one: same seed, same hyperparameters,
same trainer, same task, same card.

## `nvidia-smi` — idle

```
name, utilization.gpu [%], memory.used [MiB], memory.total [MiB]
NVIDIA GeForce RTX 5090, 0 %, 637 MiB, 32607 MiB

pid, used_gpu_memory [MiB]
2732148, 614 MiB
```

0 % utilisation. The 614 MiB belongs to pid 2732148, a `python3` with an
elapsed time of **35 days** — it long predates this repository and is not a
training run. Reported rather than rounded to "idle", because "<1 GB used"
passing on a card with something resident is worth being able to see later.
No training run is competing for the card: one run at a time (`cloud.md`).

## `df -h /home/theo` — 2.4 T free

```
/dev/nvme0n1p2  3.6T  1.1T  2.4T  33% /
```

The budget is ~40 MB of checkpoints. Not close to a constraint.

## Bundle and model digests

```
21fe4171a5499258379e20e04a091fd33c8e1a04b5f2d19acb27264140a7f235  stand-task.json
e3511559eeb3d5ab3a369852a6c7d7e5fcfc256a259f1f4f41284ffb2cef0ae1  model-model.xml
```

Both under `/home/theo/cadex-jobs/stand-task-20260802-200109/`, and both match
what experiment 001 recorded. The task is unchanged by digest, which is the
claim §4 and §5 rest on.

## Divergence guard — both branches fired on purpose

`harness/DESIGN.md` §6: *"every check must be able to fail, and must have been
made to fail once on purpose."*

```
$ uv run python tools/fire_divergence_guard.py
Firing supervise's divergence guard on purpose (DESIGN.md §6).
sigma floor 0.02

--- control (healthy progress.json) ----------------------
  action_std 0.3801, loss 24.7, floor 0.02
  divergence() returned None
  RESULT       CORRECTLY SILENT

--- control (older trainer, no action_std) ---------------
  divergence() returned None
  RESULT       CORRECTLY SILENT

--- non-finite loss -----------------------------------------
  patched      {'loss': nan}
  victim pid   648707 (alive: True)
  it    300/1500  reward/step +0.2100  episode 288.4  sigma 0.3801  best 298@0.2200  training
  STOPPING: loss is nan — the optimiser has diverged
  events       ['divergence']
  check        ['non-finite']
  stopped      {"exited": true, "pid": 648707, "signalled": true}
  process dead True   after 0.00s
  RESULT       FIRED

--- sigma collapse ------------------------------------------
  patched      {'action_std': 0.004}
  victim pid   648708 (alive: True)
  it    300/1500  reward/step +0.2100  episode 288.4  sigma 0.0040  best 298@0.2200  training
  STOPPING: action_std 0.00400 is below the floor 0.02 — exploration has collapsed
  events       ['divergence']
  check        ['sigma-collapse']
  stopped      {"exited": true, "pid": 648708, "signalled": true}
  process dead True   after 1.00s
  RESULT       FIRED

================================================================
  non-finite loss    FIRED   (killed pid in 0.00s)
  sigma collapse     FIRED   (killed pid in 1.00s)
  healthy control    silent
  no-sigma control   silent

PASS — both branches stop a live process
```

Both controls are there because a guard that always fires is as useless as
one that never does. The second control — an older trainer's run, which logs
no `action_std` at all — must read as *unknown* rather than as zero, or every
`job-task-*` run on this box would trip σ collapse on its first poll.

### What firing it found

The first run of this took **60.00 s per branch** and recorded
`"exited": false` about a process that was, in fact, dead.

`cadex_train.py` is a *child* of `tools/train.py`, and `train.py` only calls
`process.wait()` after `watch()` returns. Between the trainer exiting and the
watch loop ending it is therefore an unreaped **zombie** — and
`os.kill(pid, 0)` succeeds on a zombie. Three checks were silently wrong:

* `_stop` burned its full 60 s grace, then recorded the opposite of the truth;
* `watch`'s liveness branch **could never fire at all** — a trainer that
  crashed at 02:00 would hold the supervisor until `--timeout`, costing a
  seed its entire three-hour wall cap for a run that was not running;
* `liveness`'s `stale` — *"it claims to be training and nothing is running
  it"* — read `pid_alive: True` for exactly the case it exists to catch.

`runlog.process_gone()` now reads `/proc/<pid>/stat`. The output above is
after the fix: 0.00 s and 1.00 s, and `exited: true`.

This is the second time a check in this harness has been found to fail
*quietly* rather than loudly — the first was `WITNESS_RE` silently reporting
"no witness margin" for a margin of `1,141x` (experiment 000). Both were
caught only by deliberately pointing the check at something it should have
caught.

## Sweep machinery — exercised on CPU first

Two seeds, three iterations, `--envs 16`, `--cpu`, against experiment 000's
pendulum bundle. Not a result; a proof that the plumbing does what it says.

* `--detach` returned in **0.036 s**, and the sweep outlived the shell.
* Two run directories, `smoke-s7-…` and `smoke-s8-…`, each with its own
  bundle, model, `progress.json`, `train.log`, `train.pid` and
  `hyperparameters.json`.
* `sweep.json` named both runs, the shared hyperparameters, and the
  bundle/model digests.
* The trainer's first-iteration `sigma 0.3994` is the passthrough proving
  itself: `initial_std` reached the trainer as **0.4**. The trainer's own
  default would have printed 0.3, and that one line is the difference between
  replicating 200109 and replicating something else.

Deleted afterwards; `jobs/` is gitignored regardless.
