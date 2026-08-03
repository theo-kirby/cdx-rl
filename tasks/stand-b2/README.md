# `stand-b2` — the standing biped task, committed because it exists nowhere else

```
stand-task.json    21fe4171a5499258379e20e04a091fd33c8e1a04b5f2d19acb27264140a7f235
model-model.xml    e3511559eeb3d5ab3a369852a6c7d7e5fcfc256a259f1f4f41284ffb2cef0ae1
```

43 KB. Both are byte-identical copies of
`/home/theo/cadex-jobs/stand-task-20260802-200109/`, which is a **read-only
input** to this repository — copying *out* of it is fine; nothing here writes
back.

## Why these are in git when `MUJOCO.md` §7 says projects are not

They are not a project. They are the **exported artifacts** of one, and when
this file was written the project that exported them could not be found.

`MUJOCO.md:1521` and ADR-100 both name `~/cdx-mjc/mg-legs.cadex/script.py` as
the source. A search of the filesystem for the authoring calls
(`assembly.reset_variation`, `assembly.termination`, `assembly.disturbance`)
and for this machine's joint names (`ankle_roll`) found only the exported
bundles, the derived results in `experiments/`, and Cadex's own API
definition and docs — **no `.py` that authors this mechanism**, and no
`.cadex` project store anywhere except experiment 000's pendulum.
`MUJOCO.md:2604` explains how that happens: a rebuild is keyed by script
digest and *replaces* `script_artifacts/`, so a finished run's task can vanish
locally while its checkpoints sit beside it. What survives is the copy rsynced
to the training box — which is what `cadex-jobs/` is.

> **RESOLVED 2026-08-03, and the search was looking on the wrong machine.**
> `~/cdx-mjc` was never on a training box. It is on the **macOS laptop**,
> where every `mg-legs` run from M9 through B8 was authored and dispatched,
> and it was intact. The script is now committed at
> [`mechanisms/mg-legs/script.py`](../../mechanisms/mg-legs/script.py), and
> the **B6 revision** — which is the closest committed ancestor of the
> mechanism these bytes describe — at
> `mechanisms/mg-legs/history/0018-a3a7bd262040.py`.
>
> **This does not make `e3511559…` reproducible, and the distinction is worth
> keeping.** `stand-b2` was exported by a revision between B2 and B6 that is
> not among the two kept, so re-deriving these exact bytes would mean finding
> the matching revision in the laptop's `script_history/` and rebuilding on a
> box whose OCCT agrees to the digest. What *has* changed is that the
> mechanism is no longer a dead end: it can be re-authored, re-measured and
> moved forward, which is what experiment 003 did.
>
> The lesson generalises, and `cloud.md` now carries it: **"searched for and
> not found" is a claim about the machines you searched.** Two boxes were
> searched and the work had been done on a third.

Every number in experiments 001 and 002 is a statement about these exact
bytes, and until 2026-08-02 they lived in one directory on one machine with
no backup. That is a worse risk than the one the "no large binaries" rule
protects against, and 43 KB is not the case that rule is about.

The mechanism is **B2** (ADR-105, 2026-08-02): 302.01 g, standing CoM
144.210 mm, 10 joints and 10 actuators at ±86 N·mm (ADR-086's re-rating), 31
of 32 randomisation entries. Not the older 8-joint `mg-legs`.

## Using it

```bash
uv run python tools/train.py --bundle tasks/stand-b2/stand-task.json --label NAME …
```

`train.py` copies both files into the run directory, matching the model by the
bundle's own `model.sha256` rather than by filename, so a run directory stays
self-describing.

## On another box, pin the trainer

The bundle is only half of what makes a run comparable. The other half is the
update rule:

```bash
--require-trainer aacfa82318e4e2399f65cf2ffe234504a288b586a178fc1dbe32e539a1fe7b24
```

That is `training/cadex_train.py` at Cadex `06d1374b`, and it is what produced
`stand-task-20260802-200109` and both of experiment 002's seeds. **Same seed
and same hyperparameters mean nothing against a different trainer** — the run
would be comparable to itself and to nothing else, and the manifest would
record the difference without anything having stopped. ADR-104 says refuse;
until experiment 002 that refusal lived only in `remote_train.sh`, which local
dispatch never touches.

The venv pins matter for the same reason and are checked by `tools/smoke.py`:
`mujoco==3.10.0`, `mujoco-mjx==3.10.0`, `jax==0.7.2`+cuda12, Python 3.12.3.

## What is *not* here

The checkpoints. Experiment 002's run directories live in `jobs/`
(gitignored, 7.8 MB); the two best networks are attached to the graph node
`holy-recipe-7414`, and everything concluded from them is in
`experiments/002-seed-replication/results/`. Re-running seeds 2 and 3 needs
this bundle and a GPU, not the old checkpoints — the command is in
`experiments/002-seed-replication/results/stopped.md`.

**One caveat if they run elsewhere.** Experiment 002 measured that seed 0 does
not reproduce `200109` bitwise *on the same card* — 0 of 1500 iterations, r =
+0.9885. A different box adds an axis 002 did not measure. That is fine for
every claim 002 makes, all of which are about shape; it is not fine to leave
unrecorded, so say which box in §8.
