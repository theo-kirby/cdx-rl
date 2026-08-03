# 002 — stopped early, by request

The sweep was cancelled after **2 of 4 seeds** because the GPU was needed for
other work. §7 criterion 1 requires that a seed either completes 1 500
iterations **or the reason it stopped is recorded**. This is that record.

Stopped **2026-08-03**, 5 h 06 m after dispatch.

---

## What was asked for, and what ran

| seed | iterations | state | reward peak | witness | wall |
|---|---|---|---|---|---|
| **0** | **1500 / 1500** | `done`, exit 0 | **594** @ 0.3327 | **517×** | 2.12 h |
| **1** | **1500 / 1500** | `done`, exit 0 | **510** @ 0.3158 | **588×** | 2.12 h |
| 2 | 537 / 1500 | **stopped by SIGTERM** | 452 @ 0.3344 *(partial)* | — | 0.82 h |
| 3 | — | **never dispatched** | — | — | — |

**GPU-hours spent: ~5.1**, against the ~9.4 budgeted in §6.

## How it was stopped

The sweep driver was signalled **first**, deliberately. `tools/train.py`
runs seeds sequentially in one process, so terminating the trainer alone
would have been read as "this seed ended" and seed 3 would have been
dispatched into a GPU that was being freed.

```
sweep driver pid 653428   trainer pid 849287
sweep driver TERMed  -> gone
trainer TERMed       -> exited
```

Afterwards the card was back to its pre-flight baseline exactly:

```
0 %, 637 MiB, 32607 MiB          # 614 MiB of it the same 35-day-old
                                 # python3 recorded in preflight.md
```

No `cadex_train` process survived.

## Seed 2 is a partial run, and is labelled as one

It holds six policies — `000100` … `000500` plus `best` — all complete
files; the SIGTERM landed between checkpoints, so nothing half-written was
kept. What it does **not** have is a final `.cxpolicy`, and therefore no
`reward_curve` in a header and no witness margin: the trainer prints its
witness when it finishes, and this one did not.

Its `progress.json` still says `state: training`, which is now false, and
`supervise` says so rather than believing it:

```
liveness
  pid            849287  alive=False
  STALE — this run claims to be training and nothing is running it.
          A stale progress.json is indistinguishable from a live
          one without this check; do not wait on it.
```

That is the check working as specified. It is worth noting that this is the
branch the zombie fix (`DESIGN.md` finding 10) was about — though here the
pid was fully reaped when its parent died, so it would have read correctly
either way.

## What this costs the experiment

**§7 criteria 1, 2 and 6 are met on the seeds that ran.** Both complete
seeds finished all 1 500 iterations with witness margins of 517× and 588×
against a 100× floor, and seed 0's replication control has already been
measured (see `seed0-vs-200109.txt`).

**Criterion 5 — the headline — is weakened, and honestly so.** It asks in
how many of *four* seeds the trainer's reward peak is not the best
checkpoint. With two complete seeds it can only ever be answered "in N of
2", and two runs plus the one 001 already had is a much weaker claim about
replication than four would have been. **This does not become a four-seed
result by rounding.** Any node published from it says two.

Nothing about the stop invalidates what ran: the seeds are independent by
construction (`--seed` drives both `PRNGKey` and the episode-variation
draws), so seeds 0 and 1 are exactly as valid as they would have been had 2
and 3 followed them.

## Reproducing the missing half

The two seeds are a `train.py` invocation away, unchanged — the manifest,
the digests and all fourteen hyperparameters are recorded in
`jobs/stand9-sweep-20260803-010902/sweep.json` and in `dispatch.md`:

```bash
uv run python tools/train.py \
  --bundle /home/theo/cadex-jobs/stand-task-20260802-200109/stand-task.json \
  --label stand9 --seeds 2 3 \
  --iterations 1500 --envs 2048 --checkpoint-every 100 \
  --unroll 40 --epochs 5 --hidden 64 64 \
  --learning-rate 3e-4 --discount 0.995 --gae-lambda 0.97 \
  --clip 0.2 --entropy 2e-3 --value-weight 0.5 --initial-std 0.4 \
  --detach --supervise --require-device gpu --patience 0 --timeout 10800
```

~4.3 h at the 4.29 s/iteration this sweep measured. Seed 2 would start over
rather than resume; there is no resume, and its 537 iterations are not
comparable to a run that trained straight through.

### On `sb9x`, three things in that command are wrong

Recorded **2026-08-03**. The command above is correct for `sb1x` and was
carried to a second box unchanged; it fails there in three ways, none of them
loudly.

1. **The bundle path does not exist.** `/home/theo/cadex-jobs/` on sb9x holds
   an unrelated, older run. Use the committed copy, which is byte-identical
   to what 200109 trained on (`tasks/stand-b2/README.md`):
   `--bundle tasks/stand-b2/stand-task.json`.
2. **`--timeout 10800` truncates every seed.** sb9x measures **8.93
   s/iteration** steady against sb1x's 4.29–5.62, plus a 65 s compile and
   **~106 s per checkpoint**, so 1500 iterations with 15 checkpoints needs
   **~4.2 h**, not 2.34. A 3 h cap terminates each seed around iteration
   1 050 — and what that leaves is a directory of checkpoints with no final
   `.cxpolicy` and no witness, which is *precisely the artefact this file
   documents*. Two of those would be indistinguishable from one another and
   from a deliberate stop. Use `--timeout 20000`. `supervise` now projects
   the finish from a measured slope after iteration 10 and says so, so this
   is caught in minutes rather than hours.
3. **The trainer segfaults at 2048 environments**, and this one is **not
   solved — but it is survivable.** `tools/train.py` defaults three runtime
   fixes on (`cloud.md` §1), which is enough for short runs and not enough at
   length: a 40-iteration validation still exited `-11` as `train()`
   returned. What it left behind, however, is a usable experiment —
   see below. `train.py` now returns `EXIT_SALVAGEABLE` (4) for exactly this
   state rather than calling it infrastructure failure.

`--require-trainer aacfa823…` **passes on sb9x**: its Cadex sits ten commits
past `06d1374b`, but all ten are engine-side and `training/` is byte-identical.

The corrected dispatch, then, is the same command with one path and one
number changed:

```bash
uv run python tools/train.py \
  --bundle tasks/stand-b2/stand-task.json \
  --label stand9 --seeds 2 3 \
  --iterations 1500 --envs 2048 --checkpoint-every 100 \
  --require-trainer aacfa82318e4e2399f65cf2ffe234504a288b586a178fc1dbe32e539a1fe7b24 \
  --detach --supervise --require-device gpu --patience 0 --timeout 20000
```

**~8.4 h for the two**, against 4.3 h on sb1x. Every hyperparameter still
defaults to what 200109 ran, so the fourteen need not be repeated.

**Expect each seed to exit `-11`, and expect it to be usable anyway.** This
was checked rather than assumed, because "the run crashed" and "the run is
lost" are not the same statement and the difference here is eight GPU-hours.

On a 40-iteration run that died exactly this way:

* both `.cxpolicy` files on disk parsed as **complete policies** — full
  header, weights, `network`, `normaliser`, and the `model`/`task`/`trainer`
  digests;
* `checked_policy` runs the witness **before** writing, so a checkpoint that
  exists is one that passed — the crash cannot leave a bad file, only a
  missing one;
* the `.best` header carried the **whole 40-row `reward_curve`**, and
  `train.log` carries the series independently; and
* **`compare` consumed them end to end** — played both checkpoints in stock
  MuJoCo over 6 seeds, produced per-motor peak and mean torque, the hazard-15
  column, and a selection verdict with its binomial bound.

The `.best` file was also rewritten ~20 times during that run, each rewrite
running a fresh witness pass, without faulting. **The checkpoint path is not
what breaks; `train()`'s return is.** So a 1500-iteration seed should write
all 15 periodic checkpoints and die after the last one.

What is actually lost is `stand9.cxpolicy` — the **iteration-1500** network.
That is not the selection target: ADR-099, and both 001 and 002, say a
checkpoint is chosen by what it did when played, and 002 measured the final
iteration losing to earlier ones in 2 of 2 seeds. **Criterion 5 does not
depend on the file the crash destroys.**

Two things to hold to, though. The 1500-iteration case is **inferred from a
40-iteration run**, not measured — check the checkpoint count on completion
before concluding anything. And a seed that produces *no* checkpoints is a
different failure and is reported as such (`EXIT_INFRASTRUCTURE`, not 4).

**What this costs the comparison.** Seeds 0 and 1 ran on sb1x; 2 and 3 would
run on a different card, driver and runtime configuration. Criterion 5 is
answered *within* a seed — the reward peak and the best checkpoint are read
off the same run's own curve and its own `compare` — so it survives the move.
A claim about a *value* shared across seeds would not, and 002 already
measured that seeds do not reproduce bitwise even on one card. §8 must name
the box per seed.
