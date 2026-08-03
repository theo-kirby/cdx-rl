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
