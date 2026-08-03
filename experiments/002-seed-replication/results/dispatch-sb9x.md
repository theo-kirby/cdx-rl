# 002 — seed 2, dispatched on `sb9x`

Dispatched **2026-08-03 12:17 UTC**, cdx-rl at `979ae20`, clean tree.

```
sweep dir  jobs/stand9-sweep-20260803-121736
run dir    jobs/stand9-s2-20260803-121736
detached   pid 815205 (sweep), 815207 (trainer)
```

## One seed, not two

`stopped.md` records the missing half as seeds **2 and 3**. This dispatches
**seed 2 alone**, and the reason is a measurement rather than caution.

sb9x carries an intermittent fault (`cloud.md` §1, graph node
`winter-mouse-1809`): a general protection fault in jaxlib's CUDA plugin that
struck **1 of 3** forty-iteration runs. That base rate was measured at **40
iterations and 2 checkpoints**. A real seed is **1500 iterations and 15** —
25x the wall time, 7x the compiles. If the hazard scales with either, the
per-seed probability is higher than one in three, and *nobody has measured
which*.

Dispatching both seeds would spend 8.4 h to find that out. Dispatching one
spends 4.2 h and answers the same question, because **the failure mode is
known to be survivable**: every checkpoint the trainer writes is complete and
witness-checked before a byte lands, and `compare` was verified to consume a
crashed run's checkpoints end to end. So seed 2 either finishes or it
doesn't, and either way it produces the checkpoints criterion 5 needs.

Seed 3 follows once this one has shown what a full-length run does here.

## What is pinned, and how it is checked

| | |
|---|---|
| bundle | `21fe4171a549…` — `tasks/stand-b2/stand-task.json` |
| model | `e3511559eeb3…` |
| trainer | `aacfa82318e4…`, **enforced** by `--require-trainer` |
| host | `sb9x` (RTX 4070 12 GB, driver 595.84), recorded in `sweep.json` |
| runtime | `stack_mb 256`, `xla_preallocate false`, `child_gc false` |

sb9x's Cadex is at `ae8da6a6`, ten commits past 002's `06d1374b` — but all ten
are engine-side and `training/` is byte-identical, so the update rule is the
one that produced 200109 and seeds 0–1. The digest is what is checked; the
commit is the noisier signal.

All fourteen hyperparameters are passed explicitly and equal `RUN_200109`:
`iterations 1500, envs 2048, unroll 40, epochs 5, hidden [64,64], lr 3e-4,
discount 0.995, gae_lambda 0.97, clip 0.2, entropy 2e-3, value_weight 0.5,
initial_std 0.4, checkpoint_every 100, seed 2`.

## The stopping rule, unchanged

`--patience 0`. 001 Phase A found no reward-based rule would have found the
right checkpoint, and 002 measured survival-vs-reward at **−0.71 and −0.83**
after the peak in its two seeds. A reward-patience stop would destroy this
experiment's subject.

Stops on: non-finite loss or reward, σ below 0.02, a device that is not
`gpu`, a dead process — and a **20 000 s** wall cap.

### Why 20 000 s and not 10 800

sb1x's cap was 3 h. sb9x measures **8.93 s/iteration steady**, plus a 65 s
compile and **~106 s per checkpoint**, so 1500 iterations with 15 checkpoints
is **~4.2 h**. The old cap would terminate this seed around iteration 1 050
and leave exactly the artefact `stopped.md` documents — for the second time,
and avoidably.

`supervise` now projects the finish from a measured slope after iteration 10
and refuses to be quiet about a cap it cannot meet. If the projection fires
on this run, the cap is wrong and the run should be re-dispatched rather than
left to be truncated.

## What this seed is for

§7 criterion 5: **in how many seeds is the trainer's reward peak not the best
checkpoint?** Answered *within* a seed — the reward peak comes off this run's
own curve, the best checkpoint off this run's own `compare` — so the change of
card does not invalidate it. A claim about a *value* shared across seeds would
not survive the move, and 002 already measured that seeds do not reproduce
bitwise even on one card.

**§8 must name the box per seed**: 0 and 1 on `sb1x`, 2 (and later 3) on
`sb9x`.

## On completion

1. Check the **checkpoint count** — 15 periodic plus `best`. Fewer means the
   fault struck mid-run, and where.
2. `compare --dir … --task …` over the checkpoints; `--seeds 12` cannot crown
   a winner (20 pp bound) but can reject one.
3. Record the exit code. `0` and `4` (`EXIT_SALVAGEABLE`) are both usable
   outcomes and mean different things; `1` means nothing was produced.
