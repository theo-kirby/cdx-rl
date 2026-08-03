# 002 — seed 3, dispatched on `sb9x`

Dispatched **2026-08-03 20:04 UTC**, cdx-rl at `2483bd4` on both boxes,
sb9x smoke **13/13**, GPU idle at 1 MiB before launch.

```
sweep dir  jobs/stand9-sweep-20260803-200432
run dir    jobs/stand9-s3-20260803-200432
detached   pid 954001 (sweep), 954003 (trainer)
```

This closes the sweep. [`dispatch-sb9x.md`](dispatch-sb9x.md) is the document
for seed 2 and everything in it about the box, the runtime settings, the
stopping rule and the wall cap applies unchanged; this file records only what
is specific to seed 3.

**It ran in parallel with experiment 003's seeds 1 and 2 on `sb1x`.** The two
boxes are independent and neither result depends on the other, but a
throughput number taken from either tonight is a number taken while the other
box was also busy — which does not matter here, since they share nothing but
the operator.

## What is pinned

| | |
|---|---|
| bundle | `21fe4171a5499258379e20e04a091fd33c8e1a04b5f2d19acb27264140a7f235` — `tasks/stand-b2/stand-task.json`, verified on sb9x before dispatch |
| model | `e3511559eeb3…` |
| trainer | `aacfa82318e4e2399f65cf2ffe234504a288b586a178fc1dbe32e539a1fe7b24`, **enforced**, and byte-identical to sb1x's |
| host | `sb9x` (RTX 4070 12 GB, driver 595.84), Cadex `ae8da6a6` |
| runtime | `stack_mb 256`, `xla_preallocate false`, `child_gc false` |

Cadex on sb9x is ten commits past sb1x's `06d1374b`, all engine-side; the
trainer digest is identical on the two boxes and that is the thing checked.

## The fourteen

`RUN_200109` exactly, as seeds 0, 1 and 2 ran it — all passed explicitly:

```
iterations 1500, checkpoint_every 100, envs 2048, unroll 40, epochs 5,
hidden [64,64], lr 3e-4, discount 0.995, gae_lambda 0.97, clip 0.2,
entropy 2e-3, value_weight 0.5, initial_std 0.4, seed 3
```

Note these are **not** experiment 003's — `discount` is 0.995 against 003's
0.99 and `gae_lambda` 0.97 against 0.95. The two experiments were dispatched
within twenty minutes of each other tonight and mixing the two sets would
have produced a run belonging to neither.

`--timeout 20000` and `--patience 0`, both as seed 2.

## What it is for, and what would count

§7 criterion 5, unchanged and stated before dispatch: **is the trainer's
reward peak the best checkpoint?** Judged within the seed — the peak off this
run's own curve, the best checkpoint off this run's own `compare`.

The three-seed table stands at **2 of 3**: seeds 0 and 1 say no by 41.7 pp
and 52.1 pp, seed 2 ties at +2.1 pp. Seed 3 makes it 2 of 4, 3 of 4, or —
if it ties — leaves the headline at 2 clear of 4 with two ties.

**Stated in advance:** a tie is a result, not a failure to get one. Seed 2's
tie was informative precisely because its post-peak correlation was **+0.50**
against seeds 0 and 1's −0.71 and −0.83 — the opposite sign rather than a
smaller magnitude, and a flat survival curve from iteration 199 on with no
late optimum to beat the peak with. Record seed 3's post-peak correlation and
the shape of its survival curve whichever way the headline falls.

Hazard 15 is the secondary: 3 of 3 so far at **86.6 %, 63.3 %, 87.0 %** of the
86 N·mm rating with nothing pushing.

## The known risk

sb9x carries an intermittent general protection fault in jaxlib's CUDA
plugin — a race, measured at **1 in 3** over forty-iteration runs, and seed 2
hit it at 4.13 h having written 14 of its 15 checkpoints. That is
`EXIT_SALVAGEABLE (4)`, and it is analysable in full: `checked_policy` runs
the witness before writing, so a crash can lose a file but never corrupt one.
Only `stand9.cxpolicy` — the final iteration, which ADR-099 says you do not
select — would be lost.

So if this seed comes back `4` rather than `0`, **use it**. Check the
checkpoint count first: 15 periodic plus `best`.
