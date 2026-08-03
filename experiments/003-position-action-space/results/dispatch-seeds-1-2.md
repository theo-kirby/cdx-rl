# 003 — seeds 1 and 2, dispatched on `sb1x`

Dispatched **2026-08-03**, cdx-rl at `2483bd4`, clean tree, smoke **13/13**.
Cadex checkout `06d1374b`, clean — the pinned revision.

**This file was written before the run started.** `method.md` / ADR-097: a
metric chosen after looking at the curve is a metric chosen by looking at the
answer. Everything in §"What would count" below is committed in advance and
is not edited afterwards; where the result departs from it, the README says
so and says why.

## Why these two runs

003 is **one seed**, and experiment 002 is the reason that is not enough. 002
asked whether 001's headline was one run's accident and the answer was
**partly**: the claim went from "2 of 2" to **2 of 3**, and the seed that
broke it did not show a weaker effect — its post-peak survival-vs-reward
correlation was **+0.50** against seeds 0 and 1's **−0.71** and **−0.83**, the
opposite sign. Everything 003 §"What to do next" proposes — raising `height`,
the four-way ablation, the B9 curriculum — is built on top of a result with
n=1.

So: seeds 1 and 2, the 002 treatment.

## The one departure from seed 0: 1800 iterations, not 1200

Seed 0 ran 1200. These run **1800**, and the reason is that it is free.

Checkpoints land every 50 iterations in both, so an 1800-iteration run is a
**strict superset** of a 1200-iteration one — the replication comparison is
made at the 1200 mark exactly as if the run had stopped there, and the extra
600 iterations are additional evidence that costs nothing in comparability.

The extra 600 buy a second question that seed 0 could not answer. Its
conjunction climbed 3/12 → 6/12 → 7/12 between iterations 700 and 1050, then
sat at 7, 6, 7, 7, 6 across 1050–1199 with mean episode length **232.7 of a
300-step cap**. That is either a plateau or a pause, and 150 iterations
cannot distinguish them. Two fresh seeds at 1800 can.

What this costs, stated plainly: seed 0's 1200 was chosen so that
1200 × 2048 × 40 × 0.02 s ≈ **546 h of robot time**, which is B6's 2400
iterations at 100 Hz exactly. 1800 is **819 h**, 1.5× B6, so the "same
experience as B6" property holds only when these runs are read at their 1200
checkpoint. Read them there for the replication claim and past it for the
headroom claim; do not mix the two.

## What is pinned, and how it is checked

| | |
|---|---|
| bundle | `5572adf265aa51cb4cfea2c454695c42a4f1c2402a6788bc696c0129f14fc595` — `tasks/stand-b8/stand-task.json` |
| model | `80eaa18f6025d589796315fcad45bb70bf72e55da750864f60b5e0b3cc71fdb3` — `tasks/stand-b8/model-model.xml` |
| trainer | `aacfa82318e4e2399f65cf2ffe234504a288b586a178fc1dbe32e539a1fe7b24`, **enforced** by `--require-trainer` |
| host | `sb1x` (RTX 5090 32 GB, driver 580.159.03), recorded in `sweep.json` |

The bundle digest is the same one `results/sweep-12-seeds.json` records as
`task_sha256`, so these seeds train on the bytes seed 0 was scored on.

**A different Cadex commit is not automatically a different trainer, and the
converse also holds** — the digest is the thing that is checked.

## The fourteen

All fourteen are passed explicitly. `train.py`'s defaults are
`RUN_200109`, the **B2-era** run from experiment 001, and B8 differs from it
in two: `discount` 0.99 against 0.995, and `gae_lambda` 0.95 against 0.97.
A partial passthrough would silently have trained a different algorithm and
no output would have shown it.

From `plan.md` §4, which is the pre-dispatch document for seed 0:

```
--envs 2048 --unroll 40 --epochs 5
--discount 0.99 --gae-lambda 0.95
--initial-std 0.4 --entropy 2e-3 --hidden 64 64
--learning-rate 3e-4 --clip 0.2 --value-weight 0.5
--iterations 1800 --checkpoint-every 50 --seeds 1 2
```

Only `--iterations` and `--seeds` differ from what seed 0 ran. The resolved
set is written to `hyperparameters.json` in each run directory; check it
against this table rather than trusting this table.

## Budget and the wall cap

Seed 0 measured **3 h 22 m for 1200 iterations on sb1x** = 10.1 s/iteration.
1800 iterations ≈ **5 h 03 m** per seed, ≈ **10 h 06 m** for both, run
sequentially — one run at a time on this card.

`--timeout 25200` (7 h) per seed. **The 10800 that was right for 001 and 002
would guillotine every one of these at roughly iteration 1070**, before the
replication comparison point. A wall cap does not travel between experiments
any more than it travels between boxes; `supervise` projects the finish from
measured throughput after iteration 10 and will say so.

`--patience 0`: reward patience is off, and experiment 002 is why — the
trainer's scalar fell 43 % in seed 2 while survival held and episode length
rose to the longest of the run.

## What would count — written before the run

Scored with `harness steps` on the conjunction **stepped ≥ 10 mm AND survived
300/300**, 12 evaluation seeds, escalating to 24 for ties. Note the threshold
is a *distance*, not a step count, and the airborne threshold is a
*duration* — 003 moved to 50 Hz and a threshold in control steps would
silently have changed meaning.

**1. Replication (primary).** Seed 0's best checkpoint at or before iteration
1200 scored **7/12**, against B6's 6/12. A seed replicates if its best
checkpoint at or before 1200 scores **≥ 7/12**. Two of two replicating makes
003 a three-seed result; one of two makes it 002's situation again and the
headline has to be restated as "2 of 3".

**2. Hazard 15 (secondary, and the one I expect to hold).** Peak joint torque
at rest as a fraction of the 86 N·mm rating, nothing pushing. Seed 0:
**27.0 N·mm = 31 %**. 002's three torque-space seeds: **86.6 %, 63.3 %,
87.0 %**. This one is closest to a property of the actuation rather than of
the policy — `mj_inverse` puts the static stance at 15.6 % with no policy
involved at all — so it should replicate at **< 50 %** in both. If it does
not, the claim that the action space dissolved hazard 15 is in trouble in a
way seeds cannot rescue.

**3. Headroom past 1200 (the new question).** Does the best checkpoint in
(1200, 1800] beat the best in [0, 1200]? Judged by **McNemar over the
discordant evaluation seeds at 24**, not by the point estimate and not by the
unpaired 2σ bound — that bound is 40.8 pp at n=12 and it is the *wrong test*
besides, since checkpoints are played against the same seeds and most
episodes agree for reasons unrelated to the policy.

**Stated in advance: the most likely outcome here is "cannot separate."**
003's own three tied checkpoints scored 14, 17 and 16 of 24 and McNemar
returned p = 1.000 and p = 0.453 — indistinguishable, where the point
estimates suggested a winner and were written up as one before being
corrected. If these seeds come back the same way, *that is the result*, and
"the point estimate went up" is not to be reported as headroom.

Mean control steps is recorded alongside as a continuous secondary — it is
far less quantised than a 12- or 24-episode binomial, and seed 0 ended at
232.7 of 300, so there is visible room in it.

## On completion

**Seed 0's checkpoints are on this box** —
`/home/theo/cadex-jobs/stand-task-20260803-140221/`, 25 of them, trained on
`sb1x`, on a `stand-task.json` and `model-model.xml` whose digests are
**identical** to the committed `tasks/stand-b8/` copies. So all three seeds
can be scored on one card in one sitting, against the same bytes. That
directory is invariant 3's read-only jobs store: read from it, never write to
it, and copy out rather than scoring in place if anything needs writing.

1. **Check the exit code and the checkpoint count first.** 1800 iterations at
   every 50 is 36 periodic, plus `best` and the final — 38. Fewer means the
   run stopped early, and `EXIT_SALVAGEABLE (4)` is still fully analysable.
   Rank exit codes through `SEVERITY`, never as a scale.
2. **Confirm the throughput held.** Measured 7.50 s/iteration at dispatch on
   an otherwise idle card; `sb9x` was running 002 seed 3 concurrently but the
   two boxes share nothing. `runtime.json` records the host.
3. **Score the conjunction**, all three seeds, 12 evaluation seeds first:

   ```
   uv run python -m harness steps \
     --dir jobs/stand10-s1-20260803-195223 \
     --task tasks/stand-b8/stand-task.json --profile mg-legs --seeds 12 --json
   ```

   and the same for seed 2's directory and for
   `/home/theo/cadex-jobs/stand-task-20260803-140221` (seed 0). Escalate ties
   to 24 as 003 did, and **read the McNemar line, not the point estimate.**
4. **Read each seed twice**: at its best checkpoint ≤ 1200 for the
   replication claim, and over (1200, 1800] for the headroom claim. These are
   two different questions and the 1200 boundary is what keeps the first one
   comparable to seed 0 and to B6's robot-time budget.
5. **Hazard 15** from the same episodes — peak torque at rest as a fraction of
   86 N·mm. Seed 0 was 31 %; 002's torque seeds were 63–87 %.
6. `compare` now prints the paired test too, so a `compare` run over these
   directories will say something the 001 and 002 tables could not.

## What this run cannot answer

* **Attribution within B8.** Four things changed at once — action space,
  reward sign, control rate, and the reward's functional form. These seeds
  replicate the conjunction of all four. They do not measure which one did
  the work; that is the ablation, and it is only worth running once the thing
  being ablated is known to be real.
* **B6 as a controlled comparison.** `stand8` is 55-channel and
  torque-trained and the engine refuses it against this bundle by name. 7/12
  against 6/12 is the same criterion on the same *task* and a differently
  actuated *machine*. That remains true here.
* **The 86 N·mm rating.** Still a judgment, not a datasheet number. Nothing
  in these runs validates it.
