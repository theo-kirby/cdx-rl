# experiments/

One directory per experiment, numbered in the order they were started:
`NNN-short-name/`.

Numbering is chronological and never reused. An experiment that gets
superseded keeps its number and its directory; the successor gets a new one
and says what it supersedes. The point of a research record is that the
things that did not work are still there.

## What a directory contains

| | |
|---|---|
| `README.md` | **required.** The nine sections below. |
| `rig.py` | the Cadex script, if the experiment authors a mechanism |
| `task.md` | the task's design, when it is long enough to want its own file |
| `results/` | tables, envelopes and plots — small text, committed |
| — | run directories live in `jobs/` (gitignored); checkpoints live there too |

Large binaries do not go in git. The run directory keeps them, the graph node
carries the ones that matter as artifacts, and the README says which run
directory.

## The README's nine sections

Written in this order. **Sections 1–7 before any dispatch.** 8 and 9
afterwards, and visibly separate, so nobody can be unsure which came first.

**An imported experiment keeps its pre-dispatch plan verbatim.** 003 was run
before this repository was its home, so its plan and the research note that
motivated it are committed unedited beside the README —
`003-position-action-space/plan.md`. Reconstructing sections 1–7 from memory
after seeing the answer is exactly what ADR-097 forbids; keeping the document
that chose the metric is what makes the reconstruction checkable.

1. **Question** — one sentence, phrased so that both answers are interesting.
2. **Metric** — named, defined, and why that one. Decided before dispatch
   (ADR-097), because the reward curve gets noisier the moment variation goes
   in and stops being comparable.
3. **Mechanism** — script, digest, actuator limit, and whether that limit
   models *the hardware* or *the mechanism*. Both are defensible; only one is
   what you will build.
4. **Task** — episode length, control rate, reward terms with weights,
   terminations, reset variation, disturbance band, **and the capture-point
   arithmetic that sized it** (ADR-100).
5. **Gate** — `feasibility`'s six checks and what each said.
6. **Budget and stopping rule** — iterations, environments, expected wall
   time, when to stop.
7. **Pass criteria** — written before the run.
8. **What happened** — peak and final, `capability`'s sweep, `compare`'s
   table, the termination mix, the torque columns.
9. **What it means, and what it does not mean.**

The full protocol is [`../method.md`](../method.md). The graph node mirrors
these sections; see [`../flywheel.md`](../flywheel.md) §4.

## Current

| | | |
|---|---|---|
| [`000-loop-validation`](000-loop-validation/) | prove every link in the chain on a 1-DOF rig | specified; modelling half verified |
| [`001-stand-biped`](001-stand-biped/) | separate "the policy is bad" from "the task is out of range" | Phases A and B measured |
| [`002-seed-replication`](002-seed-replication/) | does 001's shape replicate across seeds? | 3 of 4 seeds measured |
| [`003-position-action-space`](003-position-action-space/) | does a machine whose zero action *stands* learn what nine runs of one that falls over did not? | measured, imported, one seed |
| [`004-ceiling-and-clamp`](004-ceiling-and-clamp/) | can a *buildable* policy step, and is the bracing a dynamics requirement or a policy choice? | measured, two arms, two seeds — a policy choice; ±25° is the operating point |
| [`005-training-time-budget`](005-training-time-budget/) | does the buildable policy keep climbing, and stay buildable? | **retracted twice** — the pre-flight gate vetoed its own dispatch, and 005-ceiling then refuted the surviving premise |
| [`005-buildable-ceiling`](005-buildable-ceiling/) | where does the buildable policy top out? | measured — 1850 warm-started iterations; the clamped arm caught up (18/24 vs 18/24, p = 1.000) and no ceiling is established |
| [`006-step-not-shuffle`](006-step-not-shuffle/) | does costing joint velocity make the machine stand still, and what does that cost in stepping? | **measured, two seeds.** The kernel replicates (−69.4 % / −53.9 % on `Σ\|q̇\|`); the bracing falsifier fires. Not adoptable as written |
| [`007-price-the-bracing`](007-price-the-bracing/) | does the reward price bracing at the states the machine actually reaches? | **Phase A measured on CPU** — `effort` is live but trivially priced (0.69 % of budget separates 006's two basins). **Phase B not written** |
