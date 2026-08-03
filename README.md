# cdx-rl

Research repository and Flywheel flywheel-graph for **reinforcement learning
inside Cadex** — robot mechanisms get designed here, policies get trained
here, results get measured here, and the whole evolution of the work is
recorded as a DAG.

The thesis, in one line:

```
prompt → parametric mechanism → MJCF → task → policy → verification → a thing you can build
```

Cadex has every one of those links. What it does not have — deliberately,
ADR-088 §6 — is what *drives* them: the per-project drivers, the measurement
discipline, and the record of what was tried and what it meant. That is this
repository.

## Quick start

```bash
uv sync
cp config/env.example config/env
set -a; . ./config/env; set +a
uv run python tools/smoke.py          # 13 checks; must print PASS
```

On a box whose Cadex checkout is **not built**, `smoke.py` stops at
`engine_resolved` and the eight `cadexd` checks never run. Training still
works — `tools/train.py` goes straight to the trainer venv and never touches
the engine — but nothing that authors a mechanism does. `cloud.md` §1 has the
headless build recipe, which is **not** `pixi run build-engine`.

That section also lists the three runtime settings `sb9x` needs, the segfault
that is **still open** there, why a run that hits it is nonetheless usable,
and the per-box throughput — because **wall-clock numbers in these docs are
`sb1x`'s and do not transfer**.

## Read

**[`CLAUDE.md`](CLAUDE.md) first.** Then:

| | |
|---|---|
| [`concept.md`](concept.md) | what this is and is not |
| [`cadex.md`](cadex.md) | Cadex for an agent with no context — the CLI, the protocol, ten verified traps |
| [`method.md`](method.md) | the research protocol; read before any GPU time |
| [`flywheel.md`](flywheel.md) | the graph, as it actually is |
| [`cloud.md`](cloud.md) | compute topology, and when to leave this box |
| [`harness/DESIGN.md`](harness/DESIGN.md) | the drivers, specified |
| [`cadex-wishlist.md`](cadex-wishlist.md) | wants, captured rather than acted on |

## Ground rules

`/home/theo/cadex` is **read-only** from here — no commits, no edits, no
branch changes. `/home/theo/cadex-train-venv` is referenced and never
rebuilt. `/home/theo/cadex-jobs` is a read-only input.

*Builds* were also on that list until 2026-08-03, when `sb9x` arrived with an
unbuilt checkout and the operator lifted the rule for that box. It leaves the
tree clean — everything a build writes (`.pixi`, `build/`) is gitignored, no
tracked file is touched — but it is a deliberate exception, not a
reinterpretation. `cloud.md` §1 has the recipe.

## Status

Environment, spine, documentation, **four of the six drivers, both
experiments, and two boxes** — run, measured and published. **~10.8
GPU-hours**: 5.1 on 002 seeds 0-1, ~1.5 characterising sb9x, ~4.2 on seed 2.

```
uv run python -m harness rebuild    --project … --script … --verify
uv run python -m harness supervise  --run … --report-only
uv run python -m harness compare    --dir … --task … --seeds 12
uv run python -m harness capability --policy … --task … --seeds 48
```

`measure` and `feasibility` are specified in
[`harness/DESIGN.md`](harness/DESIGN.md) and deferred: both earn their keep
only before a *new* dispatch, and nothing here has dispatched anything but a
CPU pendulum.

**Experiment 000 — the loop closes.** All ten links, end to end on this box,
in 62 seconds: script → MJCF → bundle → trainer → `.cxpolicy` → the engine
re-verifying its own witness in float64 → rollout → a graph node. The
training half runs on CPU, because there is no trainer-side CPU guard and
`JAX_PLATFORMS=cpu` is the whole of it.

**Experiment 001 — both questions answered, on eight existing runs.**

* **A.** The best checkpoint of `stand-task-20260802-200109` is iteration
  1699 at 7/12 survival; the trainer's reward peak manages 2/12. Survival
  against the trainer's scalar is r = +0.06 over the run and −0.34 after its
  peak — **no reward-based stopping rule would have found it.**
* **B.** The task is **not** out of range. Every policy stands 48/48 unshoved
  and ~50 % survives the declared 0.3–0.8 N. What is out of range is the
  torque: with *nothing pushing at all*, the later policies hold a motor at
  **71 % of its 86 N·mm rating**. The bracing is the resting posture.

**Experiment 002 — three fresh seeds, and the headline is 2 of 3.** Seeds 0
and 1 beat the reward peak by 41.7 and 52.1 pp against a 20.4 pp bound. Seed
2 **ties at +2.1 pp**, its survival flat from iteration 199 on and its
post-peak survival-vs-reward correlation **+0.50** where the others measured
−0.71 and −0.83. What survives all three is the narrower and more useful
claim — **the trainer's scalar is not a proxy for what matters**: in seed 2
it fell 43 % while survival held and episode length rose to the longest of
the run. What does not survive is "survival keeps improving with training".
**Hazard 15 replicates 3 of 3** (86.6 %, 63.3 %, 87.0 % of rating with
nothing pushing).

Flywheel root `rapid-bar-6214`, now ten nodes:

```
cdx-rl: reinforcement learning in Cadex          rapid-bar-6214
├── Thesis and scope                             blue-wave-6018
├── sb1x environment and topology                black-cell-1407
│   └── sb9x: a second box characterised         winter-mouse-1809
├── stand-task-20260802-200109: reward vs length restless-mode-0384
│   └── 001 Phase A: best is iteration 1699      bold-violet-5086
│       ├── 001 Phase B: the task is in range    mute-shadow-9769
│       └── 002: the peak is not best, 2 of 3    holy-recipe-7414
│           └── 002 seed 2: does NOT replicate    spring-unit-9051
└── 000: the loop closes, on CPU in 62 s         calm-bird-4796
```
