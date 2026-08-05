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
| [`mechanisms/mg-legs/`](mechanisms/mg-legs/) | the biped's authoring script, and its own drivers |
| [`cadex-wishlist.md`](cadex-wishlist.md) | thirteen things wanted from Cadex, and what each one cost |
| [`cadex-engine-plan.md`](cadex-engine-plan.md) | the three that block research, scoped as PR specs |

## Ground rules

**Never push to `theo-kirby/cadex`.** Cadex work happens in the PR clone at
`/home/theo/cadex-prs` and reaches Cadex as **pull requests**, reviewed
externally. `/home/theo/cadex` is the operator's working tree: leave it alone
— no commits, no edits, no branch changes, no builds *there*.

`/home/theo/cadex-train-venv` is referenced and never rebuilt — its pins are
what make hazard 15 a comparable measurement across every recorded run.
`/home/theo/cadex-jobs` is a read-only input.

Until 2026-08-05 this repository was read-only toward Cadex altogether and
captured what it wanted in `cadex-wishlist.md`. Thirteen entries accumulated
and three became the binding constraint on the research, ahead of the GPU, so
the operator changed the policy. The wishlist stays as the record of what
each gap cost.

## Status

Environment, spine, documentation, **five drivers, three experiments, three
boxes, and the biped's authoring script**. **~14.2 GPU-hours**: 5.1 on 002
seeds 0-1, ~1.5 characterising sb9x, ~4.2 on 002 seed 2, ~3.4 on 003.

```
uv run python -m harness rebuild    --project … --script … --verify
uv run python -m harness supervise  --run … --report-only
uv run python -m harness compare    --dir … --task … --seeds 12
uv run python -m harness capability --policy … --task … --seeds 48
uv run python -m harness steps      --dir … --task … --profile mg-legs --seeds 12
```

`measure` and `feasibility` are specified in
[`harness/DESIGN.md`](harness/DESIGN.md) and still deferred **as harness
drivers** — but working, mechanism-specific ones now exist at
[`mechanisms/mg-legs/drivers/`](mechanisms/mg-legs/drivers/), and they are
what gated experiment 003. Porting them behind `harness/profiles/` the way
`steps` is done is the next harness job.

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

**Experiment 003 — the action space was the problem.** Nine `mg-legs` runs
moved the disturbance and the reward and never the action space, the control
rate or the reward's sign. Moving all three:

* **17/24 on the conjunction** — stepped ≥10 mm *and* survived — against
  B6's **6/12** on the same criterion and the same task. And `survived` and
  `stepped-and-survived` are now **the same number**: every episode it
  survives, it survives by stepping.
* **Hazard 15 dissolves.** 002 replicated the bracing 3 of 3 at 63–87 % of
  rating with nothing pushing. The same mechanism under position servos holds
  its stance at **31 % peak, 15.6 % static**. The bracing was an artefact of
  a torque action space, not of the mechanism and not of the reward — which
  is why hazard 16's "a reward term cannot fix this" was true and unhelpful.
* **The untrained policy stands**, so PPO no longer has to discover gravity
  compensation for ten joints before it can learn balance.
* One seed. 002's lesson applies before anything is built on it.

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

Flywheel root `rapid-bar-6214`, now twelve nodes:

```
cdx-rl: reinforcement learning in Cadex          rapid-bar-6214
├── Thesis and scope                             blue-wave-6018
├── sb1x environment and topology                black-cell-1407
│   └── sb9x: a second box characterised         winter-mouse-1809
├── stand-task-20260802-200109: reward vs length restless-mode-0384
│   └── 001 Phase A: best is iteration 1699      bold-violet-5086
│       ├── 001 Phase B: bracing is the resting posture  mute-shadow-9769
│       │   └── 003: the action space was the problem    broad-fire-8531
│       └── 002: the peak is not best, 2 of 3    holy-recipe-7414
│           ├── 002 seed 2: does NOT replicate   spring-unit-9051
│           └── 003 (second parent)              broad-fire-8531
├── mg-legs before cdx-rl: nine runs             rapid-voice-5955
└── 000: the loop closes, on CPU in 62 s         calm-bird-4796
```

003 hangs off **both** nodes it answers — the one that found the bracing and
the one that replicated it 3 of 3.
