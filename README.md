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
branch changes, no builds. `/home/theo/cadex-train-venv` is referenced and
never rebuilt. `/home/theo/cadex-jobs` is a read-only input.

## Status

Environment, spine and documentation are in place and the smoke test passes
end to end. The five drivers and both experiments are specified and not yet
built. Flywheel root: `rapid-bar-6214`.
