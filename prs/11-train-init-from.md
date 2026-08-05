# PR: `--init-from` — a warm start, and deliberately not a resume (ADR-124)

- **branch**: `cdxrl/train-init-from` (pushed to `origin`)
- **base**: `main`
- **commit**: `6c4dca31`
- **OPEN**: https://github.com/theo-kirby/cadex/pull/2
- **land after** [#1](https://github.com/theo-kirby/cadex/pull/1) — this one moves the trainer digest.

---

`cadex_train.py --init-from <policy.cxpolicy>` starts the actor from an existing policy's weights instead of a fresh network, restores the observation normaliser beside them, and leaves **the critic and the optimiser fresh**. Refused unless the policy matches the bundle's task and model digests, its observation channels in order, its action table on `_POLICY_ACTION_FIELDS`, and the network shape `--hidden` asks for.

## What it costs not to have it

Every run started from zero. Experiment 003's three seeds all peaked at iteration **1700–1750 of 1800** — the last two checkpoints written — so the obvious next question was where improvement actually stops, and answering it meant re-running from zero at greater length: **~6.9 h to reach 700 new iterations, with 5 h of it recomputing a curve already on disk.** One such arm was dispatched and cancelled thirty minutes in once that was written down.

Measured on a ten-joint biped, three iterations each, same bundle and same seed:

| | cold | `--init-from` iteration 1750 |
|---|---|---|
| iteration 0, reward/step | +3.27 | **+4.32** |
| iteration 0, episode steps | 88.2 | **598.0** |
| iteration 2, reward/step | +3.11 *(falling)* | **+4.68** *(rising)* |
| iteration 2, episode steps | 48.6 | **542.5** |

The episode length is the one to read: the warm-started policy is standing, the cold one is falling over. The warm policy's witness agrees to 1.6e-07, 627× inside tolerance — so the restored weights round-trip correctly.

## Scope — bigger than "one flag", and here is where

- **A warm start, not a resume.** Only the actor is in a `.cxpolicy` (`snapshot()` records what the engine can play; the critic is training scaffolding). A resume would need the optimiser moments, which the container does not carry.
- **Leaving the optimiser fresh is free**, not a compromise: Adam is hand-rolled and its moments are `zeros_like(params)`, correctly-shaped zeros whether the actor is swapped in before or after they are taken.
- **A fresh critic has a cost worth stating**: a trained actor with a random critic produces large early advantages, which interacts with `--clip`. At the default clip the first iterations improve rather than diverge; a warm start into an unusually large `--clip` is the case to watch.
- **The normaliser travels with the weights**, or the transfer is mostly wasted — the actor reads *normalised* observations and a fresh normaliser feeds it raw ones. `seen` is not in the container and restarts at `1.0e-4`, so restored statistics are re-estimated quickly rather than frozen.
- **Provenance is a digest, never a path.** `policy_header` folds every option into `hyperparameters`, so left alone this flag would stamp one machine's filesystem layout into every policy. `init_from` is excluded there and recorded under `training.init_from` as the source's `sha256`, label, iterations and `trainer_sha256`.
- **A fourth implementation of the container.** The trainer may not import `CadexDynamics` (`test_dynamics_policy_trainer` asserts it appears only as a deferred, caught import), so the format is now written twice and read twice. Mitigated by `test_the_trainers_decoder_agrees_with_the_engines`, beside the two encoder-agreement tests it mirrors.

## It changes `cadex_train.py`'s sha256

Unavoidably — which invalidates every recorded trainer pin plus `remote_train.sh:199`'s own check. Flagging it rather than letting it be discovered.

**The flag is a no-op when unused**, verified more strongly than the existing regression asks. Two runs of the modified trainer and one run of the *unmodified* trainer at the same seed produce the **same** digest over `observations`, `network`, `normaliser`, `evaluation`:

```
modified run a  : 32444f5e8518c39f…
modified run b  : 32444f5e8518c39f…
BASELINE (main) : 32444f5e8518c39f…
```

Comparisons that must cross the digest boundary should pay for a **bridge run** — one seed of an existing arm retrained under the new trainer and scored against where the old one landed — rather than treating the boundary as uncrossable.

## Verification

`pixi run test-engine` → **1511 passed, 22 skipped**. Four new tests: decoder agreement with the engine, decoder refusals (bad magic, no header length, truncated, partial float32), the `_POLICY_ACTION_FIELDS` table, and the flatten/unflatten round trip that pins the weight layout — `(inputs, outputs)` row-major then bias, which transposed would produce a network of exactly the right size computing something else.

Both refusal paths exercised end to end:

```
$ … --init-from <clamped policy>   # against the unclamped bundle
--init-from: the task digest does not match this bundle.
  bundle: 5572adf265aa51cb…
  policy: 3d627ef4b9a509fe…

$ … --hidden 32 32 --init-from <64x64 policy>
--init-from: the network shape (check --hidden) does not match this bundle.
  bundle: [[58, 32], [32, 32], [32, 10]]
  policy: [[58, 64], [64, 64], [64, 10]]
```

### Two pre-existing failures, unrelated

`test_dynamics_collision.py::test_a_real_concave_part_is_refused_with_the_numbers_in_it` and `::test_the_same_bracket_is_accepted_when_the_script_says_hull` fail identically on `main` on this box — MuJoCo's `libelasticity.so` fails to map under the worker's default 6144 MB `RLIMIT_AS`. See the note on the sibling PR.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
