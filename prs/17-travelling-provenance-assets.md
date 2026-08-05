# PR: the project store holds what a policy travels with (ADR-135)

- **branch**: `cdxrl/travelling-provenance-assets` (pushed to `origin`)
- **base**: `main`
- **commit**: `ad66e9a2`
- **OPEN**: https://github.com/theo-kirby/cadex/pull/5
- **completes** [#4](https://github.com/theo-kirby/cadex/pull/4) — ADR-134 does not work without it

---

`_PROVENANCE_ASSET_SUFFIXES = {".json", ".xml"}` joins the union `put_asset` accepts. A `.json` task bundle and a `.xml` MJCF are what `assembly.policy(..., trained_task=)` binds a policy to and compares against, and until now the store would not hold either.

## What it cost, and how it was found

**ADR-134 shipped unusable, and all 52 of its unit tests passed.** The first end-to-end replay refused at the very first step:

```
ASSET_REJECTED at precondition
'clamp25-task.json' is not one of the formats this project store holds
['.cxpolicy', '.obj', '.ply', '.stl']
```

Every ADR-134 test exercised `task_semantic_digest`, `task_differences`, `model_differences` or the API's argument validation. Not one went through `store_project_asset`, so nothing noticed that the two files the whole feature depends on could not reach the directory the worker reads them from.

That is the lesson `MUJOCO.md` already states about training runs, transferring exactly: **validate at length, not at three iterations.** A surface whose unit tests all pass and whose first real use fails at step one was tested at three iterations.

## Wanted

A third constant, not two more members.

`_ASSET_SUFFIXES` **must stay exactly three**: the shell mirrors it by name at `cadex_backend.py:53`, and every line of the `shell/` diff is a future merge conflict against upstream Blender (ADR-091). `_POLICY_ASSET_SUFFIXES` already exists as a separate constant for that reason (ADR-084), and this is the same move a second time — three questions with three answers.

**The generic suffixes are not a hazard.** `.json` and `.xml` are the only generic extensions the store holds, and nothing is interpreted on arrival: a `.json` is read only when a script names it as `trained_task`, a `.xml` only when its digest matches the one that bundle records. An asset nothing names is bytes in a directory. The staging budget is unchanged at 64 files / 128 MB; a replay set is four files and 380 kB.

## Measured after

Both `mg-legs` arms replay from source on Linux, against bundles built elsewhere:

| arm | trained on | script built | same task (semantic) | verdict |
|---|---|---|---|---|
| `b8` | `5572adf265aa…` *(macOS, pre-snap)* | `6dc1c580f4bc…` | `6bb66e9bcafa…` | **accepted** |
| `clamp25` | `3d627ef4b9a5…` *(hand-edited, ADR-131's predecessor)* | `3dbc680589b1…` | `17f1f46fbfcf…` | **accepted** |

`clamp25` is the one worth reading twice: that policy could not be replayed by any script before this, and the fix required neither retraining nor reverting ADR-131's honest `source` string.

And the refusals still refuse. Five mutations of the **script** — not of the travelling bundle, which fails `verify_policy` check 1 first and proves nothing — each refused, each naming the field:

| mutation | stage | what the refusal said |
|---|---|---|
| reward weight 0.2 → 0.9 | `policy_task_equivalence` | `reward[0].weight: 0.9 here, 0.2 there` |
| episode 6.0 s → 9.0 s | `policy_task_equivalence` | `episode.episode_seconds: 9.0 here, 6.0 there` |
| tip threshold 0.15 → 0.25 | `policy_task_equivalence` | `termination[0].above: 0.25 here, 0.15 there` |
| command range ±25° → ±30° | `policy_task_equivalence` | `actions[0].high: 30.0 here, 25.0 there` |
| bracket plate 2.5 → 2.9 mm | `policy_model_equivalence` | `body_ipos: 0.0363 relative drift (0.000727 absolute)` |

The last justifies ADR-134's model comparison existing. A 0.4 mm plate changes masses and inertias and changes **no field of the task bundle at all** — same joints, same limits, same action table, same observations. A bundle-only check would have accepted a policy against a different machine and said nothing.

## A note on the suite baseline

**The two `test_dynamics_collision.py` failures are intermittent, and this PR is where that got measured.** Three runs of that file alone, no code change between them: **2, 1, 2** failures. So `docs`-level statements of the form "the baseline is exactly 2 failures" are slightly wrong — the `RLIMIT_AS` defect is resource-dependent, not deterministic, and a suite comparison should read that file as 1–2.

This run: **1685 passed / 4 failed / 22 skipped**, against `main`'s 1612 / 5 / 22 before ADR-133 and ADR-134 — and across three full-suite runs on this branch, 5, 4 and 5 failures, which is the intermittency above rather than anything moving. The four are one collision test (the intermittent pair, one of which passed), `test_part_blending.py` and two in `test_part_organic.py` — the three that arrived with the merge renumbering ADR-131 and ADR-132. None is touched here.

## Tests

`test_the_store_holds_what_a_policy_travels_with` stores a `.json` and a `.xml` through `store_project_asset` and reads both back, and pins the three constants apart — including that `_ASSET_SUFFIXES` is still exactly `{.stl, .obj, .ply}`, which is the one the shell mirrors. The existing rejection test's comment was updated to name three kinds rather than two; a `.txt` is still refused.
