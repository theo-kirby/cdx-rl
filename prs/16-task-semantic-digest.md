# PR: a policy may carry the bundle it trained on (ADR-134)

- **branch**: `cdxrl/task-semantic-digest` (pushed to `origin`)
- **base**: `cdxrl/inertial-snap`, then `main`
- **commit**: `4129cbf3`
- **land after** [#3](https://github.com/theo-kirby/cadex/pull/3) — this is its compatibility path
- closes `cadex-wishlist.md` **#16** *(wishlist #16, GitHub #4)*

---

`assembly.policy(..., trained_task="<bundle>.json")` names, as a project asset, the task bundle a policy was **actually trained on**. Given it, `verify_policy` is pointed at *that* bundle — the same whole-file digest check as ever, unweakened — and the bundle this script just built must then be proved **equivalent** to it: every field that decides behaviour, plus the two models compared as models. Omit the keyword and nothing changes.

## What it costs, measured

Two of this engine's own corrections orphan trained policies, and neither is a mechanism change.

**ADR-133** (#3) snaps inertial coordinates below a nanometre. That moves every model digest, so every bundle embedding one, so every policy trained before it.

**ADR-131** made a ±25° command range on a ±45° joint sayable in a script and reports its provenance honestly as `command_limits_degrees`. The arm it replaces was produced by editing the derived bundle by hand, which reported `angle_limits_degrees` — the joint's limits, which are not where ±25 came from. All ten actuators, `low`, `high`, `unit` and `scale` are identical between the two bundles; `label` and `actions[].source` are not, and they move the whole-file hash:

```
trained on        3d627ef4b9a509fe…
declared against  bd8071b50360eaab…
```

Before this, the remedies were **retrain** — four to five GPU-hours a seed — or **revert the correction**, which buys one replay and reintroduces a bundle that misreports where its numbers came from.

## Wanted, and why not the obvious version

The obvious alternative is to make `verify_policy`'s task check semantic. That weakens every policy in the system to buy compatibility for a few.

This does the opposite. The policy is still bound to **one exact bundle** by whole-file digest; what is new is a *proof that a second bundle is the same task*. `test_verify_policy_was_not_weakened` asserts the first check is still a string comparison against a whole-file hash and that `task_semantic_digest` appears nowhere inside it.

It also needs **no change to `training/cadex_train.py`** — which matters, because that file's digest is pinned in every cdx-rl run record and moving it costs a bridge run. `cadex_train.py` writes `task.sha256` as a whole-file hash and keeps doing so.

## What "the same task" means

`TASK_SEMANTIC_FIELDS` is written out rather than derived as "everything except the exclusions", so a schema that grows a field is a decision rather than a silent widening — the discipline `_MJCF_MODEL_FIELDS` already keeps. Excluded, each for a stated reason:

| excluded | why |
|---|---|
| `label` | the script's name for the task. `stand12` by hand, `stand` from source |
| `model` | a path, a byte count and a digest. The model itself is **not** excluded |
| `actions[].source` | how the bundle *derived* a bound. ADR-131 |
| `actions[].fallback` | what to write with no policy, which a trained policy never reads |

`mujoco_version` is *in*, and it is the one that looks like metadata and is not: MuJoCo disclaims cross-version numerical reproducibility outright, so two bundles differing only there describe two different dynamics.

Numbers canonicalise through `repr(float(x))` — the shortest decimal that round-trips a double, identical on every platform for the same double. Two normalisations: `30` and `30.0` agree, and `-0.0` is `0.0`. Strings stay quoted and numbers bare, so the string `"30.0"` and the number `30.0` do not digest alike.

## The half that keeps it honest

**A bundle comparison that dropped `model.sha256` and stopped there would be a hole.** Two bundles can agree on every number while naming different mechanisms — same joint names, same limits, different masses — and the action table would match perfectly. So `model_differences` compiles both MJCF files and diffs them on `_MJCF_COUNT_FIELDS` exactly and on `_MJCF_MODEL_FIELDS` plus `_MJCF_OPTION_FIELDS` at `MJCF_FIELD_TOLERANCE`: the same lists and the same bound `export_mjcf` already holds its own round trip to.

The trained model is found in `assets/` **by the digest the trained bundle records**, not by name — the bundle's own `model.path` points inside the attempt directory that produced it and means nothing in a new project.

**One field needs an absolute floor, and only one.** `_field_drift` divides by the field's own largest magnitude, so a model whose only inertial offset is symmetry noise has a `body_ipos` whose entire scale is 5e-11, and `5.10087e-11` against `0` reads as **1.0 relative drift** — total disagreement about two numbers that are both zero. That is measured: the first version of this comparison did exactly that and failed its own test. So `body_ipos` also passes if the worst *absolute* difference is under a nanometre, which is the most ADR-133's snap can move it and the only field it touches.

The floor is deliberately **not** blanket. 1e-9 is negligible against a mass in kg, and it is *looser* than the relative bound for an inertia tensor: a sub-kilogram limb's moments are around 1e-5 kg·m², so a 1e-9 floor would admit 1e-4 relative where the field bound admits 1e-5.

## Measured, on the files this was written for

`tasks/stand-b8/stand-task.json` — the macOS-authored bundle `stand10.001700.cxpolicy` was trained on — against a freshly built post-snap bundle on Linux:

| | whole-file sha256 | semantic sha256 | `task_differences` |
|---|---|---|---|
| trained on (macOS, pre-snap) | `5572adf265aa…` | `6bb66e9bcafaf856` | — |
| script-built (Linux, post-snap) | `6dc1c580f4bc…` | `6bb66e9bcafaf856` | *empty* |

And the three MJCF variants pairwise through `model_differences`: macOS pre-snap (`80eaa18f`, 14 179 B), Linux pre-snap (`0fe04cfc`, 14 179 B) and post-snap (`203f746e`, 14 169 B) are **all mutually equivalent**. Two whole-file digests disagree; three models are one machine.

## What landing it costs downstream

Nothing already written changes behaviour: `trained_task` defaults to `""` and every pre-existing script takes the old path. Three new refusal stages appear in the failure envelope — `policy_trained_task`, `policy_task_equivalence`, `policy_model_equivalence` — and a receipt gains an optional `equivalence` block whose **absence** is also a record: it says the policy was checked whole-file against the bundle the script built.

One existing test moved: `test_the_surface_takes_a_task_and_two_required_keywords` pins the `policy` signature, which legitimately gained an optional keyword. It now also asserts `trained_task` stays optional, because every script written before this makes the older claim.

## Tests

`test_dynamics_task_identity.py`, 52 tests. **Nineteen are one behaviour-deciding field each**, changed one at a time and required to move the digest *and* be named in the diff — a comparison that missed a field would be one that quietly accepted a different task. The rest pin the provenance exclusions, the float canonicalisation, the bounded diff, the absolute floor's scope and the arithmetic that rules a blanket floor out. Nine more in `test_dynamics_policy_api.py` cover the new keyword's validation, including that naming the policy as its own training bundle is refused.

Full engine suite **1683 passed / 5 failed / 22 skipped** against **1612 / 5 / 22** on `main`: +71 tests, the same five failures.
