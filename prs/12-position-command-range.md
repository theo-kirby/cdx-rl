# PR: A policy's command range, separate from the joint's travel (ADR-123)

- **branch**: `cdxrl/position-command-range` (pushed to `origin`)
- **base**: `main`
- **commit**: `57be248a`
- **OPEN**: https://github.com/theo-kirby/cadex/pull/1

---

`assembly.actuator(..., command_limits_degrees=[-25, 25])` — and `command_limits_mm` for a joint that slides. Optional, **position actuators only**, both endpoints required, **refused if it reaches outside the joint's own declared travel**. It narrows the exported action range and nothing else.

## Why

A position servo's action range is its joint's limits, full stop (`_ACTION_SOURCES`, `_action_bound`). That rationale — *a setpoint outside a joint's travel is a command the joint cannot obey* — is right, and survives here unchanged as a **ceiling**. What it did not allow for is the other direction: **a joint may legitimately move further than any controller should *ask* it to.** Those are two different statements about two different things, and the surface had one spelling for both.

## The evidence

A downstream RL project needed exactly this, could not say it, and **edited the derived task bundle by hand** — capping the action table with a script while copying the MJCF through unchanged. It worked (the MJCF sets `ctrllimited="false"`, so nothing re-clamps downstream), and the cost was that the artifact defining its best policy sat *downstream of the script that is supposed to be the source of truth*.

What the cap bought, measured on a ten-joint biped: the fraction of time the worst servo sat above 90 % of its torque rating fell from **51.8 % to 13.5 %**, for no measurable task cost (15/24 against a control's 18/24, McNemar p = 0.375). That is the difference between a policy describing a machine somebody can build and one that cooks its servos.

## Design notes worth reviewing

- **Each endpoint clamps independently.** The hand-rolled version took `abs()` of both bounds and wrote a symmetric ±cap — a no-op on a symmetric joint, wrong on an asymmetric one. `[-30, 45]` on a ±95° joint means `[-30, 45]`. The test uses unequal endpoints deliberately: a symmetric fixture passes even when an implementation collapses both bounds to one magnitude.
- **`source` says where the range came from.** `actions[].source` becomes `command_limits_degrees`. A bundle reporting the joint's limits as the origin of a number they did not produce is the quiet misattribution this codebase refuses elsewhere. It is deliberately **not** in `_POLICY_ACTION_FIELDS`, so `verify_policy` is unaffected — two bundles differing only in `source` verify the same policy.
- **Both endpoints required.** A one-sided *joint* limit means "free in that direction" and gets a hundred turns' margin; a one-sided *command* limit would have to mean "and the joint's own endpoint for the other side", which is a second, quieter spelling of something already sayable. One meaning per spelling.
- **A narrowed policy will not verify against an unclamped bundle.** `_POLICY_ACTION_FIELDS` compares `low`/`high` exactly, so a policy trained under a narrowed range carries the narrowed numbers. Correct — it is a different action space — but surprising, so it is called out in `docs/XSCRIPT.md`.

## It does not touch the model

Asserted, not assumed. The regression reloads the exported MJCF and checks the joint kept its full travel — in degrees with an **absolute** tolerance, because MuJoCo stores a joint range in float32 and a 95° limit reloads as 94.99984; a relative tolerance there is a float32 detector, not a leak detector.

End to end on the biped: narrowing all ten actuators produced a **byte-identical MJCF**, digest and all. Against the hand-edited bundle it replaces, every action-range number matches on all ten actuators and the only differing field is `source`.

## Verification

`pixi run test-engine` → **1514 passed, 22 skipped**. Seven new tests: the narrowed range with asymmetric endpoints, the untouched joint range, the out-of-travel refusal with both numbers in it, both accepted spellings, the wrong-unit refusal, the not-a-position-servo refusal, and the half-stated / zero-width / inverted refusals.

`test_the_actuator_surface_did_not_change` is renamed `test_the_actuator_surface_is_exactly_this` and updated. It is a guard on the surface and this is a deliberate change to it — flagging it explicitly rather than letting it look incidental.

### Two pre-existing failures, unrelated to this change

`test_dynamics_collision.py::test_a_real_concave_part_is_refused_with_the_numbers_in_it` and `::test_the_same_bracket_is_accepted_when_the_script_says_hull` fail identically on `main` on this box:

```
…/mujoco/plugin/libelasticity.so: failed to map segment from shared object
```

`_live_dynamics` opens its project with no budgets, so the worker runs at the default **6144 MB `RLIMIT_AS`** and MuJoCo's plugin cannot `dlopen`. Patching that one call to pass `memory_limit_mb: 32768` makes both pass. Worth a separate issue — the same cap also makes a real assembly take ~500× longer (1787 s of mostly *system* time against 8.2 s, with 218 MB RSS and `memory_exceeded: false`).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
