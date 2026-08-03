# mg-legs' own drivers — imported, mechanism-specific, **not** the harness

These eight scripts gated and scored experiment 003. They are committed
because they are working instruments and because two of them are the
`feasibility` and `measure` that [`harness/DESIGN.md`](../../../harness/DESIGN.md)
§2–3 specify and defers.

**They are not `harness` drivers and must not be mistaken for them.** They
hardcode `mg-legs`: joint name lists, per-joint moment arms, geom names,
the support polygon, the paths of one working directory. Porting them behind
`harness/profiles/` the way `steps` is done is the next harness job; until
then this is the honest place for them — an imported tool that works on one
mechanism, rather than a generic driver that has only ever been run on one.

They also expect the **laptop's** layout (`CADEX_REPO`, a `mg-legs.cadex`
project beside them) rather than cdx-rl's `config/env`. Read the header of
each before running it.

| | |
|---|---|
| `feasibility.py` | the six-check gate. **Carries the action-space branch** — see below |
| `measure.py` | mass, standing CoM, joint heights, and the reward's measured constants |
| `swirl_scale.py` | measures the σ of every shaped reward quantity off the bundle |
| `reward_standing.py` | the positive-kernel hazard-9 check: every term must pay its own weight standing |
| `reward_decompose.py` | per-term reward over the states a policy actually visits |
| `rebuild.py` | drives `cadexd` over NDJSON to accept a script |
| `dispatch_b8.sh` | the run, with its hyperparameters and its pre-flight checks |
| `install_checkpoint.py` | copy a policy into a project and rewrite the `assembly.policy` call |

## What is worth reading before porting

**`feasibility.py` checks 5 and 6 are the re-specification `method.md` §7
describes**, and they are the reason this directory exists rather than a
note. Check 5 branches on the compiled model's actuator kind: under torque
the zero action must fall, under position it must *stand* and non-degeneracy
is measured by running the zero action against the declared task
(`degenerate()`). Check 6 stops sweeping gains when the PD is in the model
and reports the servo's own settle, peak effort and saturation angle instead.

**`swirl_scale.py` measures four scales off one sweep** — angular momentum,
centre-of-mass speed, joint deviation, actuator force — over the *recovery
regime* only, which is the force levels at which the held stance actually
goes over. Folding in the absorbable end of the band halves every scale,
because most of those samples are a machine standing still. That is B6's
saturated-`capture` mistake generalised, and avoiding it is worth the minute
it costs.

**`reward_standing.py` is small and is the check to keep.** It is 120 lines
and it caught nothing on 003 only because 003 was built to satisfy it; on any
future reward it is the cheapest thing between a stale measured constant and
a wasted run.

## Two traps these scripts encode

* **`rebuild.py` restores the pose parameters to the script's declared
  defaults, not to zero.** Through B7 the nominal pose *was* all zeros so the
  two were the same instruction; 003's nominal pose is a crouch declared in
  those defaults, and a driver that zeroed them would silently straighten the
  legs and train a different robot than every measurement was taken on.
* **A refused `write_script` reverts `<project>/script.py`** to the last
  accepted source, destroying the edit that caused the refusal — and the next
  rebuild then reports the *previous* revision hash, which reads exactly like
  success. Copy the file before every rebuild and restore it on failure.
