# PR: an inertial coordinate below a nanometre is zero (ADR-133)

- **branch**: `cdxrl/inertial-snap` (pushed to `origin`)
- **base**: `main`
- **commit**: `1c354580`
- **OPEN**: https://github.com/theo-kirby/cadex/pull/3
- closes `cadex-wishlist.md` **#15** *(wishlist #15, GitHub #3 — mind the two numbering schemes)*

---

`CadexDynamics.body_inertial` snaps every component of the centre of mass it returns to exactly `0.0` when its magnitude is below one nanometre. The rule is **absolute**, applies to coordinates and nothing else, and drops the sign with the magnitude so a symmetry zero is never written as `-0`.

## What it costs, measured

A trained policy cannot be replayed on the machine that did not author its mechanism, and the whole difference is one float.

`mg-legs` is a symmetric biped, so its pelvis centre of mass is zero in x by construction. OCCT does not read it as zero, and does not read it as the same non-zero on two platforms:

| | pelvis `<inertial pos>` x | MJCF sha256 | task bundle sha256 |
|---|---|---|---|
| macOS 26, arm64 | `5.10066e-11` | `80eaa18f6025…` | `5572adf265aa…` |
| Ubuntu 24.04, x86-64 | `5.10087e-11` | `0fe04cfce228…` | `0b4d160cd436…` |

**That line is the only difference between the two 14 179-byte files**, verified with `cmp`; and the only difference between the two bundles is the model digest the bundle embeds. But `verify_policy` check 1 is a whole-file hash, so:

```
policy output 'balance' was trained on a task bundle whose digest is
'5572adf265aa51cb…', and the task it is declared against digests to
'0b4d160cd436fd16…'
```

The refusal is correct in principle — a policy is only meaningful for the task it was trained on — and wrong in this instance. 2.1e-15 m on a coordinate that is zero by symmetry is not a different robot.

## Wanted

Absolute, not relative, and a physical number rather than a float epsilon.

The two readings differ in their **fifth significant figure**, so no relative tolerance calls them equal. The reason is cancellation: a symmetric body's x-centroid is a difference of near-equal sums, so a last-bit disagreement in OCCT's per-solid readings arrives amplified by eleven orders of magnitude. `math.fsum` is correctly-rounded, so identical inputs give identical output and the residual can only have come from the inputs — which rules out fixing it with a summation order or a compensated sum too. A tolerance is the only thing that can work.

A nanometre comes from the machine shop, not the arithmetic: four orders below the tightest tolerance anything in `mg-legs` is modelled to, three below the chord tolerance a collision mesh is tessellated at. Nothing that survives this snap was ever a feature.

Applied inside `body_inertial`, after the weighted sum and **before** the parallel-axis loop. Two publications read that one number — the MJCF's `<inertial pos>` and the `dynamics` summary's `center_of_mass_mm` — so snapping at the publication sites would be two chances to disagree.

## Measured after, on both boxes

Same policy-free `mg-legs` script — byte-identical input, sha256 `c37cabeb6425b08e…` — built through `cadexd` on each machine against byte-identical engine sources (`c4cc1215edfc…`):

| | script build digest | MJCF | task bundle |
|---|---|---|---|
| macOS 26, arm64 | `560a33a4bfce810e…` | `203f746e9bb8a857…`, 14 169 B | `6dc1c580f4bcd01a…`, 30 213 B |
| Ubuntu 24.04, x86-64 | `560a33a4bfce810e…` | `203f746e9bb8a857…`, 14 169 B | `6dc1c580f4bcd01a…`, 30 213 B |

`cmp` reports both pairs identical. The file is ten bytes shorter, which is `5.10087e-11` becoming `0`.

## What it deliberately does not do

**It does not snap mass, and it does not snap the inertia tensor.** A product of inertia that is zero by symmetry has the same cancellation problem in principle, and a nanometre is not a tolerance for kg·m² — the analogous bound would be relative to the body's own moments, which is a different decision with a different justification. For this mechanism it does not arise: both platforms print identical `quat` and `diaginertia`. The boundary is named in the ADR so a mechanism that does hit it is recognised rather than rediscovered.

## What landing it costs downstream

**It moves every model digest**, and that is the point rather than a side effect: any fix that makes two platforms agree must change at least one of them. The alternative — declaring digests platform-specific forever — is what makes a cross-machine pipeline impossible.

Every policy trained against a pre-snap bundle is therefore refused against a freshly built one. **[#4](https://github.com/theo-kirby/cadex/pull/4) (ADR-134) is the compatibility path** and should land with or after this one; it is measured against exactly these files.

## Tests

`test_dynamics_inertial_snap.py`, ten tests, of which **seven fail without the change**. They carry the two measured platform readings as constants, assert the snap is invisible to a relative tolerance, assert a 1 µm coordinate survives bit for bit, assert the tensor cost is 9e-23 against moments of 1e-5, and follow the chain out to the exported MJCF text.

Full engine suite **1622 passed / 5 failed / 22 skipped** against a pre-change baseline of **1612 / 5 / 22** on the same commit: +10 new tests, the same five failures. Two of those five are the `RLIMIT_AS` collision defect; the other three arrived in `test_part_blending.py` and `test_part_organic.py` with the merge that renumbered ADR-131 and ADR-132. Neither set is touched here.
