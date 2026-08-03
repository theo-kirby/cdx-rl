# mechanisms/mg-legs — the biped, and its authoring script

**This directory closes the gap that `CLAUDE.md` named as blocking every task
change.** Until 2026-08-03 that file said:

> *The biped's authoring script does not exist anywhere on this box.
> `~/cdx-mjc/`, which `MUJOCO.md` §7 and ADR-100 both name, is gone; searched
> for. `model_sha256 e3511559…` is therefore **not reproducible**, and every
> number in 001 and 002 is a claim about the exact bytes now committed at
> `tasks/stand-b2/`. Locating or re-authoring that script is the first step of
> any task change.*

`~/cdx-mjc/` was not gone. It was on the **macOS laptop**, which is a
different machine from either training box, and it is where every `mg-legs`
run from M9 through B8 was authored and dispatched. `script.py` here is that
script, at the revision that produced the B8 result.

So `mg-legs` is now reproducible from source: change a parameter, rebuild,
and the MJCF and the task bundle come out of OCCT rather than out of a
committed byte-blob nobody can regenerate.

---

## What is here

| | |
|---|---|
| `script.py` | the authoring script, accepted revision `feb5a884…`, sha256 `56bba536…` |
| `history/0018-a3a7bd262040.py` | the **B6** revision — the 6/12 baseline experiment 003 is measured against |
| `history/0023-5339f0e37000.py` | the **B7** revision — torque actuators, cost-shaped reward, the last of the old family |

The full accepted history (24 revisions, M9 through B8) is in the project
store at `<project>/script_history/` on the laptop and is not committed; the
two kept here are the two that other documents make claims about.

## Reading it

It is 3 000 lines and roughly two thirds comment, deliberately. The comments
are the record of why each number is what it is, and several of them are the
only surviving statement of a measurement. The parts worth finding:

| what | search for |
|---|---|
| the crouch, and why 15° | `B8: THE NOMINAL POSE IS A CROUCH` |
| why the machine had to be dropped back onto the floor | `THE CROUCH LIFTS THE MACHINE OFF THE FLOOR` |
| position actuators, the measured gains, and the saturation argument | `THESE ARE SERVOS NOW` |
| the measured constants every reward term is built from | `Z0 = ` |
| the nine positive kernels and what each is for | `EVERY TERM IS POSITIVE` |
| the shove band and why it did not move | `B8 PUTS IT BACK TO [0.30, 0.80]` |
| 50 Hz, and why the "nothing rounded" argument survives | `B8: 50 Hz` |

## The mechanism

263 g of PLA and **ten MG90S** (eight through M9c; the two ankle rolls are
B2's), 302 g assembled. Twelve links, ten revolute joints, a welded ball-of-
foot hinge, and a floating pelvis — there is no joint to the floor, which is
the point of the machine.

| | |
|---|---|
| mass | 302.0 g → 2.963 N |
| standing CoM | 140.944 mm (B8's crouch; 144.210 straight-legged) |
| ω₀ = √(g/h) | 8.3428 rad/s |
| support polygon | 45.5 mm forward, 24.5 mm back, ±50 mm lateral |
| actuator limit | **86 N·mm**, and it models the **hardware** |

**86 N·mm is a judgment, not a datasheet number, and the experiment README
must keep saying so** (`method.md` §5). The MG90S *stalls* at 216 N·mm — a
momentary figure no hobby servo holds — and 86 is ~40 % of stall, the range
within which small RC servos are conventionally run for sustained holding.
ADR-083 measured what the 216 N·mm build did with the headroom: it held four
motors at 93–99 % of stall for entire six-second episodes. That is
`MUJOCO.md` hazard 15, and it is the hazard experiment 003 turned out to
dissolve.

## Provenance

Authored on the macOS laptop, not on either training box. Rebuilt through
`~/cdx-mjc/rebuild.py`, which drives `cadexd` over NDJSON exactly as the
shell does — `open_project` then `write_script` — so what it proves is what
the product would do.

**Never hand-edit a `.cadex` project's `script.py`.** The store binds an
accepted digest to that file, so an edit makes every later entry point fail
with *"the restore pass digest does not match the accepted digest"*. And a
**refused** write silently reverts the file to the last accepted source,
destroying the edit that caused the refusal — measured on 2026-08-03, where
a `reset_variation_penetrates` refusal reverted a 40-line change and the next
rebuild reported the *previous* revision hash, which reads exactly like
success. Copy the file before every rebuild and restore it on failure.

## Rebuilding this script

`script.py` as committed has the B8 policy **installed** — it carries a live
`assembly.policy(stand, weights="stand10.cxpolicy", sha256="dc375a42…")` and
the matching `result` rows. So a rebuild needs that asset present:

```bash
cp tasks/stand-b8/stand10.001150.cxpolicy <project>/assets/stand10.cxpolicy
```

Without it the accept fails on a missing asset rather than on anything about
the mechanism. Comment out the `balance` / `play` pair and their two `result`
rows if you want the mechanism without the policy.

**Installing a policy is two edits, not one.** The `assembly.policy(...)`
call *and* the `"balance": balance` / `"stand_play": play` rows in the
`result` dict. With only the first, the project accepts green, the engine
verifies the witness, and the policy produces **no output at all** — no
`balance-policy.json`, no `assembly-simulation-trace.json` — so the shell
animates nothing and it reads as a broken simulation rather than a missing
return value.
