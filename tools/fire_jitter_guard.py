#!/usr/bin/env python
"""Make ``jitter.py`` separate a still machine from a chattering one, on purpose.

``harness/DESIGN.md`` §6: *"every check must be able to fail, and must have
been made to fail once on purpose."* ``jitter`` does not decide an exit code —
it decides whether experiment 006 dispatches a 5 GPU-hour arm, which is worse,
because a driver that reports plausible numbers for everything reads exactly
like one that works.

**Run under the trainer interpreter, never `uv run`:**

```
/home/theo/cadex-train-venv/bin/python tools/fire_jitter_guard.py
```

Three synthetic controllers are played through the same model, the same
bundle and the same seeds as a real policy would be, and the driver's own
statistics are read off them:

* **hold** — the nominal pose, every step. Command jitter must be *exactly*
  zero and the reversal rate exactly zero. Anything else means the driver is
  measuring its own arithmetic.
* **chatter** — the action range's corners, alternating every control step.
  This is the worst case the surface admits, so it must pin the reversal rate
  at the ceiling (``1 / interval_s``) and drive ``Σ|q̇|`` far above **hold**.
* **sweep** — a slow full-amplitude triangle wave. **This is the case that
  matters**, and it is why a magnitude statistic alone is not enough: its mean
  ``|Δ|`` is comparable to a fidgeting policy's while its reversal rate is
  near zero. A driver that called this "jitter" would condemn a machine that
  is tracking smoothly.

Exit 0 if all three separate as declared. The thresholds are written here
before the run, not fitted to it.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
MODULE_DIR = os.environ.get("CADEX_ENGINE_DEV_TREE", "/home/theo/cadex")
sys.path.insert(0, str(Path(MODULE_DIR) / "src" / "Mod" / "cadex"))
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "mechanisms" / "mg-legs" / "drivers"))

import CadexDynamics as cd  # noqa: E402

import jitter  # noqa: E402

TASK = REPO / "tasks" / "stand-b8-clamp25" / "stand-task.json"
SEEDS = [0, 1, 2]


def play(task, model_xml, controller, seeds):
    """The three statistics the guard reads, over `seeds` episodes.

    **No settle window here, unlike the driver**, and the reason is worth
    stating: `chatter` slams every joint to its corner and the machine is down
    inside half a second, so a 1 s settle window would leave it with *no*
    frames and every statistic would come back 0.0 — which reads exactly like
    a controller that does not chatter. The first version of this guard did
    that and reported four failures that were all the same missing window.

    This is a check on the arithmetic, not a measurement of posture, so it
    reads the whole episode and starts from the bare keyframe.
    """

    deltas, reversals, qsum, lengths = [], [], [], []
    interval = float(task["episode"]["control_interval_s"])
    settle = 0
    velocity_scale = next(float(r["scale"]) for r in task["observations"]
                          if str(r.get("kind")) == "velocity")
    addresses = None

    for seed in seeds:
        model = cd.load_model(model_xml)
        mujoco = cd._mujoco_module()
        if addresses is None:
            addresses = [int(model.jnt_dofadr[int(mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, str(a["joint"])))])
                for a in task["actions"]]
        frames = {"a": [], "q": []}

        def sample(_step, data, _final, action):
            frames["a"].append(None if action is None
                               else [float(v) for v in action])
            frames["q"].append([abs(float(data.qvel[i])) * velocity_scale
                                for i in addresses])
            return None

        cd.evaluate_episode(model, task, actions=controller, sample=sample,
                            seed=seed)
        issued = [a for a in frames["a"] if a is not None]
        lengths.append(len(issued))
        if len(issued) < 3:
            continue
        d = jitter.command_deltas(frames["a"], interval)
        signed = np.diff(np.asarray(issued, dtype=float), axis=0)
        start = max(0, settle - 1)
        if d[start:].size:
            deltas.append(float(np.mean(d[start:])))
        if signed[start:].shape[0] >= 2:
            reversals.append(float(np.mean(
                jitter.sign_reversals_per_s(signed[start:], interval))))
        q = np.asarray(frames["q"], dtype=float)[settle:]
        if q.size:
            qsum.append(float(np.mean(q.sum(axis=1))))
    return {"delta": float(np.mean(deltas)) if deltas else 0.0,
            "reversals": float(np.mean(reversals)) if reversals else 0.0,
            "qsum": float(np.mean(qsum)) if qsum else 0.0,
            # Carried so that "every statistic is 0.0" can be told apart from
            # "the episode was too short to have any" — the two are identical
            # in the numbers and opposite in meaning.
            "steps": float(np.mean(lengths)) if lengths else 0.0,
            "usable": len(deltas)}


def main() -> int:
    task = json.loads(TASK.read_text())
    model_xml = (TASK.parent / "model-model.xml").read_bytes()
    interval = float(task["episode"]["control_interval_s"])
    ceiling = 1.0 / interval
    lows = [float(a["low"]) for a in task["actions"]]
    highs = [float(a["high"]) for a in task["actions"]]
    nominal = [0.0] * len(lows)

    def hold(_step, _obs):
        return nominal

    def chatter(step, _obs):
        return highs if int(step) % 2 == 0 else lows

    def sweep(step, _obs):
        # A full triangle every 2 s: full amplitude, near-zero reversal rate.
        period = int(round(2.0 / interval))
        phase = (int(step) % period) / period
        frac = 4.0 * phase - 1.0 if phase < 0.5 else 3.0 - 4.0 * phase
        return [h * frac for h in highs]

    # From the bare keyframe, with nothing pushing: the guard is checking the
    # arithmetic, and a reset drop plus a shove schedule only shortens the
    # episodes the falling controllers get.
    sys.path.insert(0, str(REPO / "harness"))
    from _episodes import apply_variant  # noqa: E402
    bare = apply_variant(task, {"disturbance": False, "reset_variation": False})

    cases = {name: play(bare, model_xml, fn, SEEDS)
             for name, fn in (("hold", hold), ("chatter", chatter),
                              ("sweep", sweep))}

    print(f"fire_jitter_guard — {len(SEEDS)} seeds, whole episode, no reset "
          f"variation, nothing pushing\n  reversal ceiling "
          f"{ceiling:.1f} /s\n")
    print(f"  {'controller':<10} {'steps':>7} {'mean Δ deg':>12} {'rev/s':>10} "
          f"{'Σ|q̇| deg/s':>14}")
    for name, r in cases.items():
        print(f"  {name:<10} {r['steps']:>7.1f} {r['delta']:>12.4f} "
              f"{r['reversals']:>10.2f} {r['qsum']:>14.2f}")

    failures = []
    for name, r in cases.items():
        if r["usable"] == 0:
            failures.append(
                f"{name} produced no usable episode ({r['steps']:.0f} control "
                f"steps) — every statistic below is 0.0 because there were no "
                f"frames, not because the controller was quiet")

    # hold: exactly zero, not merely small.
    if cases["hold"]["delta"] != 0.0:
        failures.append(f"hold has a non-zero command delta "
                        f"{cases['hold']['delta']}")
    if cases["hold"]["reversals"] != 0.0:
        failures.append(f"hold reverses {cases['hold']['reversals']} /s")

    # chatter: pinned at the ceiling, and far noisier in the joints.
    if cases["chatter"]["reversals"] < 0.95 * ceiling:
        failures.append(f"chatter reverses {cases['chatter']['reversals']:.2f} "
                        f"/s, under 95 % of the {ceiling:.1f} ceiling")
    if cases["chatter"]["qsum"] <= 5.0 * max(cases["hold"]["qsum"], 1.0e-9):
        failures.append("chatter does not move the joints appreciably more "
                        "than hold")

    # sweep: the case a magnitude statistic gets wrong.
    if cases["sweep"]["delta"] <= 0.0:
        failures.append("sweep produced no command motion at all")
    if cases["sweep"]["reversals"] >= 0.1 * ceiling:
        failures.append(f"sweep reverses {cases['sweep']['reversals']:.2f} /s "
                        f"— a smooth ramp must not read as chatter")
    if cases["chatter"]["reversals"] <= 10.0 * max(cases["sweep"]["reversals"],
                                                   1.0e-9):
        failures.append("chatter and sweep do not separate on reversal rate, "
                        "which is the whole reason the rate is reported")

    print()
    if failures:
        for f in failures:
            print(f"  FAIL  {f}")
        return 1
    print("  PASS  hold is silent, chatter pins the ceiling, and a smooth\n"
          "        full-amplitude sweep is NOT read as chatter — which is the\n"
          "        separation a magnitude statistic alone cannot make.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
