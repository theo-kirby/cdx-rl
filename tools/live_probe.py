"""Does live mode answer a push? The verification, headlessly.

``live_open`` / ``live_step`` / ``live_close`` are the ops the Shell's panel
drives (ADR-109/110/111), so driving them over the same NDJSON
``tools/cadexd_client.py`` already speaks proves the loop without a GUI.
``prepare_live``'s three digest re-checks happen inside ``live_open`` -- it
plays the exact files the accepted rollout used -- so a successful open **is**
those three checks passing.

Two shapes worth writing down, both learned the hard way:

* a frame carries ``component_placements``, keyed by component name, each
  ``{position_mm, rotation_xyzw}``. Not a flat 16-float matrix.
* a push needs a **body**. Without one the reply is ``live: false`` with a
  reason and **zero frames** -- a successful op that declined, exactly as
  ``live_open`` declines a project with no rollout. Reading ``frames[-1]``
  without checking ``live`` is an IndexError instead of the sentence.

    python live_probe.py <project> [output]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cadexd_client import CadexdClient, Engine

project = sys.argv[1]
output = sys.argv[2] if len(sys.argv) > 2 else "stand_play"
BODY = "pelvis"

engine = Engine.resolve()
print(f"engine    {engine.describe().get('root')}")


def height_mm(frame):
    placement = (frame.get("component_placements") or {}).get(BODY) or {}
    return float((placement.get("position_mm") or [0, 0, float("nan")])[2])


def advance(client, steps, push=None):
    args = {"steps": steps}
    if push:
        args["push"] = push
    reply = client.checked("live_step", args)
    if not reply.get("live"):
        raise SystemExit(f"DECLINED  {reply.get('reason')}")
    return reply


with CadexdClient(engine) as client:
    client.open_project(project)
    opened = client.checked("live_open", {"output": output, "seed": 4,
                                          "variation": False})
    if not opened.get("live"):
        raise SystemExit(f"DECLINED  {opened.get('reason')}")
    identity = opened.get("policy") or {}
    print(f"live      {opened['control_hz']} Hz, {opened['episode_seconds']} s, "
          f"{len(opened.get('components') or [])} components, "
          f"{len(opened.get('actuator_channels') or [])} channels")
    print(f"policy    {identity.get('label')!r}  {identity.get('weights')}  "
          f"{str(identity.get('sha256'))[:12]}…  "
          f"trained_label={identity.get('trained_label')!r}")

    settled = advance(client, 25)
    z0 = height_mm(settled["frames"][-1])
    print(f"\nsettled   pelvis z = {z0:.1f} mm at t = {settled['time_s']:.2f} s")

    # 0.6 N due world +X for 60 ms, at the pelvis. Inside the task's own
    # declared 0.3-0.8 N band (ADR-106's revision, experiment 001 phase B), so
    # this is a shove the policy was trained to expect rather than a stunt.
    pushed = advance(client, 5, {"newtons": 0.6, "azimuth_rad": 0.0,
                                 "duration_s": 0.06, "body": BODY})
    z1 = height_mm(pushed["frames"][-1])
    print(f"pushed    0.6 N at azimuth 0 for 60 ms on {BODY} "
          f"-> z = {z1:.1f} mm")

    after = advance(client, 120)
    z2 = height_mm(after["frames"][-1])
    print(f"answered  z = {z2:.1f} mm at t = {after['time_s']:.2f} s")
    print(f"          terminated={after['terminated']}  "
          f"termination={after.get('termination')!r}  "
          f"resets={after['reset_count']}")

    client.checked("live_close", {})
    print("\nclosed.")
    if after["terminated"] or after["reset_count"]:
        print("It fell. That is a result about the policy, not about the loop "
              "-- the push was answered, and the answer was wrong.")
    else:
        print(f"THE POLICY ANSWERED THE PUSH: still standing "
              f"{after['time_s']:.2f} s in, {z2 - z0:+.1f} mm of the "
              f"settled height.")
