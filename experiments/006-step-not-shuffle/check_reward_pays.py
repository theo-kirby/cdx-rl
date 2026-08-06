"""Hazard 9: every reward term must PAY its weight at the nominal pose.

A kernel that reads 0 where the machine is doing the right thing is a constant
penalty, not a shaping term. `reward_standing.py` is the mechanism's own
version of this check; this one runs against the derived bundle directly,
through the ENGINE's own evaluator rather than a re-derived one.
"""
import sys, json
from pathlib import Path
sys.path.insert(0,"/home/theo/cadex-prs/src/Mod/cadex")
sys.path.insert(0,"/home/theo/cdx-rl/harness"); sys.path.insert(0,"/home/theo/cdx-rl")
import CadexDynamics as cd, numpy as np
from _episodes import apply_variant

for name in ("stand-b8-clamp25","stand-b8-clamp25-quiet"):
    task=json.loads(Path(f"tasks/{name}/stand-task.json").read_text())
    xml=Path(f"tasks/{name}/model-model.xml").read_bytes()
    bare=apply_variant(task,{"disturbance":False,"reset_variation":False})
    m=cd.load_model(xml)
    obs={}
    def s(step,data,_f,_a):
        if step in (None,0):
            obs.update(cd.observation_values(bare, data.sensordata))
        return None
    cd.evaluate_episode(m,bare,actions=lambda _s,o:[0.0]*len(task["actions"]),sample=s,seed=0)
    print(f"=== {name} ===")
    total=0.0
    for r in task["reward"]:
        fn=cd.compile_reward(r["expression"], names=list(obs), context=r["label"])
        v=float(cd.evaluate_reward(fn, obs, context=r["label"])); w=float(r["weight"]); total+=w*v
        flag="" if v>0.5 else "   <-- pays little at the nominal pose"
        print(f"  {r['label']:<14} w {w:4.2f}  kernel {v:7.4f}  pays {w*v:6.4f}{flag}")
    print(f"  {'TOTAL':<14}          {'':7}  pays {total:6.4f} of {sum(float(r['weight']) for r in task['reward'])}\n")
