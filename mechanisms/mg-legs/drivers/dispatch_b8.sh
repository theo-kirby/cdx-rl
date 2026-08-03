#!/usr/bin/env bash
#
# Dispatch B8 to the GPU box, with the run's own bundle kept beside it.
#
#     ~/cdx-mjc/dispatch_b8.sh probe      100 iterations, checkpoint every 50
#     ~/cdx-mjc/dispatch_b8.sh run N      N iterations, checkpoint every 50
#
# Two reasons this is a file rather than a command line in a message.
#
# **The hyperparameters.** Thirteen flags, and the run they configure is
# hours long. A typo in `--discount` is not something the trainer can
# refuse -- 0.97 and 0.997 are both legal and mean a 0.33 s horizon and a
# 3.3 s one -- so the numbers live somewhere they can be read and diffed.
#
# **The bundle goes with the checkpoints.** ADR-099 section 5: a `.cxpolicy`
# without the task bundle it was trained against cannot be replayed, scored
# or installed, and `runs/b6/` is the standing example of what that costs --
# thirty checkpoints and no bundle. This copies the accepted bundle AND the
# MJCF it references into the run directory before anything is dispatched.
#
# ---------------------------------------------------------------------------
# WHAT B8 CHANGES AGAINST B7, AND WHY THE FLAGS MOVE WITH IT
# ---------------------------------------------------------------------------
#
# B8 is four structural changes, all of them in the script: position
# actuators, a crouched nominal pose with symmetric limits, an all-positive
# reward, and 50 Hz control. The TASK does not move -- B6's band, B6's
# windows, B6's reset -- because with four things changing at once the one
# thing that has to hold still is the thing being measured against.
#
# THE HORIZONS ARE B6'S, IN SECONDS, WHICH IS WHY THE NUMBERS LOOK DIFFERENT.
# `--discount` and `--gae-lambda` are both counted in STEPS and the control
# rate halved, so keeping B6's step counts would have doubled every horizon
# in seconds:
#
#     flag           B6/B7 at 100 Hz          B8 at 50 Hz
#     --discount     0.995  -> 2.0 s          0.99   -> 2.0 s
#     --gae-lambda   0.97   -> 0.35 s         0.95   -> 0.35 s
#
# 1/(1-gamma) is 200 steps at 0.995 and 100 at 0.99; both are 2.0 s. The GAE
# credit chain 1/(1-gamma*lambda) is 34.8 steps at B6's pair and 17.4 at
# this one; both are 0.35 s. Nothing about the temporal structure of the
# problem changed -- only how many steps it is counted in.
#
# `--unroll 40` at 50 Hz is 0.8 s per segment, against B6's 0.4 s, and it is
# left alone deliberately: a segment that spans most of a recovery is what
# the halved control rate was bought for.
#
# ---------------------------------------------------------------------------
# 1200 ITERATIONS AND NOT 2400, AND IT IS THE SAME EXPERIENCE
# ---------------------------------------------------------------------------
#
# At 50 Hz each control step runs TEN solver substeps instead of five, so an
# iteration costs about twice the physics -- expect ~8-9 s against B7's 4.36.
# 1200 iterations is therefore about 2.9 h, which is B6's wall clock, AND it
# is the same simulated robot-time:
#
#     1200 x 2048 envs x 40 unroll x 0.02 s  =  546 h of robot time
#     2400 x 2048      x 40        x 0.01 s  =  546 h
#
# So this is not a shorter run. It is the same run at half the sample rate.
# 24 checkpoints at every 50, which is B6's scoring cadence.
#
# `--initial-std 0.4` CARRIES OVER NUMERICALLY AND MEANS SOMETHING ELSE.
# Under B7 it was noise on a torque; here it is +-0.4 in normalised action
# units, which is about +-12 deg of hip-pitch setpoint jitter and +-8 deg at
# the knee. That is a lot of jitter for a machine whose whole premise is
# that the zero action stands, and it is the first flag to try if the probe
# is worse than the zero-action baseline below. Flagged as a B9 lever rather
# than moved here, because B8 already changes four things.
#
# ---------------------------------------------------------------------------
# THE PROBE BAR, AND IT IS SHARPER THAN ANY PREVIOUS RUN'S
# ---------------------------------------------------------------------------
#
# With a PD action space ITERATION 0 SHOULD ALREADY SHOW LONG EPISODES. The
# measured baseline is not a guess: `feasibility.py` check 5 runs the ZERO
# ACTION against this exact task and gets
#
#     survived 0/12, mean 60.2 of 300 steps  =  1.20 s, all `tipped`
#
# against B7's iteration 0 of 85 steps at 100 Hz = 0.85 s. So an untrained
# B8 policy starts from 1.4x B7's episode length in SECONDS before it has
# learned anything, and the exploration noise above is what it has to
# overcome to show it. If the first iterations are not clearly better in
# seconds than 0.85 s, the action-space change did not take, and the first
# thing to check is the exported MJCF -- `<general ... biastype="affine"
# gainprm="0.3" biasprm="0 -0.3 -0.01"/>` on all ten actuators -- before
# looking anywhere else.
#
# Also check the usual: witness margin >= 100x, `episode_steps` not pinned at
# envs x unroll, reward and loss finite, and that the machine is not simply
# standing rigidly -- mean `com_z` near the new Z0 of 140.9 mm. That last one
# is the risk B8's sign change INVERTS: `upright`, `height`, `posture` and
# `effort` are all maximised by standing perfectly still and are worth 2.0 of
# the 5.3 between them. If it stands and never steps, the lever is raising
# `capture` relative to them, NOT adding a cost.
#
# THE BOX IS sb1x AND THAT IS AN OVERRIDE, printed rather than assumed.
# `training/.remote.env` says sb9x, which is an RTX 4070 with 12 GB where
# `--envs 2048` is a real memory risk. sb1x is a 5090 with 32 GB, and the box
# B6 ran 2048 envs at unroll 40 on for 3.9 h. It also happens to be the box
# this laptop can reach unattended -- sb9x authenticates with a
# passphrase-protected key that needs an ssh-agent, sb1x with one that does
# not. Set CADEX_TRAIN_SSH_HOST to move it back.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${CADEX_REPO:-${HOME}/cadex}"
PROJECT="${HERE}/mg-legs.cadex"
RUN_DIR="${HERE}/runs/b8"
export CADEX_TRAIN_SSH_HOST="${CADEX_TRAIN_SSH_HOST:-sb1x}"

mode="${1:-probe}"

# The trainer's own name for the policy, and the project's convention: the
# run number, so `stand8` was B6, `stand9` was B7 and this is B8.
NAME=stand10

# CHECKPOINT EVERY 50 IN BOTH MODES, where B7 used 100 for a full run. 1200
# iterations at every 100 would be twelve checkpoints to score where B6 had
# fifteen, and the scoring is a conjunction over checkpoints -- `best` tracks
# reward and reward has selected the policy that does NOT step three times
# now, so the checkpoints ARE the result and there is no point being thrifty
# with them.
case "${mode}" in
    probe) iterations=100   ; every=50  ; name="${NAME}-probe" ;;
    run)   iterations="${2:?usage: dispatch_b8.sh run <iterations>}"
           every=50         ; name="${NAME}" ;;
    *)     echo "usage: $(basename "$0") probe | run <iterations>"; exit 2 ;;
esac

# The accepted bundle, found the way `compare.py` finds it -- newest by
# mtime under script_artifacts -- so this cannot be dispatched against a
# stale build somebody forgot to rebuild.
bundle="$(ls -t "${PROJECT}"/script_artifacts/*/*/outputs/*-task.json 2>/dev/null | head -1)"
[ -n "${bundle}" ] || { echo "FAIL: no task bundle. Run rebuild.py first."; exit 1; }
outputs="$(dirname "${bundle}")"

echo "==> bundle  ${bundle}"
# WHAT THIS PRINTS IS THE FOUR CHANGES, so a dispatch against a stale build
# is visible in the first ten lines rather than three hours later. The
# action-space line is the one to read: `position` is B8, `motor` is not.
python3 - "${bundle}" <<'PY'
import json, sys
task = json.load(open(sys.argv[1]))
channels = sum(int(o["dim"]) for o in task["observations"])
print(f"    {channels} channels, {len(task['reward'])} reward terms, "
      f"{task['episode']['episode_seconds']:g} s at "
      f"{task['episode']['control_hz']} Hz "
      f"({task['episode']['max_steps']} steps, "
      f"{task['episode']['solver_steps_per_action']} solver steps each)")
kinds = sorted({str(a["kind"]) for a in task["actions"]})
print(f"    action space {'/'.join(kinds)}, {len(task['actions'])} actuators")
for action in task["actions"][:1]:
    print(f"    e.g. {action['joint']:14s} "
          f"[{float(action['low']):+g}, {float(action['high']):+g}] "
          f"{action['unit']}, so zero action is "
          f"{0.5 * (float(action['low']) + float(action['high'])):+g}")
budget = sum(float(row["weight"]) for row in task["reward"])
signs = "ALL POSITIVE" if all(float(r["weight"]) >= 0 for r in task["reward"]) \
        else "MIXED SIGN"
print(f"    reward {signs}, budget {budget:+.3f}")
for row in task["reward"]:
    print(f"    {row['label']:14s} {float(row['weight']):+.3f}")
for entry in task["disturbance"]:
    if not entry["sustained"]:
        print(f"    {entry['label']:14s} "
              f"{float(entry['newtons_low']):g}-{float(entry['newtons_high']):g} N "
              f"at {float(entry['at_low_s']):g}-{float(entry['at_high_s']):g} s")
if signs != "ALL POSITIVE":
    raise SystemExit("FAIL: this bundle's reward is not all-positive, so it "
                     "is not B8's. Rebuild.")
if "position" not in kinds:
    raise SystemExit("FAIL: this bundle's action space is not position, so "
                     "it is not B8's. Rebuild.")
PY

mkdir -p "${RUN_DIR}"
cp -f "${bundle}" "${RUN_DIR}/"
for model in "${outputs}"/*-model.xml; do cp -f "${model}" "${RUN_DIR}/"; done
echo "==> bundle and model copied into ${RUN_DIR} (ADR-099 section 5)"

# THE CARD HAS TO BE FREE FIRST, and this check is here because skipping it
# cost a dispatch. 2048 environments hold about 25 GB of the 5090's 32 GB, so
# two of these do not fit and the second does not queue -- it dies about
# twenty seconds in with `XlaRuntimeError: INTERNAL: cuSolver internal error`,
# which reads like a driver fault and is really "there was no memory left to
# put a workspace in".
#
# The way to get there is not exotic: dispatch the moment a previous run's LOG
# reaches its last iteration. The log is written before the process exits --
# there are still checkpoints and a witness rollout to write after it -- so
# the old job is still holding the whole card while the new one starts.
free_mib=$(ssh -o BatchMode=yes "${CADEX_TRAIN_SSH_HOST}" \
    "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits" 2>/dev/null | head -1)
if [ -z "${free_mib}" ]; then
    echo "FAIL: could not read GPU memory on ${CADEX_TRAIN_SSH_HOST}."; exit 1
fi
echo "==> gpu ${free_mib} MiB free"
if [ "${free_mib}" -lt 28000 ]; then
    echo "FAIL: only ${free_mib} MiB free and 2048 environments want ~25 GB."
    echo "      Something is still on the card -- most likely a previous run"
    echo "      that has finished its log but not yet exited. Check with:"
    echo "        ssh ${CADEX_TRAIN_SSH_HOST} nvidia-smi"
    exit 1
fi

echo "==> dispatching ${mode} to ${CADEX_TRAIN_SSH_HOST}: ${iterations} iterations, checkpoint every ${every}"
exec "${REPO}/training/remote_train.sh" train "${bundle}" \
    "${RUN_DIR}/${name}.cxpolicy" --detach -- \
    --envs 2048 \
    --unroll 40 \
    --epochs 5 \
    --discount 0.99 \
    --gae-lambda 0.95 \
    --initial-std 0.4 \
    --entropy 2e-3 \
    --hidden 64 64 \
    --learning-rate 3e-4 \
    --clip 0.2 \
    --value-weight 0.5 \
    --iterations "${iterations}" \
    --checkpoint-every "${every}" \
    --seed 0 \
    --label "b8 ${mode}"
