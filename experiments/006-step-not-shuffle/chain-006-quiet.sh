#!/usr/bin/env bash
# Experiment 006 — arm Q, both seeds, then score. One card since sb9x was
# retired, so these are serial by necessity.
#
# THERE IS NO FORK IN THIS SCRIPT, AND THAT IS THE DESIGN.
#
# The plan's original shape was two treatments (a band raise and this one) with
# a pre-registered fork choosing which one got a replication seed. The band arm
# was vetoed by Phase A's capability sweep -- `harness capability` puts ~50 %
# survival at 0.914 N where the plan proposed 1.8 N, and the incumbent survives
# 0/12 there -- so both slots go to this arm instead.
#
# The second seed is therefore UNCONDITIONAL. 004's fork spent its second slot
# on a third treatment, criterion 4 went unmet, and every number in it was n=1
# in seeds. A gate that could stop after one seed would reproduce exactly that.
# If seed 2 produces nothing, seed 1 is still a fresh seed and still worth its
# five hours.
#
# The control is already paid for: jobs/stand12-s2-20260804-163759 and
# jobs/stand12-s1-20260804-214522. No new control run.
#
#   experiments/006-step-not-shuffle/README.md sections 1-7 are the
#   pre-registration and were written before this ran (ADR-097).
#
# Run it from anywhere; it cds to the repository root itself.
#
#   nohup experiments/006-step-not-shuffle/chain-006-quiet.sh &
#
# It lives HERE rather than in jobs/ because jobs/ is gitignored in full and
# this script is the executable form of README sections 6 and 7. 004 kept its
# chain script in jobs/ and it was never committed.
#
# To cancel:  pkill -f chain-006-quiet.sh
set -u
cd /home/theo/cdx-rl

BUNDLE=tasks/stand-b8-clamp25-quiet/stand-task.json
COMMON=tasks/stand-b8-clamp25/stand-task.json
LABEL=stand15
LOG=jobs/chain-006-quiet.log
RESULTS=experiments/006-step-not-shuffle/results

# Verified in-session by `sha256sum /home/theo/cadex-prs/training/cadex_train.py`
# and pasted from that command's output. Never retyped: this repository has
# published an invented digest twice, most recently one whose first 12
# characters were right and whose other 52 were fabricated.
TRAINER=bb133b64d57d8f2b521c22b1111e182428ef70e4f2088a5e7cee945a0ec71dc2

# The bundle this experiment pre-registered. If it has moved, the README's
# sections 1-7 describe a different task than the one about to be trained.
BUNDLE_SHA=5d8dd7c1dfe7be5d39d7b62fa4c80f29667b959c3c9c8827d47b2003b7fb7c01

say() { echo "$(date -u +%FT%TZ)  $*" | tee -a "$LOG"; }

# **`pgrep -f trainer_launch.py` IS TOO LOOSE, and it aborted this chain's
# first launch.** `-f` matches the whole command line of every process, so any
# shell, editor or grep that merely *mentions* the name counts as a running
# trainer — the first launch died on `ABORT: a trainer is already running`
# with an idle card, because the wrapper shell that started it had the string
# in its own argv.
#
# `tools/train.py` builds the command as `<interpreter> <trainer_launch.py>
# <cadex_train.py> …`, so anchoring at the interpreter is what distinguishes
# the real thing from a mention of it. `cadex_train` is included because
# `--child-gc` drops the shim and runs the trainer directly.
#
# Verified both ways before use: this pattern matches a venv python running
# such a file and does NOT match a shell whose argv merely contains the name.
TRAINER_RE='^[^ ]*/python[0-9.]* [^ ]*/(trainer_launch|cadex_train)\.py'
trainer_running() { pgrep -f "$TRAINER_RE" >/dev/null 2>&1; }

say "experiment 006 arm Q — two seeds of ${LABEL} on ${BUNDLE}"

# --- refuse to start against a bundle that is not the pre-registered one ---
HAVE=$(sha256sum "$BUNDLE" | cut -d' ' -f1)
if [ "$HAVE" != "$BUNDLE_SHA" ]; then
    say "ABORT: ${BUNDLE} digests ${HAVE}, and this experiment pre-registered"
    say "       ${BUNDLE_SHA}. Regenerate it with make_arm_bundle.py or update"
    say "       the README -- do not train against an undeclared task."
    exit 1
fi
say "bundle digest matches the pre-registration: ${BUNDLE_SHA}"

set -a; . ./config/env; set +a

# One run at a time on this card.
if trainer_running; then
    say "ABORT: a trainer is already running"
    exit 1
fi

dispatch() {
    local seed="$1"
    say "dispatching ${LABEL} seed ${seed}"
    # ALL FOURTEEN HYPERPARAMETERS, EXPLICITLY. `tools/train.py` defaults
    # --discount to 0.995 and --gae-lambda to 0.97 from RUN_200109, and that
    # silent substitution killed 005's first dispatch. The two lines below are
    # 004's and 005's values and must not be dropped.
    uv run python tools/train.py \
      --bundle "$BUNDLE" \
      --label "$LABEL" \
      --seeds "$seed" \
      --iterations 1800 \
      --checkpoint-every 50 \
      --envs 2048 --unroll 40 --epochs 5 \
      --hidden 64 64 \
      --learning-rate 3e-4 \
      --discount 0.99 \
      --gae-lambda 0.95 \
      --clip 0.2 \
      --entropy 2e-3 \
      --value-weight 0.5 \
      --initial-std 0.4 \
      --require-trainer "$TRAINER" \
      --require-device gpu \
      --timeout 25200 \
      --patience 0 \
      --supervise \
      --detach >>"$LOG" 2>&1
    say "dispatch of seed ${seed} returned $?"

    # `--detach` returns immediately, so this has to wait on the TRAINER, not
    # on the dispatcher. **A fixed `sleep 30` here is a race**: if the trainer
    # takes longer than that to appear, the wait loop below sees nothing
    # running, returns at once, and the next seed is dispatched CONCURRENTLY
    # onto a single card. So the appearance is polled for, and a trainer that
    # never appears aborts the chain rather than being treated as finished.
    local waited=0
    while [ "$waited" -lt 300 ]; do
        trainer_running && break
        sleep 5
        waited=$((waited + 5))
    done
    if ! trainer_running; then
        say "ABORT: seed ${seed} dispatched but no trainer appeared in 300 s."
        say "       Refusing to dispatch anything else — check ${LOG} and the"
        say "       newest jobs/${LABEL}-s${seed}-* directory."
        exit 1
    fi
    say "seed ${seed} trainer is up after ${waited} s"

    while trainer_running; do sleep 60; done
    say "seed ${seed} is no longer running"
}

dispatch 2
dispatch 1

# --- score both, on CPU, against the COMMON bundle ---
#
# `--task` is deliberately clamp25 for both arms: the yardstick the control was
# measured on. Legal because `harness steps` and `jitter.py` never call
# `verify_policy` -- proved by running it, README section 4c. `capability` and
# `compare` WOULD refuse, which is why neither appears here.
say "scoring on CPU — this does not contend with the card"
mkdir -p "$RESULTS"

for SEED in 2 1; do
    RUN=$(ls -dt jobs/${LABEL}-s${SEED}-* 2>/dev/null | head -1)
    if [ -z "$RUN" ]; then
        say "WARNING: no run directory for seed ${SEED} — skipping its scoring"
        continue
    fi
    say "scoring ${RUN}"

    # Scored one checkpoint set at a time. `harness steps` keys its results by
    # policy BASENAME, so two seeds of the same label in one invocation are
    # silently SUMMED -- the two stand12 seeds came back as `survived 36/24`
    # and the 36 is the only tell.
    uv run python -m harness steps --dir "$RUN" --task "$COMMON" \
        --seeds 24 --json >"$RESULTS/steps-${LABEL}-s${SEED}.json" 2>>"$LOG"

    /home/theo/cadex-train-venv/bin/python \
        mechanisms/mg-legs/drivers/jitter.py --series "$RUN" --stride 250 \
        --task "$COMMON" --seeds 12 --json \
        >"$RESULTS/jitter-${LABEL}-s${SEED}.json" 2>>"$LOG"

    /home/theo/cadex-train-venv/bin/python \
        mechanisms/mg-legs/drivers/hazard15.py --series "$RUN" --stride 50 \
        --task "$COMMON" --seeds 6 --json \
        >"$RESULTS/hazard15-${LABEL}-s${SEED}.json" 2>>"$LOG"
done

# --- read the pre-registered criteria, without deciding anything ---
say "reading criteria Q1/Q3 against the control (README section 7)"
uv run python - <<'PY' 2>&1 | tee -a "$LOG"
import json, glob, statistics
R = "experiments/006-step-not-shuffle/results/"
# The matched control, measured in Phase A. Literals, so this reading cannot
# drift with anything measured later.
CONTROL = {2: 1522.33, 1: 1573.88}
CONJ    = {2: 15, 1: 15}
for seed in (2, 1):
    try:
        j = json.load(open(f"{R}jitter-stand15-s{seed}.json"))
        s = json.load(open(f"{R}steps-stand15-s{seed}.json"))
        h = json.load(open(f"{R}hazard15-stand15-s{seed}.json"))
    except FileNotFoundError as exc:
        print(f"  seed {seed}: not scored ({exc.filename})")
        continue
    last = j["rows"][-1]
    q = last["sum_abs_qvel_mean_deg_s"]
    base = CONTROL[seed]
    best = max((v.get("both", 0) for v in (s.get("results") or {}).values()),
               default=0)
    duty = max(r["settled_duty_worst"] for r in h["rows"])
    print(f"  seed {seed}:  settled sum|qdot| {q:8.2f} vs control {base:8.2f}"
          f"  -> {100*(1-q/base):+6.1f} %   Q1 {'PASS' if q <= 0.6*base else 'FAIL'}")
    print(f"           conjunction {best}/24 vs control {CONJ[seed]}/24"
          f"           (Q2 needs the paired McNemar — read `harness steps`)")
    print(f"           hazard15 duty {duty*100:5.2f} %"
          f"                     Q3 {'PASS' if duty < 0.25 else 'FAIL'}")
PY

say "done. Sections 8 and 9 of the README are written from these files, by hand,"
say "     and nothing above section 8 may be edited."
