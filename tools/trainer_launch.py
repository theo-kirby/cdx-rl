#!/usr/bin/env python
"""Run ``cadex_train.py`` with CPython's cyclic collector switched off.

```
python tools/trainer_launch.py <trainer.py> [the trainer's own arguments…]
```

``tools/train.py`` puts this in front of the trainer. It exists because
``/home/theo/cadex`` is read-only from cdx-rl (``CLAUDE.md`` invariant 1), so
the only place to set a process-wide interpreter option is *before* the
trainer starts.

**What it is for.** JAX builds enormous reference-cycle graphs while tracing,
and a collection that lands mid-trace walks them. On sb9x that walk segfaults:
``faulthandler`` catches it with ``Garbage-collecting`` at the top of the stack
and MJX's ``step_env``/``stepped`` tracing frames underneath. Measured at 2048
environments, with the stack already raised and XLA preallocation already off:

===========  ===========================  =====================
collector    artefacts                    exit
===========  ===========================  =====================
**on**       all written, witness passes   **SIGSEGV** at shutdown
**off**      all written, witness passes   **0**
===========  ===========================  =====================

The artefacts survive either way, so this is not about losing a policy. It is
about the **exit code**, which is the only thing ``train.py`` and the sweep
have to tell a finished seed from a broken one — a seed that wrote everything
and then died at teardown would be recorded as infrastructure failure, and a
sweep would be pausing on runs that actually worked. Tracing is also about
1.5x faster without the collector walking those graphs.

**What it costs.** Reference cycles are freed only when the process exits.
That was an open worry when this was written and has since been **measured
and cleared**: RSS across a 40-iteration run at 2048 environments, sampled
against the iteration counter, was *flat to the byte* from iteration 1 to 18
and again from 19 to 38. What grows is per-*compile*, not per-iteration —
about 500-750 MB at each checkpoint, which the collector would not have
reclaimed either. There is no accumulation across training to worry about.

**What this does not fix.** The table above is about the exit code of a run
that reached the end. It does not make a long run reach the end: with this
shim, the raised stack and XLA preallocation all in place, a **40-iteration**
run still dies with ``SIGSEGV`` as ``train()`` returns, before the final
policy is written. See ``cloud.md`` §1 — sb9x can train and cannot finish,
and that is open. Do not read a clean 3-iteration run as evidence otherwise;
every one of these faults is scale-dependent.
"""

from __future__ import annotations

import gc
import runpy
import sys

if len(sys.argv) < 2:
    raise SystemExit("usage: trainer_launch.py <trainer.py> [args…]")

gc.disable()

trainer = sys.argv[1]
# Hand the trainer the argv it would have had if it were argv[0]. Its own
# parser sets prog= explicitly, so --help stays correct.
sys.argv = sys.argv[1:]
runpy.run_path(trainer, run_name="__main__")
