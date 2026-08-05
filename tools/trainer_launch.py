#!/usr/bin/env python
"""Run ``cadex_train.py`` with CPython's cyclic collector switched off.

```
python tools/trainer_launch.py <trainer.py> [the trainer's own arguments…]
```

``tools/train.py`` puts this in front of the trainer. It exists because a
process-wide interpreter option has to be set *before* the trainer starts,
and cdx-rl does not edit the trainer to do it.

That was originally a hard constraint — cdx-rl was read-only toward Cadex.
Since 2026-08-05 it could be a PR instead. It should not be: the cyclic GC is
switched off here to work around a **jaxlib** fault, not a Cadex one, and
baking a workaround for somebody else's bug into `cadex_train.py` would be
the wrong place for it. A launcher is the honest shape. See ``cloud.md`` §1.

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

**Read that table with suspicion.** It is one run per row, and sb9x's
remaining fault is a **race** — the kernel calls it a general protection
fault in jaxlib's CUDA plugin, and it has struck at iteration 0, iteration 7,
a checkpoint, and ``train()``'s return. One run per configuration cannot
separate a setting's effect from a coin flip, so treat this as *"the
collector was off for every run that finished"*, not as a demonstration that
turning it off is what made them finish. ``cloud.md`` §1 has the full
argument; establishing a real effect needs repeats.

What is not in doubt is the **cost of getting the exit code wrong**: it is
the only thing ``train.py`` and the sweep have to tell a finished seed from a
broken one, which is why ``post_mortem`` now reads the run directory instead
of trusting it. Tracing also measured about 1.5x faster without the collector
walking those graphs — that one is a timing difference large enough to see in
a single run.

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
policy is written. Measured at n=3: **2 runs of 40 iterations exited 0, one
exited -11**, so the fault is intermittent rather than certain. See
``cloud.md`` §1; it is open, and it is a race, so no single run — clean or
crashed — is evidence about any setting here.
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
