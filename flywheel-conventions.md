# How cdx-rl uses Flywheel

**This is the normative document.** It says what we do and why. Its companion
[`flywheel.md`](flywheel.md) is descriptive — the API surface, the measured
traps, and a render of the graph. When they disagree, this file wins on
*policy* and `flywheel.md` wins on *what the server does*.

Rewritten **2026-08-06**, when the graph was forked and restructured. The v1
graph (`rapid-bar-6214`, 19 nodes) is **frozen as the historical record** and
must not be deleted — v2's artifact blobs are aliases into its storage. Every
rule below is either a mechanic that was executed or a defect that was
measured in v1 and named here so it does not come back.

---

## 0. The one-paragraph version

The graph holds **claims and their evidence**, arranged by **what caused
what**, in **four lanes**. A node is under 4 KB so it can be corrected. An
experiment gets a **protocol node before dispatch** and a **result node
after**. A replication is a **child that judges its parent**. Being wrong is
handled by **retracting in public**, never by editing quietly. If you write a
node and cannot name its parent, you have not understood the work yet.

---

## 1. What the graph is for

Success criterion 4: **a fresh agent can rebuild the picture from the graph.**
Not from a conversation, not from a directory listing — from nodes, their
artifacts, and the edges between them.

That sentence decides most of the arguments below. If a convention makes the
graph readable to someone arriving with no context, it stays. If it only makes
it tidy, it does not.

**The graph is not a copy of the repository.** The repo holds the full
write-up, the drivers, the bundles and the run directories. The graph holds
the *claim*, the evidence for it, and the edge to what it changed.

### The failure v1 actually had, stated plainly

**v1 was written to and never read from.** Every experiment in this project
was designed out of `CLAUDE.md` and the experiment READMEs; the graph was a
publication target. Two consequences followed, and both are addressed by
structure rather than by exhortation:

* it **duplicated** `CLAUDE.md` — which was the better index, so the graph paid
  a cost for a job something else was already doing;
* it **gated nothing** — no run was ever blocked on a node existing, so a
  4.75 GPU-hour experiment (`stand13`) finished and left no trace.

§4's protocol node is the fix: it makes the graph a thing you write **before**
you spend, not only after.

---

## 2. Node structure

There are no typed nodes in Flywheel. Type is carried by tags (§5).

**Three fields matter and they have different jobs:**

| field | job | rule |
|---|---|---|
| `title` | the index entry — what every tree render, list and search shows | must be true **on its own**, with no child node required to correct it |
| `summary` | the abstract — read by anyone deciding whether to open the node | headline number **and** its caveat; **under ~500 characters** |
| `content` | the record | Markdown, **under 4 KB**, per the skeleton in §3 |

**The title rule is the one that was being broken.** In v1, `broad-fire-8531`
was titled *"…and hazard 15 dissolves"* for two days — a claim its own child
retracted. Anyone scanning titles got a false picture. **A title that asserts
something retracted is a defect, not history.** And v1's `holy-recipe-7414`
was titled *"the reward peak is not the best checkpoint in 2 of 2 fresh
seeds"*, which read as a general claim when the project-level answer was 2 of
3. v2 titles the node for the seeds it actually covers.

**The summary rule is new.** v1's capture node had a **1297-byte summary** —
longer than several node bodies, and rendered in full in every list view. A
summary that long is a second body.

### The 4 KB rule, and why it is a design constraint rather than style

`commit_node` publishes a **full snapshot, not a diff**, and requires a stage
lease that is **~60 s from acquire**. So the budget between `acquire` and
`commit` is exactly how long it takes to emit the entire body.

**A node too long to re-emit inside 60 s is a node that cannot be corrected.**
v1 had **11 of 19 bodies over 4 KB**, up to 9782 B — and three of them had to
be condensed under time pressure before their retraction banners could land.

**v2 has 0 of 37 over 4 KB.** Keep it that way. The node carries the claim,
the headline table and the pointers; the repo carries the write-up.

### `repo_context`, and the one thing the fork lost

Six keys, every time, `null` explicitly rather than omitted:

```
repo_url                git@github.com:theo-kirby/cdx-rl.git
branch_name             main
head_commit_sha         <git rev-parse HEAD>
origin_host             sb1x
updated_by              theo@quarry.capital
external_transcript_ref null, or a path/URL
```

`head_commit_sha` is **the commit whose tree matches the claim** — commit the
evidence to git *first*, then write the node.

> **`export_subgraph` does not carry `repo_url` / `branch_name` /
> `head_commit_sha`.** The v2 fork therefore has them empty on all 37 imported
> nodes. This costs almost nothing, because `get_node` never echoed those
> fields in any projection — their only use is filtering `list_nodes` by repo,
> and there is one repo. **But it is why every node carries a `## Provenance`
> section**, and why **nodes created from here on must use `commit_new_node`**,
> which does set them.

---

## 3. The body skeleton

Structure is all there is, so use it consistently. **v1 declared four
templates and adherence decayed monotonically** — the `## Capability sweep`
heading appeared in **zero** nodes, the decision template in **zero of one**
decision node, and `## Open` was present in the six oldest nodes, then renamed
ad hoc ("What to do next", "What runs next"), then dropped.

**Four skeletons. `## Provenance` and `## Open` are mandatory in all four.**

**Protocol** (`type/protocol`, written or transcribed *before* dispatch)

```markdown
## Question          one sentence, phrased so both answers are interesting
## Metric            named, defined, and why THIS one. Before dispatch (ADR-097)
## Mechanism         script, digest, actuator limit — hardware or mechanism?
## Budget and stopping rule    iterations, wall time, when to stop
## Pass criteria     written before the run
## Provenance
## Open
```

**Result** (`type/empirical`)

```markdown
## Claim             the one sentence a reader should leave with
## What happened     the headline table, with the number
## What it means
## What it does not mean      ← never omit this one
## Provenance        commit, host, run dir, GPU-hours
## Open
```

**Insight** (`type/insight`)

```markdown
## Observation
## Why it matters
## Evidence
## Provenance
## Open
```

**Decision** (`type/decision`)

```markdown
## Decision
## Context
## Alternatives considered
## Consequences
## Provenance
## Open
```

### `## Open` is the highest-value field and it is the one that rots

It is what a fresh agent reads to answer *"what now?"*. **Never rename it, and
never write "nothing" when you mean "nothing I thought about."** Legitimate
contents include *"no artifacts on this node, and here is where the evidence
actually lives"* — several v2 nodes say exactly that, which is honest and
actionable where silence was neither.

### `## Provenance` states GPU cost

**Every result node names its GPU-hours and its run directory.** v1 recorded
cost nowhere — not a tag, not a field, not an artifact — and `CLAUDE.md`'s
prose ledger drifted to ~27.5 h against a measured **39.66 h** in `jobs/`,
omitting three runs. Cost in the body is the cheap half of the fix; the
`platform` lane's GPU-ledger node is the other half.

---

## 4. Graph shape — four lanes, and protocol → result → replication

**A node hangs off what it is a consequence of, never off the root.** The root
carries only the four lane nodes and the charter.

### The four lanes

v1 was a **caterpillar**: one 9-node spine with short legs, and three of its
five root children were leaves. Four genuinely independent workstreams were
collapsed into one lineage, and two of them were simply absent — six of seven
drivers had no node, and **five merged Cadex PRs existed only as prose inside
an unrelated node**.

Flywheel's own contract says to avoid root-only branching *"unless work items
are truly independent."* These are.

| lane | holds | spends GPU? |
|---|---|---|
| **research** | the question: can the biped stand, is the policy buildable | yes — only this one |
| **instrument** | the drivers, and the measurement errors they exist to stop | no |
| **substrate** | Cadex itself — wishlist, merged PRs, pins, traps | no |
| **platform** | the boxes, the GPU ledger, environment reproducibility | no |

A node belongs to the lane that owns its **subject**, not the lane that
happened to discover it. The torque-instrument defect was found while scoring
003; it lives in `instrument`, because the next driver written can repeat it.

### Inside the research lane: protocol → result → replication

```
PROTOCOL 00N          type/protocol, status/planned    ← written BEFORE dispatch
  └── result 00N      type/empirical, status/provisional
      └── replication type/empirical  → graduates the parent to status/measured
```

**The protocol node is the change that matters most.** ADR-097 — *state the
metric before dispatch* — is this project's central discipline, and in v1 the
graph could not express it: `status/planned` was defined on day one and
assigned to **zero nodes for four days**. Every node was retrospective.

With protocol nodes, **"what is in flight" becomes the query**
`type/protocol AND status/planned`, and a finished run with no result node is
visible instead of invisible.

**Honesty rule for transcribed protocols.** v2's five protocol nodes were
authored on 2026-08-06 from experiment READMEs that were genuinely written
before dispatch. Each one **says so in a blockquote at the top**, naming the
file and quoting its own pre-registration header. **Never write a protocol
node dated before it existed.** A transcription with stated provenance is
honest; a backdated plan is not.

### Three patterns that earned their place

**1. A replication is a child that judges its parent.** Four instances now.
**Tag the parent `status/provisional` when you write it, and graduate it to
`status/measured` when the replication lands.**

**2. Multi-parent when one measurement bears on two nodes.** `lucky-tooth-6594`
confirms 004 and refutes part of 005 — one run, so **one node with two
parents**. Splitting it would have implied two runs. A result that bears on
two nodes must be reachable from either.

**3. A run that never happened still gets a node** if it produced a finding.
`frosty-hat-9494` cost **zero GPU-hours** and its veto is real evidence — and
it went on to be retracted twice, which is the graph working.

### Two shapes to avoid

* **A chronological edge.** In v1 the replay node hung off the capture node
  because it happened next, not because it followed from it. If the only
  sentence you can write for the edge is *"and then we did this"*, the parent
  is wrong.
* **A tooling node on the experiment spine.** Instrument work goes in the
  instrument lane, with a **second parent** into the experiment it bears on if
  it genuinely bears on one.

---

## 5. Tags — the vocabulary

Tags are defined on the **root** and are graph-wide. `tag_ids` is the per-node
assignment. **Assignment needs no stage lease** — only `expected_revision`.

| namespace | values | means |
|---|---|---|
| `type/` | `protocol`, `empirical`, `insight`, `decision` | what kind of claim |
| `status/` | `planned`, `provisional`, `measured`, `resolved`, `superseded` | where it stands |
| `exp/` | `000`…`005` | which experiment — makes "everything 004 touched" a query |
| `mechanism/` | `pendulum`, `stand-biped` | which machine |
| `hazard/` | `bracing`, `peak-regress`, `out-of-range`, `action-space` | which `MUJOCO.md` hazard |
| `criterion/` | `1`, `3`, `4`, `5` | which success criterion this bears on |

`type/protocol`, `status/resolved`, `exp/` and `criterion/` are **new in v2**.
`hazard/*` and `exp/*` are the two that earn their keep: they make *"show me
everything where the mechanism turned out to be the limit"* and *"show me
everything 004 touched"* queries rather than memory.

**There is no `task/` namespace, deliberately.** v1 declared one and there has
only ever been one task, so it would partition nothing. Add it the day a second
task exists, not before — an unassigned tag is the defect v1 had.

**`mechanism/` goes on a node whose SUBJECT is the machine**, not on every node
that happened to use it. Otherwise it lands on nearly every research node and
stops being a filter.

> ### State as of 2026-08-06: 25 tags created, all 37 nodes assigned
>
> **107 assignments over 32 nodes.** The five without tags are the **root and
> the four lane nodes**, which are navigation rather than claims.
>
> **`status/planned` is defined and assigned to nothing, and that is the
> answer, not a gap** — `type/protocol AND status/planned` is the in-flight
> query, and nothing is in flight. All five protocol nodes are
> `status/resolved` because every one has its result node. **The first node
> written before the next dispatch takes `status/planned`, and that is the
> whole point of the shape.**
>
> Ten tag ids worth having to hand: `type/protocol` `tag-0e83ecde05b4`,
> `type/empirical` `tag-f346bf39f8b2`, `type/insight` `tag-1e073cba80d5`,
> `type/decision` `tag-d9b8398d099c`, `status/planned` `tag-a56c74ef4bb4`,
> `status/provisional` `tag-fd88c4893dc4`, `status/measured` `tag-16c006bf9f86`,
> `status/resolved` `tag-afce4fb41a3d`, `status/superseded` `tag-180b590a48f6`,
> `hazard/bracing` `tag-66f9b8e93d5d`. The rest come back on any `get_node`.
>
> **Two mechanical notes, both measured while doing this.** `create_node_tag`
> takes the **graph** revision and bumps it by one per call, so creations are
> strictly serial. `set_node_tag_assignments` takes the **node's own** revision
> and touches nothing else, so assignments **parallelise safely** — the whole
> assignment pass ran in batches of four with no conflict. A node's revision
> is *also* moved by tag creation on the root, so **read revisions after the
> last `create_node_tag`, not before.**
>
> The v1 defect this closes: `status/planned` was assigned to **zero** nodes
> for four days and four of nineteen nodes carried **no tags at all**,
> including the one that retracted the project's headline.

### `status/` carries real weight — use it precisely

| | |
|---|---|
| `planned` | pre-registered, **not yet run**. The in-flight query. |
| **`provisional`** | **measured, but n=1 in TRAINING seeds.** The claim is a hypothesis. |
| `measured` | replicated across ≥2 training seeds, or not a seed-dependent claim |
| `resolved` | a protocol whose result node exists |
| **`superseded`** | **contains a retracted claim. Read its banner before quoting.** |

**`provisional` is the one that earns its keep**, because one-seed claims that
don't replicate is this project's single most repeated failure — 002 (2 of 3),
003's hazard 15 (retracted), 005's mechanism (retracted). In v1 it was applied
to **two** nodes, one of which also carried `status/measured` — a
contradiction the vocabulary allowed and nobody caught.

**`superseded` composes with `measured`/`provisional`** — a node can be a solid
two-seed measurement whose *interpretation* was retracted. Tag both.
**`measured` and `provisional` are mutually exclusive.** Pick one.

**Every substantive node gets `type/` and `status/`.** In v1, **four nodes
carried zero tags**, and one of them — `broken-cloud-4296`, which retracted the
project's headline and carried the p = 0.0391 headroom result — was untagged,
unevidenced and at revision 1: never touched after creation.

### Distinguish the two kinds of seed, always

* **training seed** — which policy you got. Drives replication. This is what
  `status/provisional` counts.
* **evaluation seed** — which scenario you played it on. Drives the *n* in
  "15/24".

In artifact metadata use **`seed_trained`** and **`seeds_eval`**, never bare
`seed`/`seeds`. In prose write "24 evaluation seeds" and "seed 1" in full.
v1 declared this rule and then left three nodes violating it, written before
the rule and never retrofitted; v2 corrected them on import.

---

## 6. Artifacts

Publish evidence **before** the completed commit. Batch related uploads: one
`finalize` bumps the revision once.

```
prepare_artifact_uploads  →  PUT raw bytes (expect 202)  →  finalize_artifact_uploads
```

| what | type |
|---|---|
| `progress.json`, a driver's `--json` envelope | `json` |
| a `{columns, rows}` table | `table` |
| the MJCF | `text` |
| **a `.cxpolicy`** | **`binary`** — the server refuses `cadex-policy-v1` for `checkpoint` |
| a plot | `image` |
| **an `.mp4`** | **`binary`**, `media_type: video/mp4` — no video type in the contract |

**The type is validated against the bytes at the PUT**, returning 422, and a
batch with any unfilled slot cannot be finalized — one wrong type wastes the
whole batch. `max_upload_bytes` is **10 MiB**; our checkpoints are 88–451 KB.

### Every node has evidence, or says in `## Open` why it does not

v1 had **8 of 19 nodes with zero artifacts**, including three that were
load-bearing: the four-seed retraction (whose "Artifacts and code" section was
a *text list of repo paths*), the 002 seed-2 node (4.13 GPU-hours of evidence,
none attached), and the sb9x characterisation.

Lane nodes, protocol nodes and register nodes legitimately have none. **A
result node with none is a defect**, and v2 names each remaining instance in
its own `## Open` rather than leaving it silent.

### Titles and notes

**Title**: `<subject> — <what it is>`, recognisable in a flat list.
v1 used four different conventions at once.

**Note**: says **what the numbers mean**, not what the file is — and it is
**per artifact**. v1's replay node carried *the same 753-byte note on four
different artifacts*, and the same 917-byte note on four more; only
`metadata.role` disambiguated them.

**Metadata** carries the machine-readable provenance. Required where
applicable: `sha256`, `task_sha256`, `model_sha256`, `trainer_sha256`,
`head_commit_sha`, `driver`, `run`, `seed_trained`, `seeds_eval`.
v1 left **21 of 55 artifacts with `metadata: {}`** — the discipline arrived at
node 12 and was never backfilled.

### Never hand-type a digest

Twice now a published digest was wrong: once truncated to 12 characters with a
guessed byte count, and once with **the first 12 characters right and the
remaining 52 invented**, because 12 is what the driver prints. **A short
prefix is more dangerous than an obvious blank — it looks like a digest.**

**Copy the JSON a driver or a ledger wrote. Do not read a number off a table
and retype it.** `replay/ledger.json` and `video/ledger.json` exist so that no
field is ever typed.

### The MP4 obligation

**Every MP4 `harness capture` records becomes the artifact of a node** — a
checked obligation, because `video/` is gitignored and an unpublished clip is
evidence on exactly one disk.

```bash
uv run python -m harness capture --pending            # what is owed, with the payload
uv run python -m harness capture --mark-published <node_id>
```

**A re-render under the same filename is new bytes and owes a new artifact**;
the ledger clears the node id when the digest changes rather than claiming the
graph holds bytes it does not.

**The driver cannot publish for itself**, and the reason is measured: cdx-rl's
nodes are owned by the **MCP identity**, the CLI holds a different account and
returns 403, and a Python subprocess has no MCP.

---

## 7. Retractions — how we handle being wrong

We do **not** delete and we do **not** silently edit.

1. **The new measurement gets its own node**, as a child of what it retracts.
2. **The retracted node gets a `> ## ⚠ RETRACTED IN PART` banner at the very
   top of its content**, naming the retracting node and splitting explicitly
   into *what stands* and *what is retracted*.
3. **`[RETRACTED IN PART]` in the title**, and the retraction in the `summary`.
4. **Tag `status/superseded`.**

The body stays otherwise intact. Being wrong in public, with the reasoning
visible, is how the generalisable lesson gets extracted — **003's retraction is
what made 004 ask the right question**, and 004 is the experiment that produced
the first buildable policy.

**A node can be retracted more than once.** `frosty-hat-9494` carries two
banners: its mechanism went first, to a seed replication, and its last
surviving premise went second, to the measurement it had vetoed. Number them.

> v1 also prescribed an in-place marker at the retracted section, on the theory
> that a reader might scroll past the banner. With bodies under 4 KB the banner
> is never far from the claim, and the duplicate wording cost budget that was
> already tight. **Banner, title, summary, tag — four places, not five.**

---

## 8. Editing an existing node — the lease budget

`commit_node` **publishes a full snapshot, not a diff.** The staged payload
*is* the new body. And it requires a stage lease of **~60 s from acquire**.

> **`flywheel.md` said "acquire → heartbeat → commit" and that advice is
> wrong.** Measured: `acquire` returned `expires_at` 60 s out; `heartbeat`
> moved it to **+15 s from now**, not a fresh 60. The heartbeat *shortened* the
> window. Two commits failed with `409 stage lease missing or expired` before
> this was understood.

**The working pattern:**

```
read the node (core or full)  →  compose the ENTIRE payload  →  acquire  →  commit
```

Acquire as late as possible.

### Other mechanics that bite

* **`projection: "topology"` returns `content: ""` and `summary: ""`.** Never
  stage a commit from a topology read — you would publish an empty body.
* **`commit_new_node` bumps every parent's revision.** A cached revision is
  stale; `acquire_stage_lease` 409s with *"stale committed revision"*. Re-read.
* **`add_parent` takes two revisions and both must be current.** Chained edge
  surgery will 409 on the second call — read the number out of the error and
  retry; the error states the current value.
* **Tag assignment needs no lease** — only the node's own `expected_revision`.
  It is the cheap, safe way to flag a node you cannot currently rewrite.
* **`set_node_tag_assignments` replaces the whole list.** Send every tag.
* **`expected_revision` differs between the tag calls.** `create_node_tag`
  wants the *graph* revision (from the root); `set_node_tag_assignments` wants
  the *node's own*.
* **A tag create returns the ROOT NODE, not the tag.** Confirm by re-reading
  the root's `graph_tags` — a caller who looks for `tag_id` in the response
  concludes it failed and retries, which is how three duplicate tags once got
  created.

---

## 9. Forking the graph

There is no fork verb. The mechanism is
`export_subgraph(include_descendants: true)` → `import_subgraph`, which mints
new node IDs and maps every edge. **Measured 2026-08-06, on a throwaway node
that was then deleted:**

* **Artifact blobs are aliased, not copied.** The imported artifact keeps the
  **source `artifact_id`** and a `storage_path` pointing into the *original*
  node's directory. It resolves and downloads fine, and deleting the fork does
  **not** harm the original. **But the fork's evidence is a pointer into the
  old graph's storage, so the old graph must never be deleted.**
* **`tag_ids` survive; the tags do not.** The new root comes back with
  `graph_tags: []`, so carried tag ids dangle. Recreate and reassign.
* **`repo_context` is not carried** (§2).
* **Invented node IDs work.** Any UUID-shaped string in the payload is mapped
  to a fresh id, so new nodes can be authored directly into an import
  alongside forked ones, edges and all. Multi-parent edges survive.
* **A payload with no parentless-node edge becomes its own root.** To graft a
  cluster into an existing graph, import it and then `add_parent` its root.

**Import in clusters of 1–5 nodes.** A single large payload is one malformed
character away from a total loss, and the tool rejects unparseable JSON as a
whole.

---

## 10. Reading

* **`flywheel_get_graph` and `flywheel_list_nodes` will blow the output cap.**
  `get_graph` returned 864 KB and spilled to a file. Use
  `flywheel_summarize_node_tree` for a fast overview and
  `flywheel_export_subgraph` + `jq` when you need every field.
* **`summarize_node_tree` renders a DAG as a spanning tree and it will
  mislead you.** It picks a shortest path, so a multi-parent node appears under
  one arbitrary parent and the causal shape looks wrong. **Read
  `incoming_ids`/`outgoing_ids` from an export before concluding anything about
  topology** — the v1 review nearly recorded a defect that was an artefact of
  this render.
* `projection` is `core` < `topology` < `full`. Ask for `core` unless you need
  artifacts.
* `get_node` does not guarantee complete relationship arrays; use
  `get_node_children` / `get_node_parents` for real traversal.

---

## 11. Compute

**We do not use Flywheel's managed compute.** This box has an RTX 5090 and runs
are local. The graph still records the run — where it ran, on what, at which
commit, and **how many GPU-hours** — since without an `execution_id` there is
nothing structural to say it. See [`cloud.md`](cloud.md).

---

## 12. The checklist

Before dispatching anything that costs GPU time:

- [ ] A **protocol node** exists, `type/protocol` + `status/planned`, with the
      metric and pass criteria written **before** the run

Before writing a result node:

- [ ] Evidence committed to git; `head_commit_sha` is that commit
- [ ] Parent is *what this is a consequence of* — and a second parent if it
      bears on two nodes
- [ ] Title true standalone; summary carries the headline **and** its caveat,
      under ~500 characters
- [ ] **Body under 4 KB**, using §3's skeleton
- [ ] `## Provenance` names the commit, host, run directory and **GPU-hours**
- [ ] `## Open` is present and says something real
- [ ] `type/` and `status/` assigned — **always both** — plus `exp/`,
      `hazard/`, `criterion/` and `mechanism/` where they apply
- [ ] **`status/provisional` if n=1 in training seeds** — and the parent
      protocol moved to `status/resolved`
- [ ] Artifacts batched; `.cxpolicy` and `.mp4` as `binary`; **notes per
      artifact**, saying what the numbers mean; digests **copied, never typed**
- [ ] `harness capture --pending` is empty
- [ ] If it retracts a parent: banner, title, summary, `status/superseded`

---

## 13. The v2 node table

Root **`steep-pine-4944`** = `91c66efb-66f3-48aa-b7c1-14bfe00bb09f`.
37 nodes, 55 artifacts, 40 edges, **0 bodies over 4 KB**, 4 multi-parent nodes.
v1 (`rapid-bar-6214`) is frozen: 19 nodes, 21 edges, 11 bodies over 4 KB.

**Tags below are assigned on the graph as of 2026-08-06** — 107 assignments
over 32 nodes; the root and the four lane nodes carry none by design. `P` =
parents.

| lane | slug | node_id | tags to assign |
|---|---|---|---|
| — | `steep-pine-4944` | `91c66efb-66f3-48aa-b7c1-14bfe00bb09f` | *(root — none)* |
| — | `small-hall-5435` | `b6061a79-a50c-4ecc-ac7d-fdfeb8d1ad9c` | `type/insight` |
| — | `raspy-pond-9300` | `6601df73-1838-4eaf-ba58-b5b00f4eeb58` | *(lane)* |
| — | `bold-bonus-7821` | `dbc7c94e-b810-406d-98ed-77919a1cd5ee` | *(lane)* |
| — | `dawn-union-2199` | `cccd37ab-b79a-487f-a082-68c4c7cfd906` | *(lane)* |
| — | `raspy-shadow-9896` | `d3b06fc4-d9d8-43e3-9ef5-711d1f6762cc` | *(lane)* |
| plat | `wispy-cell-2014` | `c28d9b53-9f47-4dac-8eea-82f2e3265b3f` | `type/insight`, `status/measured` |
| plat | `gentle-math-3665` | `3270c2b6-fba9-4f9f-8296-042a7d59e99d` | `type/empirical`, `status/measured` |
| plat | `small-pine-9389` | `8f936dcc-a999-4dce-af8d-7886af5adf34` | `type/decision`, `status/measured` |
| plat | `cool-grass-5029` | `3bf66c04-0e4f-4074-9e50-1032c975b8f1` | `type/empirical`, `status/measured` |
| sub | `polished-bar-9134` | `ab71fab3-a36b-415b-8e2a-b6e49efe856b` | `type/decision`, `status/measured` |
| sub | `spring-field-7039` | `fc7a91bd-7e3e-465d-ace7-43929e811137` | `type/empirical`, `status/measured` |
| sub | `plain-sun-6624` | `656c94ef-0e10-47b0-8256-bf646ce50e61` | `type/insight`, `status/measured` |
| sub | `tiny-rice-4100` | `7a4de365-d217-4a25-b435-68295597fcaa` | `type/empirical`, `status/measured` |
| inst | `late-moon-2834` | `28e096f8-7d4b-490b-bec4-abcfe9d592bd` | `type/empirical`, `status/measured`, `exp/000`, `mechanism/pendulum`, `criterion/1` |
| inst | `royal-brook-3544` | `34e910d1-d043-4fde-b461-a066bf66a222` | `type/insight`, `status/measured` |
| inst | `purple-glade-7987` | `ca95b596-1280-4428-8b0f-fb39124fa251` | `type/insight`, `status/measured`, `hazard/bracing` |
| inst | `purple-frog-3659` | `c1650170-fc98-4dff-acfa-c8c9185609e8` | `type/insight`, `status/measured` |
| inst | `patient-hall-2416` | `c70481fc-77d0-453f-855e-f9704922c05e` | `type/insight`, `status/measured`, `exp/004`, `hazard/bracing` — P: `bold-lab-1179`, `royal-brook-3544` |
| res | `flat-snow-8336` | `d40c631c-92ce-4ce0-840a-92002e43e151` | `type/insight`, `status/measured`, `mechanism/stand-biped` |
| res | `bitter-cake-2117` | `36e51671-3159-4465-8061-c96ea969fb49` | `type/empirical`, `status/measured`, `hazard/peak-regress` |
| res | `delicate-unit-4684` | `d79b8d46-0204-4d8a-a209-c715b625d4fa` | `type/protocol`, `status/resolved`, `exp/001` |
| res | `patient-queen-1723` | `2af6c33b-867e-4663-957d-d529fcd37864` | `type/empirical`, `status/provisional`, `exp/001`, `hazard/peak-regress` |
| res | `royal-shadow-3079` | `67cfce4c-2faa-4105-9c52-1043941a1dcc` | `type/empirical`, `status/provisional`, `exp/001`, `hazard/bracing`, `hazard/out-of-range`, `criterion/3` |
| res | `hidden-disk-0740` | `de339057-7048-44b9-9a0d-a35c63d8d4b7` | `type/protocol`, `status/resolved`, `exp/002` |
| res | `spring-brook-0043` | `3cc3223b-0a6c-4d41-a37d-832065e80d15` | `type/empirical`, `status/measured`, `exp/002`, `hazard/peak-regress`, `hazard/bracing` |
| res | `icy-dust-2040` | `861bdc47-066f-429d-8a69-b31de9d35402` | `type/empirical`, `status/measured`, `exp/002` |
| res | `dark-frog-3380` | `c4b128a2-a715-4031-9a56-5254a6164eed` | `type/protocol`, `status/resolved`, `exp/003` — P: `spring-brook-0043`, `royal-shadow-3079` |
| res | `yellow-thunder-9504` | `b2af2c7a-5953-418c-a4f9-615491935a60` | `type/empirical`, `status/provisional`, `status/superseded`, `exp/003`, `hazard/action-space`, `hazard/bracing` |
| res | `billowing-truth-5245` | `040f0954-b583-4237-b329-594374b797c3` | `type/empirical`, `status/measured`, `exp/003`, `hazard/bracing`, `hazard/action-space` |
| res | `black-frog-9747` | `154d3610-2dae-443e-9fe9-9fbd822224b9` | `type/protocol`, `status/resolved`, `exp/004` |
| res | `bold-lab-1179` | `0be02d09-d1f7-4aff-bbe0-cb94ef08ceb2` | `type/empirical`, `status/measured`, `exp/004`, `hazard/bracing`, `criterion/5` |
| res | `frosty-hat-9494` | `f5b83892-ece9-49ce-bad1-1536a1508531` | `type/decision`, `type/empirical`, `status/superseded`, `exp/005`, `hazard/bracing` |
| res | `lucky-tooth-6594` | `be07a0db-afcf-40d9-9ae2-b1a209c111b1` | `type/empirical`, `status/measured`, `exp/004`, `exp/005`, `hazard/bracing`, `criterion/4` — P: `frosty-hat-9494`, `bold-lab-1179` |
| res | `rapid-grass-1358` | `ab49f364-1681-47a3-a119-9125fcb4d572` | `type/protocol`, `status/resolved`, `exp/005` |
| res | `young-bush-8065` | `b7a4d827-2413-40cf-95c9-2a295a4eb74b` | `type/empirical`, `status/provisional`, `exp/005`, `hazard/bracing`, `criterion/5` |
| res | `odd-dust-3102` | `a940b58e-696d-4084-a416-3691d04c7537` | `type/insight`, `status/measured`, `criterion/5` — P: `spring-field-7039`, `bold-lab-1179` |

### Evidence still owed, in priority order

1. **`billowing-truth-5245`** (003 at four seeds) — zero artifacts, and it is the
   node that retracts the project's headline. Files exist:
   `experiments/003-position-action-space/results/{seed1,seed2,seed3}-12-seeds.json`,
   `seed0-12-seeds-rescored.json`, `three-seed-tiebreak-24.json`,
   `hazard15-three-seeds.json`.
2. **`icy-dust-2040`** (002 seed 2) — 4.13 GPU-hours, zero artifacts.
3. **`young-bush-8065`** (005-ceiling) — 4.75 GPU-hours, zero artifacts; its
   policy is on the graph only as the replay set's `clamp25 — policy`.
4. **`cool-grass-5029`** (GPU ledger) — the sum is hand-run and will drift again
   until a driver writes it.
