# How cdx-rl uses Flywheel

**This is the normative document.** It says what we do and why. Its companion
[`flywheel.md`](flywheel.md) is descriptive — the API surface, the measured
traps, and a render of the graph as it currently stands. When they disagree,
this file wins on *policy* and `flywheel.md` wins on *what the server does*.

Written 2026-08-05, after the 003/004/005 chain and the seed-1 replication
made the gaps obvious. Every mechanic below was executed, not assumed.

---

## 1. What the graph is for

Success criterion 4: **a fresh agent can rebuild the picture from the graph.**
Not from a conversation, not from a directory listing — from nodes, their
artifacts, and the edges between them.

That single sentence decides most of the arguments below. If a convention
makes the graph readable to someone who arrives with no context, it stays. If
it only makes it tidy, it does not.

**The graph is not a copy of the repository.** The repo holds the full
experiment write-up, the drivers, the bundles and the run directories. The
graph holds the *claim*, the evidence for it, and the edge to what it changed.
A node that duplicates a README is a maintenance liability — see §6, where the
stage lease turns long bodies into a practical problem, not just a stylistic
one.

## 2. Node structure

There are no typed nodes in Flywheel. Type is carried by tags (§3).

**Three fields matter and they have different jobs:**

| field | job | rule |
|---|---|---|
| `title` | the index entry — it is what every tree render, list and search shows | must be true *on its own*, with no child node required to correct it |
| `summary` | the abstract — read by anyone deciding whether to open the node | must carry the headline number and its caveat |
| `content` | the record | Markdown; templates in `flywheel.md` §4 |

**The title rule is the one that was being broken.** For two days
`broad-fire-8531` was titled *"…and hazard 15 dissolves"* — a claim its own
child retracted. Anyone scanning titles got a false picture, and only someone
who walked to the leaf found out. **A title that asserts something retracted is
a defect, not history.**

### `repo_context` — all six keys, every time

```
repo_url                git@github.com:theo-kirby/cdx-rl.git
branch_name             main
head_commit_sha         <git rev-parse HEAD>
origin_host             sb1x
updated_by              theo@quarry.capital
external_transcript_ref null, or a path/URL
```

Pass `null` explicitly rather than omitting a key. **`head_commit_sha` is the
commit whose tree matches the claim** — commit the evidence to git *first*,
then write the node. If the measurement was taken at an earlier commit than the
node is written at, say both in the content; the field has nowhere to put it.

**Restate the commit in the Markdown.** `get_node` does not echo `repo_url`,
`branch_name` or `head_commit_sha` back in any projection. The structured
field is a filter key for `list_nodes`; the content is what a human reads.

## 3. Tags — the vocabulary

Tags are defined on the **root** and are graph-wide. `tag_ids` is the per-node
assignment. Assignment needs **no stage lease** — only `expected_revision`.

| namespace | values | means |
|---|---|---|
| `type/` | `insight`, `empirical`, `decision` | what kind of claim |
| `mechanism/` | `pendulum`, `stand-biped`, … | which machine |
| `task/` | `stand`, `swing`, `recover`, … | which question of it |
| `status/` | `planned`, `provisional`, `measured`, `superseded` | where it stands |
| `hazard/` | `bracing`, `peak-regress`, `out-of-range`, `action-space`, … | which `MUJOCO.md` hazard it is about |

### The `status/` values carry real weight — use them precisely

| | |
|---|---|
| `planned` | written before dispatch; no measurement yet |
| **`provisional`** | **measured, but n=1 in TRAINING seeds.** The claim is a hypothesis. |
| `measured` | replicated across ≥2 training seeds, or not a seed-dependent claim at all |
| **`superseded`** | **the node contains a retracted claim. Read its banner before quoting anything from it.** |

**`provisional` is the one that earns its keep**, because one-seed claims that
don't replicate is this project's single most repeated failure — 002 (2 of 3),
003's hazard 15 (retracted), 005's mechanism (retracted). It was defined on the
root and applied to *nothing* for three days while `broad-fire-8531` sat tagged
`measured` with the words "**One seed.**" in its own body.

`superseded` and `measured`/`provisional` are **not exclusive** — a node can be
a solid two-seed measurement whose *interpretation* was retracted. Tag both.

### Distinguish the two kinds of seed, always

They are different things and the graph must not blur them:

* **training seed** — which policy you got. Drives replication. This is what
  `status/provisional` counts.
* **evaluation seed** — which scenario you played it on. Drives the *n* in
  "15/24".

In artifact metadata use **`seed_trained`** and **`seeds_eval`**, never bare
`seed`/`seeds`. In prose, write "24 evaluation seeds" and "seed 1" in full.

## 4. Graph shape

**A node hangs off what it is a consequence of, never off the root.** The root
is reserved for things that are nobody's consequence — `000` (the floor) and
the pre-cdx-rl history node.

The main chain reads as **claim → isolation → qualification**:

```
mute-shadow-9769   the bracing is the resting posture
  └── broad-fire-8531        003 — the action space (RETRACTED IN PART)
      └── broken-cloud-4296  003 at four seeds — hazard 15 never dissolved
          └── white-cloud-2565      004 — it was the COMMAND RANGE
              └── small-recipe-2040     005 — the gate's veto (RETRACTED IN PART)
                  └── solitary-salad-0490   seed 1: 004 replicates, 005's mechanism does not
```

### Three established patterns

**1. A seed replication is a child node that judges its parent.** Three
instances now — 002 → `spring-unit-9051`, 003 → `broken-cloud-4296`, 004/005 →
`solitary-salad-0490`. **Tag the parent `status/provisional` when you write it,
and graduate it to `measured` when the replication lands.**

**2. Multi-parent when one measurement bears on two nodes.** `broad-fire-8531`
answers both 001 Phase B and 002. `solitary-salad-0490` **confirms** 004 and
**refutes part of** 005 — one run, so one node, with two parents. Splitting it
would have implied two runs. A result that bears on two nodes must be reachable
from either; `add_parent` is one call and is canonical the moment it succeeds.

**3. A run that never happened still gets a node** if it produced a finding.
`small-recipe-2040` cost zero GPU-hours and its veto is real evidence.

### Retractions — how we handle being wrong

We do **not** delete and we do **not** silently edit. The sequence is:

1. **The new measurement gets its own node**, as a child of what it retracts.
2. **The retracted node gets a `> ## ⚠ RETRACTED IN PART` banner at the very
   top of its content**, naming the retracting node, and splitting explicitly
   into *what stands* and *what is retracted*.
3. **An in-place marker on the retracted section**, so a reader who scrolls
   past the banner still hits it.
4. **`[RETRACTED IN PART]` in the title**, and the retraction in the `summary`.
5. **Tag `status/superseded`.**

The body stays otherwise intact. The point is that being wrong in public, with
the reasoning visible, is how the generalisable lesson gets extracted — 003's
retraction is what made 004 ask the right question.

## 5. Artifacts

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
| **an `.mp4` from `harness capture`** | **`binary`**, `media_type: video/mp4` — there is no video type in the contract |

### Every MP4 `harness capture` records becomes the artifact of a node

Not a habit — a checked obligation, because a clip that only exists in
`video/` is invisible to criterion 1's *"a graph node with artifacts"* and to
criterion 4 entirely. `video/` is gitignored, so an unpublished MP4 is
evidence that exists on exactly one disk.

**The driver cannot publish for itself**, and the reason is measured rather
than laziness: cdx-rl's nodes are owned by the **MCP identity**, the CLI
holds a different account and returns 403 against them (`flywheel.md` §5),
and a Python subprocess has no MCP. So the driver records the obligation and
refuses to let it go quiet:

```bash
uv run python -m harness capture --pending          # what is owed, with the payload
#   → items[] for flywheel_prepare_artifact_uploads, ready to send
uv run python -m harness capture --mark-published <node_id>
```

`video/ledger.json` holds one entry per MP4 with its `sha256`, its
`upload_item` and the node it landed on. Every capture prints the
outstanding count — on stderr under `--json`, so a machine-readable mode is
not a way for the debt to go unmentioned. **A re-render under the same
filename is new bytes and owes a new artifact**: the ledger clears the node
id when the digest changes, rather than claiming the graph holds bytes it
does not.

Notes are generated from the sidecar and state what the numbers mean — how
many seeds survived, the per-seed duty above 90 % of forcerange, and that
the shapes are `mjVIS_INERTIA` boxes rather than CAD solids. `--note` adds
the part no driver can know: *why this policy*. Metadata carries
`seeds_eval` and `seed_trained` separately, per §3.

**The type is validated against the bytes at the PUT**, returning 422, and a
batch with any unfilled slot cannot be finalized — one wrong type wastes the
whole batch. An abandoned batch expires quietly and bumps nothing, which is the
cheap way out.

`max_upload_bytes` is **10 MiB**. Our checkpoints are 88–451 KB.

Every artifact needs a `title` a human would recognise in a list, and a `note`
that states **what the numbers mean**, not what the file is. Put digests
(`sha256`, `task_sha256`, `model_sha256`, `trainer_sha256`) in `metadata`.

**Write with the identity that owns the node.** Nodes created over MCP can only
be written over MCP; the CLI holds a different account and returns 403. Check
`flywheel_auth_status` against `flywheel auth:status` before assuming.

## 6. Editing an existing node — the lease budget

This is the section that cost the most to learn, on 2026-08-05.

`commit_node` **publishes a full snapshot, not a diff.** The staged payload
*is* the new body. And it requires a stage lease, which is **~60 s from
acquire**.

> **`flywheel.md` said "acquire → heartbeat → commit" and that advice is
> wrong.** Measured: `acquire` returned `expires_at` 60 s out; `heartbeat`
> moved it to **+15 s from now**, not a fresh 60. The heartbeat *shortened* the
> remaining window. Two commits failed with
> `409 stage lease missing or expired` before this was understood.

**The working pattern is:**

```
read the node (core or full)  →  compose the ENTIRE payload  →  acquire  →  commit
```

Acquire as late as possible. The gap between `acquire` and `commit` is exactly
how long it takes to emit the payload, and that is the whole budget.

**Consequence, and it is a design constraint not a workaround: keep node bodies
under about 4 KB.** A ~9 KB body cannot be re-emitted inside 60 s, which means
**a node too long to edit is a node that cannot be corrected.** 003 and 005
both had to be condensed before their retraction banners could be committed.

So: the node carries the claim, the key tables and the pointers. The repo
carries the full write-up. When condensing an existing node, **say so in the
body** and name where the full text lives — the Flywheel revision number and
the repo path at a commit. Prior revisions are retained, so condensing is
superseding, not destroying.

### Other mechanics that bite

* **`projection: "topology"` returns `content: ""` and `summary: ""`.** Never
  stage a commit from a topology read — you would publish an empty body.
  Use `core` or `full`.
* **`commit_new_node` bumps every parent's revision.** A revision you cached
  before creating a child is stale, and `acquire_stage_lease` will 409 with
  *"stale committed revision"*. Re-read.
* **Tag assignment needs no lease** — only the node's own `expected_revision`.
  It is the cheap, safe way to flag a node you cannot currently rewrite.
* **`set_node_tag_assignments` replaces the whole list.** Send every tag you
  want to keep.
* **`expected_revision` differs between the tag calls.** `create_node_tag`
  wants the *graph* revision (from the root); `set_node_tag_assignments` wants
  the *node's own*.
* **A tag create returns the ROOT NODE, not the tag.** Confirm by re-reading
  the root's `graph_tags` — a caller who looks for `tag_id` in the response
  concludes it failed and retries, which is how three duplicate tags once got
  created.

## 7. Reading

* **`flywheel_get_graph` and `flywheel_list_nodes` will blow the output cap.**
  `get_graph` returned 864 KB and spilled to a file. Use
  `flywheel_get_node_tree` for structure — it is small and well-shaped — and
  `jq` on the spill file when something does overflow.
* `projection` is `core` < `topology` < `full`. Ask for `core` unless you need
  artifacts.
* `get_node` does not guarantee complete relationship arrays; use
  `get_node_children` / `get_node_parents` for real traversal.

## 8. Compute

**We do not use Flywheel's managed compute.** This box has an RTX 5090 and runs
are local. The graph still records the run — say where it ran, on what, at
which commit, since without an `execution_id` there is nothing structural to
say it. See [`cloud.md`](cloud.md).

## 9. The checklist

Before writing a node:

- [ ] Evidence committed to git; `head_commit_sha` is that commit
- [ ] Parent is *what this is a consequence of* — and a second parent if it
      bears on two nodes
- [ ] Title is true standalone; summary carries the headline **and** its caveat
- [ ] Body under ~4 KB, pointing at the repo for the full write-up
- [ ] `type/`, `mechanism/`, `task/`, `status/`, `hazard/` all assigned
- [ ] **`status/provisional` if n=1 in training seeds**
- [ ] Artifacts batched, `.cxpolicy` and `.mp4` as `binary`, notes say what
      the numbers mean, digests in metadata
- [ ] `harness capture --pending` is empty — no MP4 is sitting on one disk
- [ ] If it retracts a parent: banner, in-place marker, title, summary, and
      `status/superseded` on the parent
