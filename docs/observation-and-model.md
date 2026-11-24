# Observations and the policy network

At every decision, the network receives what the acting seat can see and returns
two things: a probability for each legal action, and an estimate of who will win.
[The rules engine](engine.md#observations) decides what the seat can see, and
[How a manabot learns](learning.md) covers what training does with those
outputs. This guide is the network between them.

- [What the network sees](#what-the-network-sees)
- [What the network does](#what-the-network-does)
- [Sizes in the TLA pilot](#sizes-in-the-tla-pilot)
- [Policy inputs](#policy-inputs)
- [Whole-game decision memory](#whole-game-decision-memory)
- [State, records and history](#state-records-and-history)
- [What we know](#what-we-know)
- [Where the code lives](#where-the-code-lives)

![A viewer-safe observation becomes typed object embeddings, mixed by self-attention. Legal actions gather their focus objects to produce one logit each, and pooled objects produce the value estimate.](images/policy-network.svg)

## What the network sees

managym owns this boundary. It supplies viewer-safe state and the complete legal
decision before the network sees either, so nothing in the encoder or browser
filters hidden information.

| Input | What it carries |
| --- | --- |
| Players, visible cards, permanents | Typed objects. Permanent rows already include effective characteristics. |
| Local facts and links | Open-ended details such as counters, plus attachment and blocking links. |
| Current continuation | The pending decision: stack entries, targets, modes and suspended payments. |
| Shared effects | Each active effect once, with the objects it affects. |
| Known hand, library positions and disclosed decklists | What this seat is allowed to know. |
| Legal actions and focus objects | Every offered action and up to two objects it refers to. |
| Card definitions | Typed rules tokens derived from each card's native definition. |

Every mechanic the engine implements is visible here, in the LLM text
observation and in the learning tokens, in the same change that adds it. A
frozen input version never silently falls behind the rules.

## What the network does

Every observation batch is prepared once, on the host, before any forward:
`agent.prepare(obs)` checks shapes, references, programs and the legal-action
mask of the NumPy buffers, does every integer lookup and returns a `Prepared`
batch, the same one every [backend](../manabot/backends/README.md) reads. The
forward reads only a prepared batch on its device, raises `TypeError` for an
observation dictionary and does no validation or lookups of its own
([observation admission](../manabot/model/agent.md#observation-admission)).

1. **Embed.** Each visible object gets a typed projection. A card's rules tokens
   pass through a token embedding and a GRU, so the model can read what a card
   does as well as its name. Facts, links and effects add to the object they
   belong to. With `stream_norm`, every stream summed into an object arrives
   at [one scale](#one-scale-for-every-stream).
2. **Mix.** Self-attention lets objects inform each other, with padding masked.
   [Live-token attention](#live-token-attention) attends over valid
   objects only; it replaced the pooled token stream
   ([token-stream report](../experiments/reports/token_stream.md)).
3. **Score actions.** Each action gathers the post-attention embeddings of its
   focus objects, and a small head gives it one logit. With `focus_film`, the
   focus objects first scale the action's own features; with `policy_film`, a
   pooled game state then scales each action embedding. Illegal slots are
   masked, so every decision type shares one action head.
4. **Estimate value.** Objects are projected, pooled by a mean over valid
   objects (`masked_mean`, as in the TLA pilot; the historical default averages
   every slot) and read by the value head. The output is a scalar or loss/draw/win logits.
   `value_token` and `attention_readout` replace the mean with a learned summary
   ([readout comparison](../experiments/reports/value-readout-252.md)).

The card GRU encodes a card's rules program. Optional recent events are off by
default. [Whole-game decision memory](#whole-game-decision-memory) adds a
separate encoder over past opponent decisions, recomputed from the observation
at every forward rather than carried between calls.

### Scoring legal actions

One head scores every kind of decision, from casting and targeting to blocking
and paying costs. The action's own features are concatenated with the
post-attention embeddings of the objects it names.

From `_gather_informed_actions` in
[`manabot/backends/torch/policy.py`](../manabot/backends/torch/policy.py):

```python
focused = post_attention_objects.flatten(0, 1)[obs.focus]
focused = (focused * obs.focus_mask.unsqueeze(-1)).flatten(1)
if self.focus_film is not None:
    actions = actions * (1 + self.focus_film(focused))
return self.action_layer(torch.cat((actions, focused), dim=-1))
```

`obs` is a device `Prepared` batch. Its host preparation has already resolved
focus indexes and selected the legal action rows. `forward_distribution`
restores their scores to the published action capacity:

```python
scores = self.policy_head(informed_actions).squeeze(-1).float()
logits = scores.new_full(capacity, -1e8)
logits = logits.index_put((batches, slots), scores.to(logits.dtype))
```

- `max_focus_objects` sets the number of focus slots per action. It must be
  positive and defaults to two. An unused slot holds −1 and contributes zero.
- Masking to −1e8 gives an illegal slot zero probability, so the learner's
  ratios and KL terms sum over legal actions only.
- `max_actions` is a declared capacity. Overflow is an error by default; the
  TLA pilot explicitly keeps the first 64 legal actions, while the engine and
  Academy keep the complete set
  ([policy action capacity](../manabot/README.md#policy-action-capacity)).

### Scoring an offer on its focus

“Attack with this creature” and “leave it back” share their focus and differ
only in the `declared` flag. With `focus_film=True`, the bias-free linear
layer scales the offer features by the focused creature's embedding. Its
orthogonal initialization lets the creature affect that scaling from the
first decision. For an unfocused offer the embedding is zero, so its
multiplier is exactly 1 throughout training.

`policy_film` independently conditions the action head's output on the
pooled board. The baseline and Elemental self-play recipes enable focus
scoring. Checkpoints that omit `focus_film` use the additive head.

On the constructed attack decision, the test trains on 100 combinations of
power and toughness. It counts updates until the model assigns at least 90%
probability to the correct choice on every board (“attack iff power is at
least 3”). Across seeds 0, 1 and 2, the additive model took 102, 119 and 108
updates. Enabling `focus_film` took 88, 90 and 81, and enabling both
`focus_film` and `stream_norm` took 83, 91 and 80. The totals were 329, 259
and 254 updates respectively for this supervised decision.

### Estimating value

The value head reads a masked mean over the same post-attention objects, so
policy and value share one encoder and both losses train it
([what the gradient touches](learning.md#4-update)). A `categorical_wdl` head
returns loss, draw and win logits, and its value is P(win) − P(loss); a
`scalar` head returns the outcome directly ([value](learning.md#value)).

## Sizes in the TLA pilot

Each object has 64 channels. Two self-attention blocks, each with four heads, mix
up to 386 padded object slots: two players, 256 cards and 128 permanents (128
cards and 64 permanents per seat; a fresh `ObservationConfig` allots 60 and
40, for 202 slots). An
action concatenates its 64-channel projection with two 64-channel focus objects,
giving 192 channels, and passes through 192 → 64 → 64 to the policy head.

The model encodes each complete shared effect once and sums its vector into every
explicitly named affected object before attention. Duplicate memberships retain
multiplicity, and every affected object's loss reaches the shared encoder. Local
facts remain a separate sum. Neither path adds attention tokens.

Complete shared records retain duration, timestamp, controller and exact source
bindings that can change future behavior.

### One scale for every stream

An object token is a sum: a projection of its numeric row, the encoding of its
card definition and name, its links, its facts, the effects on it and the
shared context of the decision. Without normalization, whichever stream is
largest sets the scale of the token. With names on definitions
(`definition_names`) the name vector is about seven times the numeric row: on
the constructed attack decision a creature growing from 2/2 to 9/9 moves its
whole token by 0.16 to 0.25 of its size over four seeds, and a token creature,
which has no name, enters attention at under half a card's scale. With the
field the same move is 0.38 to 0.56. Without names the numeric row's stat
flags already carry the change (0.48 to 0.60 without the field, 0.36 to 0.53
with it).

```python
model = ModelConfig(stream_norm=True, ...)
```

`stream_norm` requires `attention_on=True` and selects one layout, with one rule: **everything summed into an
object token is either normalized to the common scale or starts at zero.**

| Stream | With `stream_norm` |
| --- | --- |
| Numeric projections (player, card, permanent, stack) | A `LayerNorm` after each projection |
| Card definitions, after names join | One `LayerNorm` over the definition table, so a named card and an unnamed token enter at the same scale; the absent row stays zero |
| Known hand, known library, decklists | Three streams, each pooled per player with its own `LayerNorm`; each exactly zero for a player whose source is empty |
| Shared context | One `LayerNorm`; exactly zero for a decision with no relations and no events |
| Links, stack sources, facts, shared effects | Not normalized: each ends in a zero-initialized layer and grows from nothing |
| Attention | Pre-normalization, `x + attention(norm(x))` then `x + mlp(norm(x))`, and one final `LayerNorm` (`ActorCritic.final_norm`) before the action and value heads |

Query and key sizes then stay bounded however many streams a later change
adds, and the heads read a normalized table. The attention layers keep their
parameter names; only the order of operations changes. The raw definition
table that scores naming candidates and feeds decision memory is not
normalized.

Known cards are streams like these, not an exception to them. Without the
field, each known hand card, library position and decklist entry is weighted
by its count or `1 / position` and then divided by 60, so one card known to
be on top of a library reaches its player's token at about a sixtieth of a
definition. With the field the division is gone and each source is normalized
on its own: known hand cards, known library positions and the disclosed
decklist arrive separately, so learning a hand card adds a hand stream and
leaves the decklist stream exactly as it was. One norm over their sum would
let a first known card overwrite the decklist (the knowledge vector's cosine
with its decklist-only value fell to 0.07–0.29); with three it stays at
0.67–0.75, and one known card moves its player's token by about 0.7 of its
size, against about 0.03 without the field. A normalized hand stream says
which cards are known, not how many; the player row still carries hand and
library sizes. A decklist entry stays a share of the deck (count / 60) in
both layouts.

Omitted, the field serializes as nothing and the model is the historical one:
unnormalized sums, post-normalization attention and the `/ 60` pools. A stored
checkpoint therefore computes what it did. The field cannot be changed on a
trained model, since it adds parameters and moves the norms. The `baseline`
and Elemental self-play recipes set it; frozen recipes and experiment files
are unchanged. [MLX](../manabot/backends/mlx.md#what-the-backend-covers)
supports both `stream_norm` and `focus_film` with matching weights and outputs.

### Bucketed padding

Each buffer arrives padded to its declared capacity, but a real batch uses
little of it. Across 300 captured collection batches the median batch needed 59
of 512 fact rows, 51 of 2,048 effect members and 12 of 64 actions. On the
padded path the network still processes every slot, and skips lookups for an
axis that is empty in the batch, so tensor shapes change from batch to batch.

Bucketed padding is the default for CUDA training. The host measures the last
occupied or referenced row on each variable axis, including actions, objects
and effect spans. It rounds each extent up to a power of two, with a minimum
of 8 rows and a maximum of the declared capacity. Admission validates these
prefixes in the original object coordinates. Raw buffers keep those coordinates
throughout collection and learning. Preparation translates the integer gathers
into the trimmed object table. Models with belief rows or
`historical_mean` value aggregation keep the full object table. Logits retain
the original action slots.

Torch and MLX collection borrow the original group or a contiguous slice
while preparing it. Both sample once in the original row order and keep the
measured extents on the host. The learner checks the
current rollout rows at admission and uses their collection extents as a lower
bound, since packing keeps only legal actions. Later minibatches take the
maximum across their admitted rows.

Inside the network, fact ids are looked up at the fact axis's shape instead of
for selected rows, and learning encodes the complete card catalog instead of
the definitions a batch references. Shared-effect routing, fact programs and
legal-action selection keep their packed, data-dependent layouts. A forward
therefore takes one of a few shapes per input axis instead of a new shape for
nearly every batch, and does much less work than padding everything to
capacity. The complete catalog is more work than the padded path's
referenced definitions: captured Elemental learner batches referenced 65 to
154 of 360 definitions, so learning encodes 2.3 to 5.8 times as many catalog
tokens. On CPU, where the recurrent catalog encoder dominates learning, a
learner forward and backward is therefore slower than on the padded path,
most of all for small decks.

A loop's settings choose it, with
[live-token attention](#live-token-attention). Both are plain `SelfPlay`
fields that default to off, and a recipe turns them on for the device it runs
on:

```python
from manabot.recipes.builders import gray_ogre_self_play

settings = gray_ogre_self_play(horizon=100).model_copy(
    update={"device": "cuda", "bucketed_shapes": True, "live_tokens": True}
)
settings.bucketed_shapes, settings.live_tokens  # (True, True)
```

The baseline recipe does this when its `device` is `cuda` and
`padded_shapes` is false; to compare the paths on CUDA, run it with
`--config '{"device": "cuda", "padded_shapes": true}'`
([run training](training.md#run-training)). Nothing resolves the switches
from the device: self-play checks `live_tokens` per backend at run admission
(CUDA-only on Torch; `mps` and `cpu` on MLX), and bucketed shapes may be
written explicitly on CPU or MPS, where they are slower to learn on Torch CPU,
as described above.

The loop's collection and learner policies use the settings' layout. Code
that builds a policy outside a loop chooses it itself: `ActorCritic(...)`,
`MLXActorCritic(...)`, `load_checkpoint(...)` and `backend.load_policy(...)`
default both switches to off; pass `bucketed_shapes=True` or `live_tokens=True`
to enable them.
[Served evaluation](evaluation.md#shared-inference) accepts both switches and
leaves them off: its batches are small, and the padded forward was the faster
one there. Search players load their policies padded.

Bucketed shapes are 2.4× faster per learner step than padding to capacity at
w160l8 and 2.6× at w384l8 on an L4. Collection over full tables was faster at
w160l8, 2.4 s against 2.9 s per pass with both settings on
([live-token attention](../experiments/reports/live_token_attention.md)). That is the
known cost of the default at small widths; it stays on at every width.

The setting changes how a forward is computed, not the model. It changes no
`ModelConfig` field, checkpoint, binding or policy input, so a checkpoint
trained with either setting loads and runs with the other. On CPU,
action probabilities, values, losses and gradients agree with the padded path
up to the order of floating-point sums. A batch that references no card
definition is the one difference in gradient support: the catalog encoder then
receives exactly zero gradients instead of none. Capacity overflow and
malformed inputs are rejected with the same error and message as on the
padded path.

### Live-token attention

Bucketed padding still attends over every slot in a bucket, and most of those
slots are empty: a batch whose largest board holds 9 permanents pads every
observation to 16 permanent slots, including those with two. Live-token
attention, on by default for CUDA training, computes object attention and the
per-object layers only for valid objects, so cost follows what the board holds
rather than what the bucket allows.

The forward packs the valid objects of every observation in the batch into one
sequence and records where each observation's objects begin. Attention is
restricted to objects of the same observation, so each object attends over
exactly the objects it attended over before, and nothing else. A value token
is one more packed token per observation. Every attention layer (projections,
attention, normalization and feedforward) runs on the packed sequence, which
is then scattered back to the object table with zero padding, as the padded
layers leave it. Permanent links, shared-effect membership and local facts join
the table before packing; action focus slots and the critic read it after, so
an action still scores against the post-attention embeddings of the objects it
names. The catalog encoder, shared-effect routing, fact programs and the action
head run as they do without the setting; with `bucketed_shapes=True` their axes
are still trimmed to buckets.

On Torch, attention within each observation uses `scaled_dot_product_attention`
on jagged nested tensors. On CUDA that runs the fused variable-length kernels;
on CPU it falls back to one attention call per observation, which is correct
but slow: at w160l8 a 256-row collection forward took about 0.86 s against
0.46 s padded on a laptop CPU
([measurement](../experiments/reports/eval_forward_layout.md)). On MLX, every
per-object layer runs on packed `[L, D]` tokens and, around
`mx.fast.scaled_dot_product_attention` only, Q/K/V are compacted to
`[B, H, max_live, d]` using host-derived offsets (complexity `O(B * max_live^2)`,
zero device sync). Self-play therefore admits `live_tokens` per backend in
`backend.runtime.admit_execution` before opening a run, rather than in
`SelfPlay` settings validation: Torch refuses `live_tokens=True` when
`device != "cuda"` (`live tokens require CUDA on the torch backend`), while
MLX accepts `live_tokens=True` on both `mps` and `cpu`.

Packing needs one host read per forward, the batch's total number of valid
objects, to size the packed sequence. The policy's `prepare` measures it on the
host with the token offsets, as it does bucket sizes, for collection and
learning alike, so the device forward reads nothing back.

The setting resolves, serializes and reaches policies exactly as
[bucketed shapes](#bucketed-padding) do, and the two are independent: either
may be on without the other. Code that builds a policy directly passes
`ActorCritic(..., live_tokens=True)`, `MLXActorCritic(..., live_tokens=True)`,
`load_checkpoint(..., live_tokens=True)` or
`backend.load_policy(..., live_tokens=True)`, which allow it on CPU so tests
and `.to("cuda")` callers can compare or move the layouts. Like
bucketed shapes, it changes no model, checkpoint or input contract. Action
probabilities, values, losses and gradients agree with the padded path to
within 1e-5 on CPU; on CUDA they differ only by accumulation order, as bucketed
shapes do. On an L4, live tokens over buckets make a learner step 1.45× faster than bucketed shapes
alone at w160l8 and 1.91× at w384l8, where it falls from 386 ms to 202 ms
([live-token attention](../experiments/reports/live_token_attention.md)).

## Running a checkpoint on a later revision

A checkpoint keeps loading as cards are added, Rust types are renamed and
input columns are appended. The frontier Small player was written at
[`ab2b757d9`](https://github.com/loopflowstudio/manabot/tree/ab2b757d9) with
policy input v9, before about 100 of today's names and several hundred of
today's card tokens existed.
It loads from its original file:

```python
from manabot.backends.torch.checkpoint import load_checkpoint

model, observation_space = load_checkpoint("final.pt")
```

The bound `PolicyCheckpoint.report` holds the catalog binding counts.
The loader logs them at INFO level. On the unmodified u1952 file,
both player rows read 73 of the running engine's 74 columns. The model uses
360 stored card programs, of which 136 serialize differently in this engine.
All 348 learned names still resolve. The loader initializes
396 new token rows and 103 new name rows at their respective table means.
One missing field was renamed, 45 token or record names were retired, and
zero keys were unexplained.

Academy checks the same tensor binding when it lists a checkpoint. A missing
learned key appears on the bot’s catalog row with the loader’s reason.

Nothing in the file is rewritten. `model` then plays any deck the pool admits,
including cards it never saw. Evaluate its strength on those decks through
[local evaluation](evaluation.md#quick-local-evaluation).

### What a checkpoint is bound to

A checkpoint is bound to the meaning of what it reads and chooses, by name. It
can use an expanded card set on a later revision.

| Since the checkpoint was written | Loading it |
| --- | --- |
| Cards, tokens or names were added | Loads. Learned rows move to their keys' new positions. A row for a key the checkpoint never saw starts at the mean of its table's learned rows. |
| A card it trained on is serialized differently (a renamed Rust type, a variant folded into another) | Loads. The model treats every serialization change as a respelling, including a change in the card’s rules. It reads the program stored in the checkpoint for every trained card. |
| A token or record name the runtime emits was renamed | Loads when the [rename table](#the-rename-table) has the entry. Otherwise refused, naming the key. |
| A column was appended to an input array | Loads. The model reads the columns it was trained on; the rest are dropped before it sees the array. |
| A column it reads was removed, renamed or moved | Refused, naming the column. |
| A column's kind was corrected, or the rule from kind to reading changed | Loads and computes what it did. The model reads each column by the reading the checkpoint recorded (`input_readings`), never by the running rule. |
| A category gained values | Loads at its recorded width (`one_hot:14`). A value above the recorded size lights the "above" flag; every value it trained on reads as before. |
| It records a reading this revision does not know, or one with a malformed size | Refused, naming the column and the revision that wrote the checkpoint. |
| A decision kind, the naming representation or naming branching changed | Refused, naming the field. |
| The token kinds or roles changed (`learning_schema_sha256`) | Refused. |
| The model architecture differs from the saved configuration | Refused, as before. |
| The policy input version is v1–v8 | `UnsupportedPolicyInput`, naming the revision to run it from. |
| The possible-world version changed | Refused only for a model with belief inputs; no other model reads it. |
| Engine rules changed | Loads. Play uses the current rules. An evaluation measures the effect on strength. |

If the checkpoint recorded exact decks (`setups` and `content_manifest`), the
loader checks the saved column prefixes in the same way, including for the
Elemental roster. All other binding fields must match exactly. When adding
cards changes the manifest, the loader refuses the checkpoint.

The terms:

- The **training corpus** of a checkpoint consists of the decks its run actually
  trained on through the update that wrote it. Recorded outcomes identify each
  completed matchup, including which deck was on the play. The configured
  roster alone cannot establish which matchups occurred.
- A **format** (TLA Limited, Standard) is a named card pool plus deck rules.
- The **model interface** is what the table above compares: input columns,
  token and record names, and action kinds. It has no version number. A name
  is never reused with a new meaning, and a rename is an entry in the rename
  table. The existing version keys (`policy_input_version`,
  `decision_version`) keep their spelling and their checks.
- **Card identity** is the definition key (`tla.aang_swift_savior`,
  `token.spirit`). A definition's row number is a per-revision pointer and is
  never learned: tables indexed by card or name bind by key or name.
- An **execution record** (a replay, a scored evaluation, a frozen receipt)
  names the revision that produced it and is reproduced there. Main does not
  replay another revision's games exactly; a checkpoint running here after
  rules changed is new play, measured again.

A decision added to the engine keeps the previous automatic behaviour as a
selectable action or a seat default, so a pilot trained before it can still be
seated.

### The rename table

Learned rows are keyed by the hash of a name: a Rust variant, structure, field
or record name. Renaming one in the engine would leave the trained row
unreachable, so [`managym/src/cards/token_renames.rs`](../managym/src/cards/token_renames.rs)
lists every rename since the oldest supported checkpoint, beside the
serializer that hashes the names. An entry has this form:

```rust
(Field, "opponent_picks", Some("picks")),
(Structure, "ManaUse", None),
```

A learned key that is neither in the running vocabulary nor in this table
refuses the load:

```text
ValueError: unexplained record_name key 5779523629; classify it in
managym/src/cards/token_renames.rs; it records no source revision
```

Run `uv run python scripts/pinned_pilot.py names --at ab2b757d9` with the
writer revision. It traces the names hashed by that revision's Rust engine
in a temporary Git archive. Look up the family and packed key together:

```text
record_name 5779523629 execution/6/land choices remaining
```

Add an entry in the change that renames the type. The
[pinned pilots](development.md#pinned-pilots) fail a pull request that forgets.

### Checkpoints written before keys were recorded

A checkpoint now records its column names, each column's reading, its
definition keys and naming names in row order, and the revision and Etude run
that wrote it. The container carries these in `input_columns`,
`input_readings`, `writer_catalog` and `source`. A saved pilot can load on
another revision using its embedded catalog. Earlier checkpoints can lack any of these fields. Their columns are the first N of their input version's layout
at the recorded width, which holds because columns are only appended. A
checkpoint without `input_readings` gets the readings every model used before
they were recorded (`legacy_readings`): references gathered, ids looked up,
fact and effect timestamps divided by 128 (the effect one by its row
capacity, which is 128), fact payloads as stored (`fact_value_raw`), the
placed readings above, and every other column as stored, the permanent timestamp and
unscaled counts included. It computes what it always did. A warm start
or memory branch builds its model from the checkpoint and so keeps those
readings and writes them back. A resume builds from the rule and is refused
when the saved state read differently (see "How the network reads a column").
Their keys and names come from a writer catalog found by the card digest the
file records, `manabot/model/writer_catalogs/<world_digest>.json`. The loader
checks that the ordered naming candidates (name and optional definition row)
reproduce the file's naming digest and that its key count equals the file's
definition rows. Back faces and rules fixtures without keys use their registry
names.

For an earlier file, matching card and naming digests allow the loader to use
the running catalog. If the digests differ and the external catalog is missing,
the refusal names the digest and destination. Generate the catalog at the writing
revision and commit its output:

```sh
uv run python -m manabot.model.writer_catalog > "manabot/model/writer_catalogs/${digest}.json"
```

For revisions that predate the exporter, run the current
`manabot/model/writer_catalog.py` file with the writing revision’s interpreter
and engine. At `ab2b757d9` the engine package lives under `manabot/`. From
that historical checkout, with its native extension built, run
`PYTHONPATH="$PWD/manabot" uv run python /path/to/current/manabot/model/writer_catalog.py`.

## Policy inputs

Main carries policy input v11, the fresh default, and the deprecated inputs v9
and v10 that earlier checkpoints read
([below](#deprecated-inputs-v9-and-v10)).
`ObservationConfig.policy_input_version` is always serialized. A checkpoint,
replay or saved configuration bound to any other version, or omitting it,
raises `UnsupportedPolicyInput`, which names the revision to run it from
([`ac601834d`](https://github.com/loopflowstudio/manabot/tree/ac601834d) is the
last revision supporting v1–v8). Load saved configurations through
`ObservationConfig.from_checkpoint`; do not edit a checkpoint's version or
reshape its weights to get past the error.

A layout change on main does not bump the version. A column is appended to
its array and named in `managym.observation_layout`; a checkpoint trained
before it keeps reading the columns it knows
([running a checkpoint on a later revision](#running-a-checkpoint-on-a-later-revision)).
Removing, renaming or reordering a column refuses every checkpoint that reads
it.

### Input v11: one row per game thing

Each game thing the viewer sees is one row of the object table, and every
pointer (action focus, permanent and stack links, facts, effect members,
relation references, history contexts) names that row:

| Segment | One row is |
| --- | --- |
| `agent_player`, `opponent_player` | a player |
| `agent_cards`, `opponent_cards` | a card outside the battlefield and stack (hand, inspected library, revealed or searched cards, graveyard, exile, pending public entrants in their current zone; the agent's sideboard after its cards) |
| `agent_permanents`, `opponent_permanents` | a permanent, by controller, with its characteristics: mana value, six card types, legendary, token, Ally, Lesson and ward beside its state |
| `stack` | a spell, activated ability or triggered ability, top first, one segment with `is_mine` for its controller; a spell's characteristics, an ability's index and source link, mode, X and kicker |

The published observation exposes `card.characteristics` and
`permanent.characteristics` as `ObservationCharacteristics`. Effective P/T,
name, types, keywords and ward come from one layered projection on a permanent;
rules text and mana cost describe its definition. This is a viewer projection,
not the engine's full `Characteristics` type or dynamic rules text.

```python
permanent = obs.agent_permanents[0]
name = permanent.characteristics.name
power = permanent.characteristics.power
is_token = permanent.is_token
object_id = permanent.object_id  # selection, payment and history identity
```

A spell's `stack_object.characteristics` is present even for a copy;
`object_id`, `owner_id`, `is_copy` and `announced_x` belong to the stack object.
An ability has no spell characteristics and retains its captured source
definition/reference after the source leaves. `Permanent.name` keeps its old
override meaning (`None` while the printed name applies).

The old Python `.card` accessors and flat characteristic getters are derived
compatibility aliases, warn, and are removed in 0.4.0. Frozen viewer JSON
keeps its existing fields and bytes; those wire records are normalized on read.
Historical captures whose object identity is missing or ambiguous are refused.
Academy's retained `base_power`/`base_toughness` fields still mean effective
P/T, matching their existing serialized meaning.

The object table is players, agent cards, opponent cards, agent permanents,
opponent permanents, then the stack. The encoders index a battlefield object's
`object_id` and a spell's `object_id` to its permanent or stack row. Selection,
payment, a spell's costs and modes, and history contexts reach the row that
holds the object's state. V11 carries v10's turn state (`player_turn_state`,
`card_turn_state`) and refuses nothing a v10 encoder holds. Unambiguous
historical JSON is normalized to this observation shape before encoding.

### Deprecated inputs v9 and v10

Checkpoints trained before v11 read the v9/v10 rows: a battlefield object or a
stack spell was also a card row, permanent rows (38 columns) had no
characteristics and stack abilities had no row. Selecting v9 or v10 warns
(`DeprecationWarning`). They are removed once a v11 player is at least as
strong as the strongest v9 one, and not before 0.4.0. Their rows are rebuilt from
the same observation by `agent/legacy_input.rs` and
`manabot/env/legacy_input.py`, and match the encoders of 452d7b5357 byte for
byte; the model reads them through the same `ActorCritic`, without a stack
segment. Nothing else depends on those modules.

### Input v10: turn history and Powerstone mana

Three TLE bonus-sheet cards read state v9 has no place for:

| Card | State v10 adds |
| --- | --- |
| Bloodchief Ascension | Life each player has lost this turn |
| Brought Back | Which graveyard cards were put there from the battlefield this turn |
| Cityscape Leveler | Mana that can't be spent to cast nonartifact spells (its Powerstone tokens) |

V10 adds this state to v9, and v11 keeps it. To train on v10 rows, select v10
explicitly before constructing the collector and model:

```python
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ObservationConfig
from manabot.recipes.builders import gray_ogre_self_play

settings = gray_ogre_self_play(horizon=1).model_copy(
    update={"observation": ObservationConfig(policy_input_version=10)}
)
space = ObservationSpace(settings.observation)
```

Use that `space` for `open_collector` and `ActorCritic(space, settings.model)`;
the run's saved settings keep the selection. V10 is v9 plus this state,
in the native encoder, the LLM text observation and the learning tokens.

V10 adds two buffers to the v9 layout, required when `ObservationLayout` is
built for input v10 (and carried into v11):

| Key | Shape | Holds |
| --- | --- | --- |
| `player_turn_state` | 2 × 8 | Per seat, agent first: life lost this turn / 20, spells cast this turn / 10, then mana that can't be spent to cast nonartifact spells by W/U/B/R/G/C / 10 |
| `card_turn_state` | 2 × 60 | Per seat card row (aligned with `semantic_cards`): 1 for a graveyard card put there from the battlefield this turn |

The model appends each seat's row to its player row and each card value to its
card row before embedding them.

V9 and its checkpoints are unchanged: w17 checkpoints keep loading. A v9 encoder refuses a position that needs v10 state:
a card visible to the viewer reads life lost this turn or graveyard cards put
there from the battlefield and that state is present, or a pool holds mana
that can't be spent to cast nonartifact spells. `managym.input_v10_state(obs)`
names the needed state. Encoding raises `ValueError`
(`policy input v10 required: …`, with that state) before writing any buffer;
`UnsupportedPolicyInput` is for a configuration or checkpoint bound to an
unsupported version, not for a position. A v9 run over decks with these cards
stops at the first such position, so train on them with v10.
A game, replay or Academy session needs no policy input and plays these cards
under any configuration. A v9 checkpoint cannot be widened to v10; train a new
one.

### Layout

A fresh `ObservationConfig` (input v11) gives these float32 buffers.
Capacities are configuration; widths are the layout.

| Key | Shape | Holds |
| --- | --- | --- |
| `agent_player`, `opponent_player` | 1 × 74 | Player state and public game disclosure |
| `agent_cards`, `opponent_cards` | 60 × 56 | Cards outside the battlefield and stack |
| `agent_permanents`, `opponent_permanents` | 40 × 51 | Permanents with their characteristics |
| `stack`, `stack_valid` | 32 × 37 (`max_stack_objects`), 32 | Spells and abilities, top first |
| `player_turn_state`, `card_turn_state` | 2 × 8, 2 × 60 | Turn state (above) |
| `relations` | 64 × 49 | Stack entries, targets, clauses, modes, costs, suspended payments, waiting triggers |
| `facts` | 512 × 12 (`max_facts`) | One row per object and detail of an open vocabulary |
| `effects`, `effect_details`, `effect_members` | 128 × 18, 4096 × 3, 2048 × 2 | Each active effect once, with its payload tokens and affected objects |
| `known_library` | 2 × 20 × 2 | Exact library facts: definition + 1, position from top + 1; zero padding |
| `decklists`, `known_hand`, `semantic_cards` | 2 × 60 × 5, 2 × 60 × 2, 2 × 60 | Disclosed lists, known-hand minima, definition joins |
| `actions`, `action_focus`, `name_candidates` | 512 × 25 (v11), 512 × 2, 512 × 4 | Legal actions, the objects each names, name choices |
| `events` | 32 × 7, or 32 × 12 with recent events | Public events |

Policy inputs v9 and v10 keep 22 action columns. In v11, column 22
(`pays_directly`, zero-based) marks taps that directly pay a cost through
convoke, improvise or waterbend.
When such a tap could pay more than one thing, the next choice's `mode`
column names what it pays, the same in every decision: the selector 0-4 is
one {W}/{U}/{B}/{R}/{G} pip by convoke, 5 {1} by convoke, 6 {1} by improvise
and 7 {1} of the waterbend part (the column holds selector + 1). Only payable
modes are offered, and the choice's focus is the permanent just tapped.

The same `mode` column carries every indexed choice: a mode of a modal spell,
a mana-spend option, a replacement order and a value of X. In v11, columns 23
and 24 say when the index is a number. `has_x` is 1 on an offer that chooses a
value for X (for a spell, an activation or a payment, or a replicate count),
and `x` is that value in tenths, named and scaled as the stack row's `has_x`
and `x` are. The native `Action.announces_x` says the same of an offer:

```python
# arrays = game.encode_observation(observation), at a decision that chooses X
columns = managym.observation_layout()["actions"]["columns"]
rows = arrays["actions"][arrays["actions_valid"].astype(bool)]
print(rows[:, [columns["mode"]["index"], columns["has_x"]["index"], columns["x"]["index"]]])
# choosing X for a spell with three mana left after its other costs
# [[1.  1.  0. ]
#  [2.  1.  0.1]
#  [3.  1.  0.2]
#  [4.  1.  0.3]]
```

`mode` keeps its stored value on these rows. Both columns are appended, so a
checkpoint trained before them does not read them. Inputs v9 and v10 have
neither; their X offers differ only in `mode`.

The game-level `decklists_closed` flag is inverted in the `decklists_open`
column (73) of each player row: 1
means both players know both lists, 0 means neither knows the opponent's list.
The native game supplies it for both viewers; it accompanies the other game
facts on both player learning tokens and is explicit in the text observation.
Closed opponent list rows remain zero, while public reveals still update
`known_hand`. The column is appended and adds no policy-input version; a
checkpoint trained without it does not read it, and plays closed-list games
out of distribution.

Allocate and collect every key from the supplied `ObservationSpace`, on both
the scalar and vector paths. Fact, relation, effect and action overflow is a
typed Capacity error, never truncation; the one declared exception is the TLA
pilot's first-64 action prefix.
`agent/continuation.rs` projects the relation frames and
`agent/observation_encoder.rs` the fact rows once per observation;
`manabot/env/observation.py` describes their layout and writes nothing. A buffer that
is missing or has the wrong shape fails before any write. Column names and
value encodings for every buffer come from `managym.observation_layout()`
([reading arrays by name](engine.md#reading-arrays-by-name)).

### How the network reads a column

The engine declares what it wrote in each column, its kind. The model chooses
how to read the column, its reading, from that kind. A checkpoint records the
reading of every column it was trained on:

```python
import managym
from manabot.policy.checkpoint import read_metadata

columns = managym.observation_layout()["agent_permanents"]["columns"]
readings = read_metadata("policy.safetensors")["input_readings"]["agent_permanents"]
for name in ("tapped", "power", "exhausted_abilities", "timestamp", "attached_to", "definition"):
    column = columns[name]
    print(name, column["kind"], column.get("scale", ""), readings[column["index"]])
# tapped flag  as_is
# power number 10 steps:8
# exhausted_abilities count 1 log
# timestamp rank  rank
# attached_to reference  gather
# definition id  lookup
```

A reading can turn one stored column into several inputs. `steps:8` is nine:
the stored value, then one flag for each of "at least 1" through "at least 8".

| Creature | Stored `power` | What the model receives |
| --- | --- | --- |
| 2/2 | 0.2 | `0.2, 1, 1, 0, 0, 0, 0, 0, 0` |
| 3/3 | 0.3 | `0.3, 1, 1, 1, 0, 0, 0, 0, 0` |
| 12/12 | 1.2 | `1.2, 1, 1, 1, 1, 1, 1, 1, 1` |
| −1/1 | −0.1 | `−0.1, 0, 0, 0, 0, 0, 0, 0, 0` |

As a bare tenth, the step from 2 power to 3 is a change of 0.1 beside a row
of 0-or-1 flags. With the flags it is as large as tapping the creature.

A model built fresh takes each reading from this rule. `prepare` applies it
once on the host, for every backend:

| Kind | Reading | What the model receives |
| --- | --- | --- |
| `flag` | `as_is` | The stored 0 or 1 |
| `count`, `number` with `scale` 10 | `steps:8` | The stored tenths, then flags for the true value being at least 1, 2, … 8. Power, toughness, damage, mana value, X, and ward cost |
| `mana` with `scale` above 1 | `as_is` | One scalar per floating-mana pool, in stored units |
| `count`, `number` with any other `scale` above 1 | `as_is` | The stored value: the engine already divided by `scale` |
| `count`, `number`, `mana` with `scale` 1 | `log` | `sign(x) · log1p(abs(x)) / 10` |
| `category` of declared `size` n | `one_hot:n` | n + 1 flags: one for each value 1 to n, then one for "above n". Codes below 1 leave all flags zero |
| `selector` | `one_hot:15` | 16 flags: indexes 1 to 15, then "above 15" |
| `position` | `log` | The same squash as an unscaled count |
| `rank` | `rank` | The rank / 128, whatever the fact or effect capacity |
| `reference` | `gather` | The row of the object it names; never a number |
| `id` | `lookup` | A learned embedding row; never a number |
| `tagged` | `fact_value` | By the row's fact kind: a scaled payload as stored, an unscaled count through `log`, and zero for program rows, whose content is looked up |

No category or index reaches a projection as a magnitude: a decision kind of
28 is not "twice" a decision kind of 14. A category declares how many values
it has beside its kind, and the reading records that size:

```python
kind = managym.observation_layout()["relations"]["columns"]["frame_kind"]
print(kind["kind"], kind["size"])
# category 14
```

The last flag of a one-hot is always "above the size" and never shared with a
real value. When a category later gains a value, a checkpoint that recorded
`one_hot:14` keeps its width, sees the new value as "above 14" and reads
every value it trained on exactly as before; a value above the size is never
an error. A fresh model takes the new size. A mode index that packs several
chosen modes into bits lands in "above 15"; the model does not decode it.

No column is fed to a projection at the scale the engine happened to store. On
sampled Elemental games every number a fresh model receives stays within
±4. Life and turn are divided by 20 and have no fixed bound, so a very long
game or a life-gain deck can exceed that.

A fresh model reads a few columns according to their role in a structured row
(`PLACED_READINGS` in `manabot/policy/readings.py`). The `valid` flags of actions,
relations, facts and effects read `mask`: a mask already carries them, so no projection
receives them. A fact's `kind` and detail words read `lookup`; its id words
read `fact_id`, a token row for named kinds and the ordinal / 128 for frame
and ability indexes; its `duration` reads `duration`, four flags for until end
of turn, indefinite, while the source is controlled and until your next turn
(a checkpoint that records no readings reads `duration_two`, the first two
flags only). An effect's detail and member counts read `gather`
with their offsets, as the extent of a join.

A relation's `frame` and `parent` are declared `selector` but name another
relation row. A fresh model reads them `gather`: frame pooling follows them as
a join, and they are not numeric inputs.

A column whose kind has no reading stops the model from building. `gather`
and `lookup` columns never reach a projection: they are routed as joins and
embedding rows, so a reading cannot turn an id into a magnitude. A new column therefore
needs only its native declaration: give it the kind that says what it holds,
and `scale` when the encoder divides it.

Other buffers are decoded by their own preparation rather than column by
column: validity tables, known cards and decklists, naming candidates,
decision-memory records, recent public events and shared-effect details. The
model names each of them with the function that decodes it
(`OWN_PREPARATION`), and their columns read `own`; their native kinds are not
consulted. A float buffer that is neither read through readings nor named
there stops the model from building. Decision memory's program ordinal is
divided by 128 and reaches about 6 on sampled games; its preparation records
no readings, so it stays as it is.

A stored checkpoint never consults the rule. It reads each column the way
`input_readings` says, so a later correction to a column's kind or to the
rule changes fresh models only
([what a checkpoint is bound to](#what-a-checkpoint-is-bound-to)). A training
run records its model settings and input readings in the saved state. For an
older run, the initial raw policy supplies the model settings. If either the
settings or the readings differ, resume stops before loading weights. Resume
from the revision that started the run, or warm start from its last published
policy. A state saved before readings were recorded read
`legacy_readings`, so resuming such a run with a fresh model is refused too.
A warm start keeps the policy's readings but builds the requested
`ModelConfig`; a policy written before `definition_names` was recorded needs
`definition_names=True`, and the refusal names each differing setting.

### Relations and facts

Relation validity is column 31. Only the acting viewer receives private
continuation. Transport IDs are joins to the card catalog and object table,
never numeric features. At `Decision::ManaChoice` the frame's source is the
mana ability's source; a triggered mana ability's type choice (Heartbeat of
Spring) names that ability's source, with the permanent tapped for mana as
its context.

Waiting delayed abilities (frame kind 10) carry event codes independently of
frequency and expiry. Relation row kind 9 carries frequency in column 28 (1
once, 2 each matching event); column 48 retains cleanup expiry. Remembered
objects are kind-8 reference rows, separate from event context. Ordered control
schedules and delegated decisions use kinds 13 and 14.

A ward that is not paid in mana ("Ward—Pay 7 life.") sets the ward flag with
a ward cost of 0, and its price is an effect program on the permanent: fact
kind 11 under the id `Ward`, in the tokens the card's definition uses
(`PaymentCost::Life`, then the amount). The LLM text says "ward: pay 7 life".

Pooled mana whose spending has an effect (Animal Attendant's "that creature
enters with an additional +1/+1 counter") counts in its player's ordinary
or restricted mana aggregate, according to its restriction, and is stated as
an effect program on that player: fact kind 11
under the id `PooledMana`, one program per class, in the tokens the producing
card's definition uses. The mana actually spent to cast a spell (CR 601.2h) is
an execution occurrence of its stack or resolution frame (`mana spent`, a
kind-14 continuation program) when a rule still reads it: the definition
counts that mana (Toph, converge) or a spend effect applied. These programs
use the same representation in v9 and v10. Combat mana also has a pool
program, including ordinary combat mana; its class records combat expiry.
Regular and combat units never merge during payment search. Academy shows each
rider, restriction and expiry alongside the public amounts in
`pooled_mana_classes`. A class without a restriction or spend effect omits that
qualifier. Class rows state amounts first; Academy uses the aggregate combat
expiry only when no class rows are recorded.

Spending offers select one color and class at a time. The native obligation
frame (decision code 31) carries the remaining cost; its execution program
joins each offer's mode to the exact unit and retains the partial receipt.
Every offer preserves all saved payments, including nested mana-ability fees.
Equivalent units and permutations of a single forced payment need no choice.
Submitting a spending Command commits that unit; replay, forks and stale
rejection use the same authority as other decisions.

Fact rows carry source, duration and timestamp rank. Every timestamp and
order rank (permanent, fact, effect) is a `rank` column, which a fresh model
divides by 128, independent of fact and effect capacity
([how the network reads a column](#how-the-network-reads-a-column)).

### Shared effects

Each active effect is one `effects` header owning bounded slices of
`effect_details` and `effect_members`. Details hold the payload as learning
tokens, including granted, copy-exception and captured continuation programs,
changed colors and keyword-counter timestamps; members keep one-based public
object joins, and duplicate memberships survive. No rules record is deleted or
semantically coalesced. The layout, routing and sum/normalization tradeoff are
in the torch backend's shared-effect encoder
([`manabot/backends/torch/_shared_effects.py`](../manabot/backends/torch/_shared_effects.py)) and
[shared effect inputs](../manabot/README.md#shared-effect-inputs). The transport
is an engineering result; it establishes no learning-quality advantage.

Encode-only benchmark: `cargo run --release --features benchmark --manifest-path managym/Cargo.toml --example bench_observation`.
Matched full-vector benchmarks use `scripts/bench_breakdown.py --matched-corpus`
on a release extension built with the `benchmark` feature.

### Catalog naming input

Naming a card is a sequence of ordinary native decisions. A prompt over at most
`managym.NAME_BRANCHING` (64) names offers those names; a larger prompt offers a
deterministic, balanced, complete partition of its sorted names into at most 64
contiguous groups, recursively. With 928 names that is 15 groups of 61 or 62,
then one name prompt; the installed 344-name catalog uses 6 groups. Choosing a
group narrows the pending prompt and commits nothing: no search, inspection or
exile. Only choosing a name resumes the effect. Each prompt is one learner
transition through the ordinary policy, value estimate, rollout and learner,
with the standard discount and credit; no reward is assigned to group choices.
The extra decision is real temporal structure, so learning is not claimed to
equal one flat choice over every name. Single-option prompts follow the native
trivial-decision skip. Reconnect, replay and forks restore the actual current
prompt; Commands bound to an earlier prompt are stale.

`name_candidates[A,4]` at the ordinary action capacity `A = max_actions` holds
a one-based public name row, an optional one-based definition reference, then
the one-based first and last name rows of a group choice (zero means absent).
Sorted names make the span a viewer-safe identity of every member. No
card/permanent row is fabricated for a name or group. These rows suppress the
positional mode feature, so identity comes from categorical joins, never from
magnitude.

The policy adds a trainable vector per name and, where present, the
definition's learning-program embedding from the table used by game objects. A
group's vector is the mean of its members' name vectors, computed exactly from
a prefix sum over the name table: every member receives the group's gradient,
and no member programs are encoded for a group prompt. Name vectors score
NameCard candidates only: a card, permanent, known-hand or decklist token's
definition is its rules program. A unit-variance name vector added onto every
definition outweighed the rest of an object token at initialization (on
sampled games a 9/9 creature moves its token by a median of about 21% with
names added, about 60% without), and
gave training a second source of card identity. The cost is that equal-rule
cards with different public names read alike off the NameCard prompt.
Checkpoints written before this choice add name vectors onto viewer-visible
definitions before pooling and keep doing so (`ModelConfig.definition_names`,
read as true when a world-bound checkpoint omits it); closed native decklist
rows contribute no identities. Action networks score
only legal rows, and rollouts pack legal action fields and complete behavior
probabilities with original coordinates.

A checkpoint records its naming names in row order with
`naming_representation="public-groups-v1"` and `naming_branching`. A changed
representation or branching bound refuses it. Added or reordered names do
not: learned name rows bind by name, and a new name starts at the mean of the
learned rows. The semantic observation/capture contracts are 8/6.

The default universe remains real non-token catalog names, including back faces.
A separate immutable vocabulary can hold validated names without playable card
definitions; this is exercised only by the offline large-vocabulary fixture.
This does not claim that the supported catalog exhausts every name legal under
Magic's rules, and does not broaden deck admission.

### Exact library knowledge

The native observation's `known_library` maps a zero-based library position
from the top to a card definition. The owner sees the union of public and
private facts; an opponent sees public facts only. The current policy input
carries twenty rows per seat (ten public plus ten private, with coinciding
positions represented once). Deprecated inputs v9/v10 omit these facts and preserve their frozen layout,
so their checkpoints never read them.

The encoder writes definition joins and one-based positions. Prepared resolves definitions once and weights each by the
reciprocal of its one-based position. Both Torch and MLX project this separate
pooled vector, divided by 60, before adding it to hand and decklist knowledge,
so a known top card and a known hand card have distinct representations;
these rows do not extend the deprecated v9/v10 layout. It is a minimal live
read, with no measured claim about learning or an ability to infer joint
constraints. The text observation states each exact position in words.

A private return can erase independent facts; it never creates a claim that a
card occupies several possible places. Models may infer such relationships
from their available observations and history. Native sampling honors every
retained single-place fact ([engine knowledge](engine.md#remembered-library-positions)).

### Optional recent public events

`ModelConfig(recent_events=True)` adds shared policy/value context from a
bounded suffix of public events; `SelfPlay` derives
`observation.policy_history_version=1` from it. `Game::emit` maintains
the projection at the same append as the rules ledger, keeping at most 32
rows; private arrivals produce no row, and owner (not controller) is recorded.
History-on uses an `[E,12]` tensor (`1 <= E <= 32`) with a validity mask:
kind (public arrival, damage, life change, spell cast, resolved, countered),
signed amount, and source/target references `(definition+1, owner_role,
zone+1, context_index, availability)`. Valid rows are mean-pooled and added to
shared object embeddings; an empty window gives exactly zero context.
History-off keeps the historical `[E,7]` tensor and parameters. This is not
complete decision history, and no training benefit has been measured.

## Whole-game decision memory

**Unreleased.** Whole-game native logs, shared immutable prefixes, and Torch
and MLX encoders are implemented with bounded execution rows. Memory stays opt-in.
The [cost report](../experiments/opponent_memory.md) records the current
suffix A40 calibration: control costs 1.105× incumbent time and GRU-K1 1.201×. These are exploratory cost measurements;
matched training and paired strength evaluation have not run. These CUDA receipts
precede the record-cache and native opt-in cuts below. Transformer probe and
earlier calibration evidence remain separate.

Collection retains native table cursors and reuses unchanged prefix versions.
The learner folds shared chunks assembled from the same cached per-record host
features. Collection featurizes individual appended records and caches detached recurrent states after every record except
the current last one. That last record is recomputed so coalesced passes keep
their multiplicity. Each native stream owns its open serving segment; earlier
issued prefixes retain their exact version. Unbranched new suffixes run in one
packed GRU call, whose intermediate outputs supply the per-record states. Native
game generation and prefix identity bound cache reuse, and scope exit discards the
cache before a parameter update. These tensors never become observations,
replays, checkpoints or search-fork state. Dense stateless encoding remains the
reference, and arbitrary imported prefixes still undergo complete admission.
The transformer keeps its full-sequence encoder. Raw storage and recovery keep
content-addressed immutable chunks and full changed-chunk admission; this cache
cut does not remove that retention cost.

To disable history, set `decision_memory="none"`, all four decision-memory
capacities to zero and `decision_memory_include_own=False`. Training then leaves
native decision logs empty and bypasses capture. Positive record capacity enables
capture even for a log-only control. Direct native callers opt in with
`managym.Env(decision_memory=True)` or `managym.VectorEnv(decision_memory=True)`;
`decision_memory_include_own=True` also enables it. Search clones retain existing
facts and never append hypothetical decisions.

To let a policy use the opponent's earlier choices, select an encoder and
one or four summaries in its model configuration:

```python
from manabot.infra.config import ModelConfig

model = ModelConfig(
    decision_memory="transformer",
    decision_memory_summaries=1,
    attention_on=True,
    attention_layers=2,
    policy_film=True,
    value_aggregation="masked_mean",
)
```

Use this model configuration in a recipe's `SelfPlay` settings. The settings
must also declare the log's capacities and bounded execution rows, or
validation names the missing fields:

```python
from manabot.infra.config import ObservationConfig

observation = ObservationConfig(
    max_decision_records=512,
    max_decision_reference_details=16_384,
    max_decision_references=128,
    max_decision_members=8192,
    decision_memory_include_own=False,
)
settings = SelfPlay(
    match=match, model=model, observation=observation, learning=learning,
    collection_rows=2, bootstrap_rows=2, learner_rows=2,
)
```

These are bounds selected from the seeded census; a game that exceeds them stops with a
typed capacity error rather than dropping records. The policy receives the whole game's viewer-safe
opponent decision records at each decision. For example, a pass with mana open
and a card in hand remains available after later turns; a recent-event window
can lose that context. The encoder produces summaries that condition the board
attention, allowing both legal-action probabilities and the value estimate to
depend on the earlier choice. This does not establish a learning benefit; the
[three-arm pilot](../experiments/opponent_memory.md) compares both encoders with
a control, after supervised probes select one shared summary count.

| Setting | Choices and behavior |
| --- | --- |
| `decision_memory` | `"none"` (default), `"transformer"` or `"gru"`. Memory is opt-in. |
| `decision_memory_summaries` | 1 or 4 for either encoder; default 1. Ignored when memory is `"none"`. |
| `attention_on`, `policy_film` | Both must be `True` for memory. FiLM carries board context to actions without focus objects. |
| `recent_events` | Must be `False` when decision memory is enabled. The bounded event encoder remains a separate option. |
| `decision_memory_gate_scale` | Positive float, default 1.0. Multiplies each board layer's learned gate parameter inside `tanh`; a larger scale opens the join in fewer updates. Ignored when memory is `"none"`. |

A control can use `decision_memory="none"` with the same positive
observation capacities. Native encoding, immutable prefix retention and input
validation still carry the full log. The control creates no memory parameters
or prepared memory features. Its vocabulary shape differs from the
history-disabled incumbent, so the two do not share identical initialization.
Checkpoints retain the declared observation capacities in every arm.

The LLM text observation lists every opponent record chronologically. Repeated
historical references and rules descriptions appear once in alias tables; the
aliases resolve snapshots at the choice, never today's objects. Paired life and
hand values, and the two board rows, are ordered You/Opponent. Text representation
`manabot-text-10` prints the first record's context in full, then only changed
turn, phase, step, life, hand, board and untapped-source values. Omitted context
fields keep the previous record's value; `none` explicitly clears a reference
group. Choices and responses apply only to their own record, including empty
groups. Every record retains its kind, count, chronological span and selectors.

Native long-game fixtures with hundreds of distinct contexts fit the default
256 KiB budget through 1,029 records under both disclosures. This is a bounded
size check, not a guarantee for every game: sufficiently large distinct
snapshots still raise `RepresentationOverflow` with the complete size, without
trimming history. A byte fit does not establish a model's token fit; the LLM
harness must check its tokenizer and chat-template budget.

LLM games record and show this history only when `GameConfig.decision_memory`
is set (`manabot.verify.llm_game`); it is off by default, like the policy's
encoder.

Run the headless software proof with:

```bash
uv run --extra dev python -m manabot.verify.decision_memory --out .runs/decision-memory-demo
```

Use a fresh output directory. It retains both viewers' logs, captures, text,
replayed Commands, four policy checkpoints under each disclosure and a JSON
summary. It checks a private selection followed by public revelation, a pass
with mana open, a declined block, branch isolation, live and restored
observations encoding alike, separate
actor/value gradients and one ordinary CPU update with checkpoint reload.
Controlled FiLM weights exercise memory sensitivity before training; the update
uses ordinary initialization. This proves software connectivity and preserves
legality and focus; CUDA admission, throughput and learning remain to be measured.

Both encoders use record width 64 and zero dropout. The transformer has two
layers, four heads and feedforward width 128. The GRU has one layer with hidden
width 64 and projects its final valid state to the selected summaries. Both
return zero summaries for an empty history. Record embeddings use absolute
ordinal and span features. The GRU reads their chronological order; the
transformer adds a learned projection of prefix-relative recency (rank / 128)
as a sequence position signal, outside the shared record embedding. Both the
Torch and MLX backends support both decision-memory encoders (`gru` and
`transformer`, 1 or 4 summaries) and control-trunk branching. During
collection, both GRU encoders can reuse stable records for the same viewer
prefix and native epoch. A restored branch can share record content while
requiring a fresh recurrent state. The latest record is recomputed on every draw.

### What a record means

managym records committed, nontrivial opponent choices separately for each
viewer. By default, your own choices are excluded; the optional own-seat mode is
described below. Decisions with only one legal action are always excluded.
A pass is eligible only when the opponent holds priority, has at least one
untapped mana source and has a card in hand. This rule never tests whether a
hidden card is castable. Consecutive eligible passes with identical public
context, response and choice coalesce, retaining their count and chronological
span. A changed public context prevents coalescing.

Each record contains its order and turn/step, decision kind, life totals, hand
sizes, board summary, untapped mana-source snapshots, what the choice responded
to and the public projection of the chosen action. Historical characteristics
are snapshots; leaving and returning objects do not rewrite them. Private
choices remain unavailable, including after a later reveal. Open or closed
decklists follow the match's native disclosure setting; memory grants no extra
information. Native observations, JSON, policy tensors, learning tokens and LLM
text carry the same viewer-safe records.

### Replay, reset and checkpoints

A new game starts with an empty log; turn boundaries preserve it. Rejected or
stale Commands append nothing. Search forks inherit the viewer's prefix without
appending hypothetical choices; sampled hidden cards never relabel that prefix.
When capture is enabled, replaying the original Commands reconstructs the records,
including after resume. Each seat keeps its own view.

The policy summary depends on the entire available prefix. Collection may reuse
stable-record results within an unchanged inference scope; the cache is discarded
before parameter updates and excluded from game persistence. Training remains per decision;
recurrent sequence batches and burn-in are unnecessary. A checkpoint retains
the encoder choice and weights through ordinary persistence. Restoring a game
still requires its original replay or exact native state; weights alone do not
restore a game in progress.

Collection retains immutable game/viewer prefixes, sharing unchanged records and
details across decisions. Coalescing creates a new tail version; previously
sampled decisions keep their original count. Record-aligned chunks own consecutive records, their members, the snapshots
they introduce and those snapshots' details. Each immutable node is admitted
against its predecessor's ordinal/span/member/reference/detail cursor. Cached
features bind the checkpoint vocabulary's immutable bytes and definition bounds;
sequence coordinates and batch offsets are assembled for the selected prefix.
Cached record features do not change when the prefix grows. Dense imports
and explicit dense exports still undergo complete content and padding checks. Ordinary collector recovery replays its bounded native
journal and checks a compact chunk archive. Numerical incidents select rows and
budget whole prefixes before copying them; evidence that cannot fit is explicitly
marked incomplete. These diagnostics do not resume games.

Bootstrap retains immutable prefixes with ordinary next-state fields. Collection,
frozen opponents and bootstrap select prefix references and assemble one execution
chunk directly from admitted fragments. Host feature caches expire with their
prefix/segment owners and hold no learned embeddings or recurrent state. Sampling still uses one
RNG call in the original group order.

For GRU learning, each execution chunk prepares the distinct record chunks in
its selected prefixes once. The backend embeds each node once, advances its GRU
from the predecessor state and reads each policy row at its own tail. Repeated
prefixes share gradient paths: their loss contributions still add with the
logical actor and critic denominators. Rows keep their selected order; sampling
and optimizer boundaries are unchanged. This temporary tree and its neural
states expire with that execution chunk's backward graph. Collection uses the
same causal fold with detached record roots, recomputing only each current last record.
Bootstrap without an inference scope folds without cached roots. Dense inputs
and the transformer retain the stateless full-prefix encoder.

Memory-enabled self-play (including a control retaining the same logs)
requires explicit positive `SelfPlay.collection_rows`, `bootstrap_rows` and
`learner_rows` limits. For example, two rows each exercises the controlled CPU
proof size; calibrate these resource limits for the target device. These Torch
execution settings do not change native stream count,
history capacities, learner selection, shuffled order or logical optimizer
minibatches. Learner chunks accumulate gradients with the logical minibatch's
separate actor and critic denominators, then clip and step once. PPO prepares
at most one chunk ahead. Full histories remain intact even for an execution
chunk of one row; inability to fit that row is a calibration failure.

Host RSS and device memory remain separate limits. CPU chunk parity and
residency measurements do not admit a CUDA worker, guarantee learned-game
storage, or establish throughput or playing strength. The independent-game host
measurement covers its recorded cohort and one logical execution minibatch.
Recipes without memory retain their historical execution when row limits are
omitted.

Record, reference (an object's snapshot at the choice), reference-detail and
member capacities belong to the supplied
observation space: `max_decision_records`, `max_decision_references`,
`max_decision_reference_details` and `max_decision_members`. All default to zero (disabled)
and must be positive together for v11 memory. The example uses the 200-game
Elemental heuristic census
maxima rounded upward; it does not admit learned-game tails.

The native viewer log interns exact immutable reference snapshots. Records own
ordered mana-context, response and choice members, including repeated references.
Life, hand and board counts occupy fixed named record columns with exact two-word
integer transport. Eligible exact registered faces may omit their repeated rule
program; preparation expands them through the same semantic route as full
snapshots. Local table indexes and appearance identities are never numerical
learning features. The nonlinear featurizer binds each detail's field, typed
value and position before reference pooling, then pools the three member groups
separately. Transformer records use pre-norm attention; summaries are normalized
before the gated additive board join.

`decision_memory_include_own=True` additionally records your own public signals,
using the opposing viewer's entitled projection **at emission**, rebased to your
seat. Non-announced references must also be entitled to you; the other viewer's
private knowledge is never declassified into your history.
It never stores your private choice or hand view as public evidence. Actor roles
remain explicit and different actors never coalesce. Native scalar/vector
constructors take the same default-false keyword; library training wrappers pass
it from `ObservationConfig`. Configure it before reset, since missing history
cannot be filled in later. Primary census and CPU comparisons keep it false.
Disclosure is one symmetric game-level setting on `GameSetup`; both player
rows carry `decklists_open`. That bit also tells the policy whether the
opponent knows its list, so own-seat memory adds no disclosure flag.


`ActorCritic.encode_decision_memory(device_prepared)` exposes the same
production summaries as a differentiable `[batch, K, hidden_dim]` tensor for
offline probes. It uses the policy's token, name and definition embeddings;
its input is admitted Prepared data, and it evaluates no action or value head.

### Starting from a control trunk

Each arm is a fresh self-play run started through `SelfPlay.pretrained`.
Collection retains record-aligned immutable chunks. Native transport counts and
process-local game generations let each stream extend its previous viewer prefix:
one bulk comparison of native generations and cursors reuses issued prefixes
for unchanged rows before scalar capture. Changed rows reuse their game identity
and pass transport shape/capacity checks in a batch; their immutable chunks still
receive full admission. Vocabulary and definition admission batch only the new
chunks, keeping temporal and partition checks on each exact predecessor chain.
After a commit, closed chunks are reused and the open chunk is re-read to preserve coalesced
passes. Reset, fork and replay restoration change the generation even when a
seed repeats. These cursors are transport metadata, never policy features or
replay identities. Dense imports and archives still validate the complete prefix.

During collection, each record except the current last one caches a detached
state and cumulative reference embeddings for the unchanged policy's inference
scope. Preparation stops at that root and builds only newly appended records
and the current last record; cached ancestors contribute no feature assembly or
device transfer. Coalesced versions keep their own last-record fragment. New
record fragments share one host featurization batch, and unbranched suffixes
share one packed GRU call while retaining all intermediate states. Scope closure
or a parameter change invalidates neural results; these caches never enter a replay,
checkpoint or search fork. Learning continues to fold unique nodes with gradients
inside each execution chunk.

A control is a log-enabled policy with `decision_memory="none"`: it retains and
validates the log and reserves its vocabulary without encoding it. An incumbent
has history disabled. The true history-disabled incumbent remains the throughput
reference.

Set `SelfPlay.pretrained` to the trunk's published checkpoint; a launched job
receives it as `TrainingJob(start=...)` and reads `run.input("start")`
([training](training.md#reuse-a-published-policy)). Choose the memory encoder,
summary count and gate scale in the branch's `model`. The control branch uses
the trunk's identical model settings.

Loading verifies the world and observation configuration. It permits a model
change only from a control to a memory encoder, with only `decision_memory`,
`decision_memory_summaries` and `decision_memory_gate_scale` differing. The
trunk must declare the same positive decision capacities as the branch. The
loop copies every shared tensor with strict shape checks and initializes only
`decision_memory.*` and the per-layer memory projections and gates; missing
shared or unexpected keys fail. A trunk from an older world or a different
observation layout cannot be branched.

At initialization, including long and empty histories, a branch gives the
trunk's action probabilities and value bit for bit. Adam, EMA, streams and the
schedule start fresh at branch iteration one. Etude records the starting
checkpoint as the run's input; checkpoints retain their own source identity.

Each board attention layer adds a gated projection of the mean memory summary
(K=1 in the pilot; K=4 remains supported) to every real board token before its
single self-attention call:

```text
board_input + tanh(scale * gate) * projection(mean(summaries))
```

The addition is owner-neutral, skips padding and leaves the optional value token's
input untouched. The value token still reads the conditioned board through
attention. No summary tokens extend the sequence. Each layer has its own
nonzero-initialized, bias-free projection and scalar `gate` initialized to zero.
The gate receives a finite gradient on the first update; projection and encoder
receive gradients once the gate opens. Empty histories yield zero summaries and
therefore zero additions, even with open gates. At initialization, the branch
uses exactly the control's sequence, attention and feedforward arithmetic.
Every memory policy uses this gated join, including one built without a trunk.

This replaces the unmerged J1 token join, which computed both joint and control
attention at every layer and blended the results. Frozen probe and CUDA receipts
retain that earlier arithmetic and their original source identities; they do not
measure the additive join.

Training logs `decision_memory/gate_<layer>`, the value of
`tanh(scale * gate)` for each board layer. A gate that stays near zero means
the run still computes its board-only control, whatever its configuration says; read
that metric before comparing a memory run with its control. A gate moves about
one learning rate per Adam optimizer step, rather than per collection iteration,
so a short run can end before it opens. Raise
`decision_memory_gate_scale` for the whole comparison in that case, never for
one run.

To measure whether a trained policy uses its memory, play the checkpoint
against itself with the memory blanked:

```python
from manabot.eval.admission import register_checkpoint

intact = register_checkpoint(path, backend=backend, match=match,
    content_suite=suite, player_id="gru", display_name="GRU")
blanked = register_checkpoint(path, backend=backend, match=match,
    content_suite=suite, player_id="gru-blanked", display_name="GRU, memory blanked",
    blank_decision_memory=True)
```

The blanked registration carries `player_spec.blank_decision_memory=True`.
Scalar and shared inference zero all four memory tables before ordinary input
admission and preparation. The checkpoint, table shapes, current board, legal
actions and native game history stay intact. The two registrations have
distinct registration and serving keys, and are two Etude players; the default
`False` leaves the registration exactly as it was. A score near one half means
the policy plays the same without its memory. This measures use of the retained
history, not the incremental value of every fact already on the board.

Overflow or mismatched companion shapes raise a typed Capacity error with the
required counts. Keep the failed game's evidence, increase the relevant
capacities in a fresh configuration and start a compatible run. The log never
drops old choices, truncates details or retries without memory. Paid-run
overflow handling remains a pre-launch decision; retain and count every failed
game.

The changed v11 buffer layout uses normal world binding. Older checkpoints are
refused with their source revision and differing buffer names; run them from
that compatible revision. Increasing capacities also changes the input layout.
Do not rebind checkpoints to a new world or edit their saved shapes to bypass
admission. Memory does not change native legality, focus indexes or action
capacity.

## State, records and history

Current rules obligations belong in the current observation even when memory
is enabled. An active effect cannot depend on the policy remembering when it
began. Opponent memory supplements that state with evidence of past behavior;
it does not make hidden cards observable.

| Layer | Holds | Representation | Must not carry |
| --- | --- | --- | --- |
| Effective state | Current characteristics from `Game::characteristics`, zones, stack, suspended decisions, waiting triggers, hand minima, exact library positions, game-level disclosure | Object rows, library facts and the 64 × 49 relation table | Raw effect bookkeeping a consumer must replay to learn the current value |
| Shared records | Each active effect once: payload, duration, timestamp rank, controller, source definition, exact visible source and memberships | `effects`, `effect_details`, `effect_members`; local facts for copy exceptions and other per-object kinds | A recomputed layer result, or a merged record that loses a future difference |
| Recent events | Chronological public events, explicitly bounded | At most 32 rows of arrival, damage, life, cast, resolved and countered; optional and off by default | Anything needed to know the current rules state. Effect creation is not an event kind, and an activation adds no row |
| Decision memory | Whole-game viewer-safe decision records; optional own public signals | Unreleased: four decision tables and transformer/GRU summaries for board attention | Unavailable information, or a substitute for current native rules facts |

### Counterexamples

Each pair shows why effective state alone, or a merged record, loses a future.
Unless marked, the anchor shows the rule; it is not a paired-observation proof.

- **Expiry.** The same grant from an end-of-turn and an indefinite record agrees
  now and diverges at cleanup: `cr_514_2_an_end_of_turn_record_ends_at_cleanup`.
- **Source dependence.** Equal-looking records tethered to different sources
  diverge when one source leaves or changes control:
  `ty_lee_chi_blocker_locks_a_creature_down_while_you_control_her`. The effect
  header carries the exact source join
  (`test_source_dependent_shared_transport_has_exact_public_join`).
- **Timestamp order.** Add-then-remove and remove-then-add hold the same
  modifications with different results:
  `cr_613_7_a_grant_later_than_ability_loss_applies_and_an_earlier_one_does_not`.
- **Fixed membership.** A resolved effect's affected set excludes later
  entrants and returned objects, so it cannot become "your opponents'
  permanents": `shared_effects_preserve_duplicate_additive_memberships_and_later_entrants`.
  Static abilities are the opposite case: their scope does reach later entrants.
- **Copy versus effective state.** A copy takes copiable values and ordered
  exceptions, not counters, status or effects:
  `ran_and_shaw_copy_records_its_copy_effect_and_exception` and
  `ran_and_shaw_copy_ignores_the_sources_status_counters_and_effects`.
- **Ability loss.** Effective activated-ability order fixes activation and
  payment indices, so losing a grant can remove an action while power,
  toughness and keywords stay equal. The lost-abilities status alone is paired
  (`a_creature_that_lost_its_abilities_differs`, on a creature with
  no abilities to lose); the action difference is reasoned.
- **History eviction.** Paired native positions on real Commands fill the
  window with equal rows while their effect headers differ, and only one record
  survives cleanup:
  `test_effect_older_than_history_window_survives_only_in_current_state`. Its
  two grants also differ in payload and target, so it shows the window cannot
  separate them, not equal effective state.

### Safe combination

Merge two records only when they have identical future behavior across expiry,
sources, affected sets, later entrants, intervening effects and multiplicity.
Equal current effect is not enough. Repeated payloads may be factored into one
shared record because that loses nothing: the original timestamps and duplicate
edges remain. Semantic coalescing beyond that is unbuilt, and each case needs
this proof first.

### Alternatives

| Mechanism | Assessment |
| --- | --- |
| Larger fixed fact and relation tables | Admission control only. Removes neither duplication nor source-identity loss. |
| Typed shared records pooled into objects and frames | Shipped with pre-attention routing. Cheapest; a learned sum can still compress away a distinction that transport preserved. |
| Per-record token GRU with sparse joins | Tested against pooling on the earlier v7 input and not separated from it ([GPU report](../experiments/reports/state-representation-gpu.md)). The current input requires it for ordered programs, so that comparison does not cover the current effect path. |
| Variable-length game-state tokens | Unbuilt. Variable length alone does not reduce memory: the card encoder allocates padded tokens before packing. Ordered payloads suit token sequences, while unordered objects and effects need equivariant joins with explicit order features. |
| Split the relation table into typed stack, execution and trigger structures | Unbuilt. The table is a flattened typed structure, not wrong in itself. Its costs are repeated headers, capacity shared across unrelated structures, overloaded index and order features, and uniform pooling. A split must keep public stack rank, parent links, retained target eligibility and action continuation joins. |
| All objects, effects and frames as global attention tokens | Unmeasured. Moves attention from N² to (N + E + F)². Memberships need not become tokens. Compact transport alone does not justify it. |
| One whole-observation GRU or pooled vector | Rejected as the sole interface: it loses direct target addressing and imposes an arbitrary order. A summary may supplement object outputs. |
| Effective state plus a finite event window, omitting active effects | Rejected by the expiry, source and eviction cases. Recovering active records from the suffix is as hard as the missing state, and impossible once the event is gone. |
| Whole-game decision memory | Supplements complete typed state through summary tokens; see the intended interface above. |

### Unresolved and unmeasured

- **No stateless versus memory measurement exists.** Neither GPU arm carries
  memory, so "stateless first" is reasoned, not shown. Whether memory helps at
  all is open.
- The GPU comparison found no qualified plateau and no pool-versus-GRU
  difference (−1.95 points, 95% [−5.88, +1.98]). It covers the earlier v7 input on one matchup, so
  it selects no reducer.
- No paired observation test shows a lost activated ability removing an action,
  or a source that leaves and re-enters as the same physical card. Those cases
  rest on rules anchors and reasoning.
- Bucketed shapes and live tokens are measured for speed and output equality
  ([bucketed padding](#bucketed-padding)), not for learning: changed
  accumulation order is assumed not to move strength, and no matched-seed
  comparison has tested that.
- Learned sums preserve multiplicity but are not injective; whether that loses
  decision-relevant distinctions in practice is unmeasured.
- Whole-game memory throughput, capacity and learning effects remain unmeasured;
  the pilot must establish them before a default changes.

## What we know

- **The encoding hides no decision.** The learnability search found 0 clean
  input collisions, and the fixed suite runs on relevant PR updates
  ([learnability](../experiments/reports/learnability-search.md)).
- **The smallest model learned as well as larger ones at this scale.** Two
  seeds were not enough to rank sizes
  ([longer learning](../experiments/reports/longer-learning-017.md)).
- **Value-token and pooling variants are unresolved**
  ([value token](../experiments/reports/value-token-screen.md),
  [pooling follow-up](../experiments/reports/pooling-filter-followup.md),
  [three-seed depth/readout results](../experiments/reports/value-readout-252-results.md)).
- **No history input has shown a training benefit**, and the shared-effect
  transport is an engineering proof, not a learning result.

## Where the code lives

| Part | Code |
| --- | --- |
| Network | `ActorCritic` in [`manabot/backends/torch/policy.py`](../manabot/backends/torch/policy.py) |
| Observation admission and host preparation | [`manabot/policy/admission.py`](../manabot/policy/admission.py), [`prepare.py`](../manabot/policy/prepare.py) |
| Observation space and layout | [`manabot/env/observation.py`](../manabot/env/observation.py) |
| Native encoder and continuation | [`managym/src/agent/observation_encoder.rs`](../managym/src/agent/observation_encoder.rs), [`continuation.rs`](../managym/src/agent/continuation.rs) |
| Shared effects | [`manabot/env/shared_effects.py`](../manabot/env/shared_effects.py), [`managym/src/agent/shared_effects.rs`](../managym/src/agent/shared_effects.rs) |
| Recent events | [`manabot/backends/torch/_recent_events.py`](../manabot/backends/torch/_recent_events.py) |
| Card rules tokens | [`managym/src/cards/tokens.rs`](../managym/src/cards/tokens.rs), [`manabot/backends/torch/_semantic_cards.py`](../manabot/backends/torch/_semantic_cards.py) |
| Capacity and architecture receipts | [`manabot/model/agent.md`](../manabot/model/agent.md#capacity-identity-and-accounting) |
