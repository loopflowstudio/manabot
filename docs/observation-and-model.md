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
- [Proposed: state, records, history and memory](#proposed-state-records-history-and-memory)
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
| Known hand and disclosed decklists | What this seat is allowed to know. |
| Legal actions and focus objects | Every offered action and up to two objects it refers to. |
| Card definitions | Typed rules tokens derived from each card's native definition. |

Every mechanic the engine implements is visible here, in the LLM text
observation and in the learning tokens, in the same change that adds it. A
frozen input version never silently falls behind the rules.

## What the network does

Every observation batch is admitted once, on CPU, before any forward:
`admit_observation(agent, obs)` checks shapes, references, programs and the
legal-action mask and returns an `AdmittedObservation`. The forward reads only
admitted input, raises `TypeError` for a plain tensor dictionary and does no
validation of its own
([observation admission](../manabot/model/agent.md#observation-admission)).

1. **Embed.** Each visible object gets a typed projection. A card's rules tokens
   pass through a token embedding and a GRU, so the model can read what a card
   does as well as its name. Facts, links and effects add to the object they
   belong to.
2. **Mix.** Self-attention lets objects inform each other, with padding masked.
   [Live-token attention](#live-token-attention-profiling) attends over valid
   objects only; it replaced the pooled token stream
   ([token-stream report](../experiments/token_stream.md)).
3. **Score actions.** Each action gathers the post-attention embeddings of its
   focus objects, and a small head gives it one logit. With `policy_film`, a
   pooled game state first scales each action embedding. Illegal slots are
   masked, so every decision type shares one action head.
4. **Estimate value.** Objects are projected, pooled by a mean over valid
   objects (`masked_mean`, as in the TLA pilot; the historical default averages
   every slot) and read by the value head. The output is a scalar or loss/draw/win logits.
   `value_token` and `attention_readout` replace the mean with a learned summary
   ([readout comparison](../experiments/value-readout-252.md)).

The GRU encodes a card's rules program, not game history. The network has no
recurrent game memory, and the optional recent-event encoder is off by default.
Each decision is judged from its current observation alone.

### Scoring legal actions

One head scores every kind of decision, from casting and targeting to blocking
and paying costs. The action's own features are concatenated with the
post-attention embeddings of the objects it names.

From `_add_focus` in
[`manabot/model/actor_critic.py`](../manabot/model/actor_critic.py):

```python
focus_embeds = torch.gather(post_attention_focus_objects, 2, focus_indices)
focus_embeds = focus_embeds * valid_mask
focus_flat = focus_embeds.reshape(B, max_actions, -1)
```

From `forward_distribution` in the same module:

```python
logits_before_mask = self.policy_head(informed_actions).squeeze(-1)
if self.debug:
    # Save raw logits for offline diagnosis (scripts/diagnose_*).
    self.last_raw_logits = logits_before_mask.detach().cpu()
logits = logits_before_mask.masked_fill(obs["actions_valid"] == 0, -1e8)
return logits, self._value_from_objects(encoded, validity)
```

- A focus slot of −1 means the action names fewer than two objects; its
  contribution is zeroed rather than pointing at a padding row.
- Masking to −1e8 gives an illegal slot zero probability, so the learner's
  ratios and KL terms sum over legal actions only.
- `max_actions` is a declared capacity. Overflow is an error by default; the
  TLA pilot explicitly keeps the first 64 legal actions, while the engine and
  Academy keep the complete set
  ([policy action capacity](../manabot/README.md#policy-action-capacity)).

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

### Bucketed padding (profiling)

Each buffer arrives padded to its declared capacity, but a real batch uses
little of it. Across 300 captured collection batches the median batch needed 59
of 512 fact rows, 51 of 2,048 effect members and 12 of 64 actions. By default
the network still processes every slot, and skips lookups for an axis that is
empty in the batch, so tensor shapes change from batch to batch.

Bucketed padding is an opt-in alternative meant for comparing accelerator
cost. After an observation is admitted, the network trims each variable axis
(facts, effect headers, details and members, relations, actions, events, cards,
permanents, known hands and decklists) to a bucket: the next power of two at or
above what the batch uses, at least 8 and at most the declared capacity. What a
batch uses includes every row a reference can name, so trimming removes only
empty rows nothing refers to. Trimming cards and permanents renumbers object
references; it is skipped under `historical_mean` value aggregation, whose
average includes padding, and with belief rows. Logits keep their full action
capacity. Inference measures the buckets on the host while preparing a batch,
so the device forward reads nothing back to find them; learning measures them
with one read per forward.

Inside the network, fact ids are looked up at the fact axis's shape instead of
for selected rows, and learning encodes the complete card catalog instead of
the definitions a batch references. Shared-effect routing, fact programs and
legal-action selection keep their packed, data-dependent layouts. A forward
therefore takes one of a few shapes per input axis instead of a new shape for
nearly every batch, and does much less work than padding everything to
capacity. The complete catalog is more work than the default path's
referenced definitions: captured Elemental learner batches referenced 65 to
154 of 360 definitions, so learning encodes 2.3 to 5.8 times as many catalog
tokens. On CPU, where the recurrent catalog encoder dominates learning, a
learner forward and backward is therefore slower than on the default path,
most of all for small decks.

A training stage turns it on in its execution settings, next to the device:

```python
from manabot.training.regime import Execution

execution = Execution(device="cuda", bucketed_shapes=True)
```

The stage's collection and learner policies, including a frozen opponent
(`load_checkpoint(..., bucketed_shapes=True)`), then use bucketed shapes; its
evaluation and search do not. `uv run manabot train --recipe baseline
--bucketed-shapes` sets it on the CLI recipes ([run training](training.md#run-training)),
and code that builds a policy directly passes
`ActorCritic(..., bucketed_shapes=True)`. An unset field is omitted from the
serialized regime, so existing regime hashes are unchanged; a stage that sets
it records `"bucketed_shapes": true` in the regime in `run.json`. To compare
the two paths, run the same regime once with each setting.

The setting changes how a forward is computed, not the model. It changes no
`ModelConfig` field, checkpoint, world binding or policy input, so a checkpoint
trained with either setting loads and runs with the other. On CPU,
action probabilities, values, losses and gradients agree with the default path
up to the order of floating-point sums. A batch that references no card
definition is the one difference in gradient support: the catalog encoder then
receives exactly zero gradients instead of none. Capacity overflow and
malformed inputs are rejected with the same error and message as on the
default path.

### Live-token attention (profiling)

Bucketed padding still attends over every slot in a bucket, and most of those
slots are empty: a batch whose largest board holds 9 permanents pads every
observation to 16 permanent slots, including those with two. Live-token
attention is an opt-in layout that computes object attention and the
per-object layers only for valid objects, so cost follows what the board holds
rather than what the bucket allows.

```python
from manabot.training.regime import Execution

execution = Execution(device="cuda", bucketed_shapes=True, live_tokens=True)
```

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

Attention within each observation uses `scaled_dot_product_attention` on
jagged nested tensors. On CUDA that runs the fused variable-length kernels; on
CPU it falls back to one attention call per observation, which is correct but
slow, so the setting is meant for CUDA.

Packing needs one host read per forward, the batch's total number of valid
objects, to size the packed sequence. Collection measures it on the host while
preparing a batch, as it does for buckets, so the device forward reads nothing
back; learning reads it once per forward. The layout adds no other
data-dependent reads.

The stage's collection and learner policies, including a frozen opponent
(`load_checkpoint(..., live_tokens=True)`), use the layout; evaluation and
search do not. Code that builds a policy directly passes
`ActorCritic(..., live_tokens=True)`. Like `bucketed_shapes`, the field is
serialized only when set. The setting changes how a forward is computed, not
the model: no `ModelConfig` field, checkpoint, world binding or policy input
changes, and a checkpoint trained with it loads and runs without it. Action
probabilities, values, losses and gradients agree with the padded path to
within 1e-5 on CPU; on CUDA they differ only by accumulation order, as bucketed
shapes do. On an L4 at w384l8, live tokens over buckets take a learner step
from 386 ms (bucketed shapes) to 202 ms
([live-token attention](../experiments/live_token_attention.md)).

## Policy inputs

Main carries policy input v9, the fresh default, and explicit input v10
([below](#input-v10-turn-history-and-powerstone-mana)).
`ObservationConfig.policy_input_version` is always serialized. A checkpoint,
replay or saved configuration bound to any other version, or omitting it,
raises `UnsupportedPolicyInput`, which names the revision to run it from
([`ac601834d`](https://github.com/loopflowstudio/manabot/tree/ac601834d) is the
last revision supporting v1–v8). Load saved configurations through
`ObservationConfig.from_checkpoint`; do not edit a checkpoint's version or
reshape its weights to get past the error.

A layout change on main does not bump the version. Every checkpoint records the
revision and Etude run that wrote it, and its world binding holds each buffer's
shape. A checkpoint from an earlier layout is refused with the buffers that
differ and the revision to run it from. Loading an architecture does not
admit a checkpoint to the current world; see
[world identity](../manabot/README.md#world-identity).

### Input v10: turn history and Powerstone mana

Three TLE bonus-sheet cards read state v9 has no place for:

| Card | State v10 adds |
| --- | --- |
| Bloodchief Ascension | Life each player has lost this turn |
| Brought Back | Which graveyard cards were put there from the battlefield this turn |
| Cityscape Leveler | Mana that can't be spent to cast nonartifact spells (its Powerstone tokens) |

Select v10 explicitly before constructing the collector and model:

```python
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ObservationConfig
from manabot.recipes.local import local_smoke

regime = local_smoke(steps=1).model_copy(
    update={"observation": ObservationConfig(policy_input_version=10)}
)
space = ObservationSpace(regime.observation)
```

Use that `space` for `SeatRoutedCollector` and `ActorCritic(space, regime.agent)`;
the serialized `TrainingRegime` keeps the selection. V10 is v9 plus this state,
in the same native scalar, vector and Python encoders, the LLM text observation
and the learning tokens.

V10 adds two buffers to the v9 layout; their presence is what selects v10 in
the native encoders:

| Key | Shape | Holds |
| --- | --- | --- |
| `player_turn_state` | 2 × 8 | Per seat, agent first: life lost this turn / 20, spells cast this turn / 10, then mana that can't be spent to cast nonartifact spells by W/U/B/R/G/C / 10 |
| `card_turn_state` | 2 × 60 | Per seat card row (aligned with `semantic_cards`): 1 for a graveyard card put there from the battlefield this turn |

The model appends each seat's row to its player row and each card value to its
card row before embedding them.

V9 and its checkpoints are unchanged: w17 checkpoints keep loading, and fresh
configurations stay v9. A v9 encoder refuses a position that needs v10 state:
a card visible to the viewer reads life lost this turn or graveyard cards put
there from the battlefield and that state is present, or a pool holds mana
that can't be spent to cast nonartifact spells. `managym.input_v10_state(obs)`
names the needed state. The Python encoder raises `UnsupportedPolicyInput`
naming v10 and the state, and the native encoders raise `ValueError` with the
same reason, before writing any buffer. A v9 run over decks with these cards
stops at the first such position, so train on them with v10.
A game, replay or Academy session needs no policy input and plays these cards
under any configuration. A v9 checkpoint cannot be widened to v10; train a new
one.

### Layout

A fresh `ObservationConfig` gives these float32 buffers. Capacities are
configuration; widths are the layout.

| Key | Shape | Holds |
| --- | --- | --- |
| `agent_player`, `opponent_player` | 1 × 73 | Player state |
| `agent_cards`, `opponent_cards` | 60 × 56 | Cards outside the battlefield |
| `agent_permanents`, `opponent_permanents` | 40 × 38 | Permanents |
| `relations` | 64 × 49 | Stack entries, targets, clauses, modes, costs, suspended payments, waiting triggers |
| `facts` | 512 × 12 (`max_facts`) | One row per object and detail of an open vocabulary |
| `effects`, `effect_details`, `effect_members` | 128 × 18, 4096 × 3, 2048 × 2 | Each active effect once, with its payload tokens and affected objects |
| `decklists`, `known_hand`, `semantic_cards` | 2 × 60 × 5, 2 × 60 × 2, 2 × 60 | Disclosed lists, known-hand minima, definition joins |
| `actions`, `action_focus`, `name_candidates` | 512 × 22, 512 × 2, 512 × 4 | Legal actions, the objects each names, name choices |
| `events` | 32 × 7, or 32 × 12 with recent events | Public events |

Allocate and collect every key from the supplied `ObservationSpace`, on both
the scalar and vector paths. Fact, relation, effect and action overflow is a
typed Capacity error, never truncation; the one declared exception is the TLA
pilot's first-64 action prefix.
`agent/continuation.rs` projects the relation frames and
`agent/observation_encoder.rs` the fact rows once per observation;
`manabot/env/observation.py` is the matching reference encoder. A buffer that
is missing or has the wrong shape fails before any write. Column names and
value encodings for every buffer come from `managym.observation_layout()`
([reading arrays by name](engine.md#reading-arrays-by-name)).

### Relations and facts

Relation validity is column 31. Only the acting viewer receives private
continuation. Transport IDs are joins to the card catalog and object table,
never numeric features. At `Decision::ManaChoice` the activation source is
unavailable because rules state has discarded it.

Waiting delayed abilities (frame kind 10) carry event codes independently of
frequency and expiry. Relation row kind 9 carries frequency in column 28 (1
once, 2 each matching event); column 48 retains cleanup expiry. Remembered
objects are kind-8 reference rows, separate from event context. Ordered control
schedules and delegated decisions use kinds 13 and 14.

Fact rows carry source, duration and timestamp rank. The model divides
timestamp ranks by 128, independent of fact capacity.

### Shared effects

Each active effect is one `effects` header owning bounded slices of
`effect_details` and `effect_members`. Details hold the payload as learning
tokens, including granted, copy-exception and captured continuation programs,
changed colors and keyword-counter timestamps; members keep one-based public
object joins, and duplicate memberships survive. No rules record is deleted or
semantically coalesced. The layout, routing and sum/normalization tradeoff are
in [`manabot/model/shared_effects.py`](../manabot/model/shared_effects.py) and
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
and no member programs are encoded for a group prompt. Name vectors join
viewer-visible program references before card, known-hand and decklist pooling,
so equal-rule cards with different public names support different preferences;
closed native decklist rows contribute no identities. Action networks score
only legal rows, and rollouts pack legal action fields and complete behavior
probabilities with original coordinates.

The checkpoint binding includes the ordered naming vocabulary with definition
joins, `naming_representation="public-groups-v1"` and `naming_branching`, in
addition to tensor shapes and the content manifest. A changed vocabulary,
order, join or branching bound fails admission; there is no warm start or
remapping. The semantic observation/capture contracts are 8/5, so earlier
semantic world bindings are rejected for current execution.

The default universe remains real non-token catalog names, including back faces.
A separate immutable vocabulary can hold validated names without playable card
definitions; this is exercised only by the offline large-vocabulary fixture.
This does not claim that the supported catalog exhausts every name legal under
Magic's rules, and does not broaden deck admission.

### Optional recent public events

`ModelConfig(recent_events=True)` (or `with_recent_events` / `ModelOverride`)
adds shared policy/value context from a bounded suffix of public events; the
regime derives `observation.policy_history_version=1`. `Game::emit` maintains
the projection at the same append as the rules ledger, keeping at most 32
rows; private arrivals produce no row, and owner (not controller) is recorded.
History-on uses an `[E,12]` tensor (`1 <= E <= 32`) with a validity mask:
kind (public arrival, damage, life change, spell cast, resolved, countered),
signed amount, and source/target references `(definition+1, owner_role,
zone+1, context_index, availability)`. Valid rows are mean-pooled and added to
shared object embeddings; an empty window gives exactly zero context.
History-off keeps the historical `[E,7]` tensor and parameters. This is not
complete decision history, and no training benefit has been measured.

## Proposed: state, records, history and memory

**Proposed, not accepted and not built.** This section records a contract under
review. The rest of this guide describes shipped behavior; nothing here changes
a tensor, model, default or checkpoint.

### Recommendation

Give the policy complete typed current state and no persistent policy state as
the first arm. A rules obligation that is still in force belongs in the
observation, where the network reads it, not in a memory that must have seen it
begin. A reducer that runs within one decision, such as the GRU over effect
tokens, resets every forward call and is not memory. Memory is a later arm for
inference from past behavior, compared against this stateless reference.
"Sufficient" here means the viewer's decision-relevant observable state; it
does not make hidden cards observable or the game fully observed.

| Layer | Holds | Shipped | Must not carry |
| --- | --- | --- | --- |
| Effective state | Current characteristics from `Game::characteristics`, zones, stack, suspended decisions, waiting triggers, known-hand minima, deck disclosure | Object rows and the 64 × 49 relation table | Raw effect bookkeeping a consumer must replay to learn the current value |
| Shared records | Each active effect once: payload, duration, timestamp rank, controller, source definition, exact visible source, exact memberships | `effects`, `effect_details`, `effect_members`; local facts for copy exceptions and other per-object kinds | A recomputed layer result, or a merged record that loses a future difference |
| History | Chronological public events, explicitly bounded | At most 32 rows of arrival, damage, life, cast, resolved and countered; optional and off by default | Anything needed to know the current rules state. Effect creation is not an event kind, and an activation adds no row |
| Memory | Inference from everything the seat has been shown | Nothing | A fact the current observation should carry, or anything the viewer may not see |

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
  entrants and returned incarnations, so it cannot become "your opponents'
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
| Per-record token GRU with sparse joins | Tested against pooling on the earlier v7 input and not separated from it ([GPU report](../experiments/state-representation-gpu.md)). The current input requires it for ordered programs, so that comparison does not cover the current effect path. |
| Variable-length game-state tokens | Unbuilt. Variable length alone does not reduce memory: the card encoder allocates padded tokens before packing. Ordered payloads suit token sequences, while unordered objects and effects need equivariant joins with explicit order features. |
| Split the relation table into typed stack, execution and trigger structures | Unbuilt. The table is a flattened typed structure, not wrong in itself. Its costs are repeated headers, capacity shared across unrelated structures, overloaded index and order features, and uniform pooling. A split must keep public stack rank, parent links, retained target eligibility and action continuation joins. |
| All objects, effects and frames as global attention tokens | Unmeasured. Moves attention from N² to (N + E + F)². Memberships need not become tokens. Compact transport alone does not justify it. |
| One whole-observation GRU or pooled vector | Rejected as the sole interface: it loses direct target addressing and imposes an arbitrary order. A summary may supplement object outputs. |
| Effective state plus a finite event window, omitting active effects | Rejected by the expiry, source and eviction cases. Recovering active records from the suffix is as hard as the missing state, and impossible once the event is gone. |
| Persistent memory on top of complete typed state | Viable later arm. Not a repair for lossy rules state, and it requires the lifecycle below. |

### Memory lifecycle

A persistent arm needs all of these together, and a random-weight forward smoke
is not evidence for any of them:

- **Ownership.** State belongs to match instance, stream generation, viewer seat
  and policy generation, never to a vector slot or whichever player acted last.
- **Reset.** A new game, rematch or slot reuse starts empty; turn boundaries
  preserve state. A rejected or stale Command commits no transition, and forced
  decisions stay skipped.
- **Seat.** Each seat has independent state. A seat swap routes to the
  corresponding state and never relabels one player's memory as another's.
- **Hidden information.** Updates read only that seat's authorized observations
  and the public events since its committed cursor. Opponent private frames,
  sampled possible worlds and future trajectory facts never enter.
- **Eviction.** Current effects, known-hand minima, disclosure and pending
  obligations stay in native state regardless of memory or window. Old history
  rows disappear explicitly and never change a legal decision.
- **Fork.** A fork clones state and cursor, then evolves independently. Replay
  reproduces the same ordered inputs and weights. A departed object's handle
  never reconnects to a returned physical card.
- **Restoration.** Exact restore stores hidden state, cursor, viewer and match
  identity, observation schema, model hash and relevant RNG with committed
  learning state. Weights alone are not a mid-game snapshot. Without the state,
  restoring needs replay from the true initial viewer prefix; a missing prefix
  is an explicit failure, never a zero-state fallback or a last-32-row burn-in.
- **Training and serving.** Training uses contiguous same-seat sequences, episode
  and padding masks, and declared truncation and burn-in. Loss reads admitted
  learning rows only; burn-in adds no opponent reward. Parameter changes need a
  declared refresh policy. Collection and serving call one transition API.

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
- Fixed padded capacity against packed variable length is open. Collection
  batches use a small fraction of each capacity
  ([bucketed padding](#bucketed-padding-profiling)); how that affects
  accelerator cost has not been measured.
- Learned sums preserve multiplicity but are not injective; whether that loses
  decision-relevant distinctions in practice is unmeasured.
- Sequence training changes collection, sampling and recovery, and that cost has
  not been estimated.

## What we know

- **The encoding hides no decision.** The learnability search found 0 clean
  input collisions, and the fixed suite runs on relevant PR updates
  ([learnability](../experiments/learnability-search.md)).
- **The smallest model learned as well as larger ones at this scale.** Two
  seeds were not enough to rank sizes
  ([longer learning](../experiments/longer-learning-017.md)).
- **Value-token and pooling variants are unresolved**
  ([value token](../experiments/value-token-screen.md),
  [pooling follow-up](../experiments/pooling-filter-followup.md),
  [three-seed depth/readout results](../experiments/value-readout-252-results.md)).
- **No history input has shown a training benefit**, and the shared-effect
  transport is an engineering proof, not a learning result.

## Where the code lives

| Part | Code |
| --- | --- |
| Network | `ActorCritic` in [`manabot/model/actor_critic.py`](../manabot/model/actor_critic.py) |
| Observation admission | [`manabot/model/admission.py`](../manabot/model/admission.py) |
| Observation space and reference encoder | [`manabot/env/observation.py`](../manabot/env/observation.py) |
| Native encoder and continuation | [`managym/src/agent/observation_encoder.rs`](../managym/src/agent/observation_encoder.rs), [`continuation.rs`](../managym/src/agent/continuation.rs) |
| Shared effects | [`manabot/env/shared_effects.py`](../manabot/env/shared_effects.py), [`managym/src/agent/shared_effects.rs`](../managym/src/agent/shared_effects.rs) |
| Recent events | [`manabot/model/recent_events.py`](../manabot/model/recent_events.py) |
| Card rules tokens | [`managym/src/cards/tokens.rs`](../managym/src/cards/tokens.rs), [`manabot/model/semantic_cards.py`](../manabot/model/semantic_cards.py) |
| Capacity and architecture receipts | [`manabot/model/agent.md`](../manabot/model/agent.md#capacity-identity-and-accounting) |
