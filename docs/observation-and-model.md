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

1. **Embed.** Each visible object gets a typed projection. A card's rules tokens
   pass through a token embedding and a GRU, so the model can read what a card
   does as well as its name. Facts, links and effects add to the object they
   belong to.
2. **Mix.** Self-attention lets objects inform each other, with padding masked.
3. **Score actions.** Each action gathers the post-attention embeddings of its
   focus objects, and a small head gives it one logit. With `policy_film`, a
   pooled game state first scales each action embedding. Illegal slots are
   masked, so every decision type shares one action head.
4. **Estimate value.** Objects are projected, pooled by a mean over valid
   objects (`masked_mean`, as in the TLA pilot; the historical default averages
   every slot) and read by the value head. The output is a scalar or loss/draw/win logits.

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

V7 encodes each complete shared effect once and sums its vector into every
explicitly named affected object before attention. Duplicate memberships retain
multiplicity, and every affected object's loss reaches the shared encoder. Local
facts remain a separate sum. Neither path adds attention tokens.

Complete shared records retain duration, timestamp, controller and exact source
bindings that can change future behavior.

## Policy inputs

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

`ObservationConfig(policy_input_version=9)` retains v7's object, relation and
shared-effect layouts and adds `name_candidates[A,4]` at the ordinary action
capacity `A = max_actions`: one-based public name row, optional one-based
definition reference, then the one-based first and last name rows of a group
choice (zero means absent). Sorted names make the span a viewer-safe identity
of every member. No card/permanent row is fabricated for a name or group. These
rows suppress the positional mode feature, so identity comes from categorical
joins, never from magnitude.

The policy adds a trainable vector per name and, where present, the
definition's learning-program embedding from the table used by game objects. A
group's vector is the mean of its members' name vectors, computed exactly from
a prefix sum over the name table: every member receives the group's gradient,
and no member programs are encoded for a group prompt. V9 joins name vectors to
viewer-visible program references before card, known-hand and decklist pooling,
so equal-rule cards with different public names support different preferences;
closed native decklist rows contribute no identities. Action networks score
only legal rows, and rollouts pack legal action fields and complete behavior
probabilities with original coordinates.

The checkpoint binding includes the ordered naming vocabulary with definition
joins, `naming_representation="public-groups-v1"` and `naming_branching`, in
addition to tensor shapes and the content manifest. A changed vocabulary,
order, join or branching bound fails admission; there is no warm start or
remapping.

V9 composes the [v8 layer layout](#composable-layer-input-v8) with
`name_candidates[A,4]`. V1–v8 have no name identity: group and name choices
appear as ordinary modal choices, within their capacities. Missing-checkpoint defaults and fresh v6
defaults are unchanged; using v9 is explicit. The semantic observation/capture
contracts are 8/5, so earlier semantic world bindings are rejected for current
execution.

The default universe remains real non-token catalog names, including back faces.
A separate immutable vocabulary can hold validated names without playable card
definitions; this is exercised only by the offline large-vocabulary fixture.
This does not claim that the supported catalog exhausts every name legal under
Magic's rules, and does not broaden deck admission.

### Current continuation input

Fresh `ObservationConfig` uses policy input v6 with serialized `max_facts=512`.
V6 retains v5's fact columns and all other tensor layouts; v5 keeps exactly
128 fact rows. Facts remain one row per object/detail with source, duration and
timestamp rank. The model divides fact timestamp ranks by **128** in both
versions; capacity does not change the numeric scale. Larger finite buffers
still reject overflow rather than dropping state. Configuration owns allocation,
native scalar/vector buffer shapes, model shape checks and checkpoint receipts.

A native `relations[64,49]` float32 buffer carries ordered stack entries,
announced targets and clauses, chosen modes and abilities, kicker, reserved
sacrifices, suspended payments and waiting triggers. V5–v8 player, card and
permanent widths are 73/56/38. Native buffers bind relation semantics by width
(32 for v2/v3, 33 for v4, 49 for v5–v8); validity stays at column 31. The fact
axis binds capacity; at 128 rows v5 and v6 encode identically, while the saved
configuration still distinguishes their identities. Checkpoints omitting the
version load as v1 through `ObservationConfig.from_checkpoint`; legacy configs
omit `max_facts`, preserving historical architecture hashes.
`agent/continuation.rs` projects the continuation frames and
`agent/observation_encoder.rs` the fact rows once per observation;
`manabot/env/observation.py` is the matching reference encoder. Only the
acting viewer receives private continuation; overflow fails rather than
truncating. Transport IDs are joins to the card catalog and object table,
never numeric features. At `Decision::ManaChoice` the activation source is
unavailable because rules state has discarded it.
Waiting delayed abilities (frame kind 10) carry event codes independently of
frequency and expiry. In v5–v8, relation row kind 9 carries frequency in column
28 (1 once, 2 each matching event); column 48 retains cleanup expiry. Remembered
objects remain kind-8 reference rows, separate from event context. Historical
captures without a frequency field retain unknown frequency rather than
inferring it from the event. V1–v4 omit these delayed facts.

Encode-only benchmark: `cargo run --release --features benchmark --manifest-path managym/Cargo.toml --example bench_observation`.
Matched full-vector benchmarks use `scripts/bench_breakdown.py --matched-corpus`
and `scripts/bench_observation_pair.py` on a release extension built with the
`benchmark` feature.

### Input v7 capacity

Select `ObservationConfig(policy_input_version=7)` for the new transport: 128
headers × 18 columns, 512 detail tokens × 3, and 2,048 memberships × 2, plus a
configurable local-fact table (128 rows by default). Overflow fails explicitly.
The retained 140/270-fragment Drill positions become 7/9 complete payloads and
70/135 memberships. All original timestamps and duplicate edges survive; no
rules records are deleted or semantically coalesced.

The three padded effect arrays use 31,744 bytes per observation; local facts add
6,144. This exceeds v6/256's 12,288-byte fact array, although the retained
270-fragment case needs only 1,944 active effect bytes. The layout, routing and
sum/normalization tradeoff are in [`manabot/model/shared_effects.py`](../manabot/model/shared_effects.py)
and [shared effect inputs](../manabot/README.md#shared-effect-inputs). These
engineering proofs establish no learning-quality advantage. Persistent memory
remains future work. Fresh defaults remain v6; frozen v1–v6 checkpoints keep
their meanings, and the TLA v10 pilot trains on v7.

### Composable layer input v8

For a new run requiring composable layer programs, explicitly select v8 before
constructing the collector and model. Fresh configurations remain v6. For example,
this prepares a recipe and its model input without starting training:

```python
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ObservationConfig
from manabot.recipes.local import local_smoke

recipe = local_smoke(steps=1)
recipe.observation = ObservationConfig(policy_input_version=8)
space = ObservationSpace(recipe.observation)
```

Use that same `space` for `SeatRoutedCollector` and
`ActorCritic(space, recipe.agent)`. The serialized `TrainingRegime` retains the
selected input. V8 carries structured granted, copy-exception and captured
continuation programs, changed colors and keyword-counter timestamps. V7 remains
available for representable shared effects; it cannot encode every new layer
payload.

The shared `validate_policy_input` boundary rejects unrepresentable v1–v7 layer
payloads in both native and Python encoders before writing encoded observations,
selecting an action prefix or running inference. The diagnostic names the
selected version, for example: `policy input v6 cannot represent layer payload;
select v8`. A starting state may encode successfully and a later state fail;
this is observation admission, not whole-game preflight or rollback of an action
already executed. Defined-card eligibility, input representability and
checkpoint/world admission are separate checks.

On rejection, preserve the failed run's configuration and evidence. For new work
requiring these payloads, configure v8 and construct a compatible model. Load saved
observation configurations through `ObservationConfig.from_checkpoint` and keep
their saved versions, omissions and model shapes; an absent version means v1.
Do not edit a checkpoint's input version or reshape its weights to bypass the
error. Saved-policy evaluation uses that policy's input and normal world
admission. Successful architecture loading alone does not admit a checkpoint to
the current world; see [world identity](../manabot/README.md#world-identity).
Selecting v8 does not restart a failed run or swap its live policy.

V7 and v8 share `effects[128,18]` headers and `effect_members[2048,2]` joins.
V7 has `effect_details[512,3]` and defaults to `facts[128,12]`; v8 has
`effect_details[4096,3]` and defaults to `facts[512,12]`. Local fact capacity is
configurable through `max_facts` in either version. Headers own bounded slices
of details and members. Allocate and collect every key from the selected
`ObservationSpace`; overflow rejects rather than truncating a payload.

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

## What we know

- **The encoding hides no decision.** The learnability search found 0 clean
  input collisions, and the fixed suite runs on relevant PR updates
  ([learnability](../experiments/learnability-search.md)).
- **The smallest model learned as well as larger ones at this scale.** Two
  seeds were not enough to rank sizes
  ([longer learning](../experiments/longer-learning-017.md)).
- **Value-token and pooling variants are unresolved**
  ([value token](../experiments/value-token-screen.md),
  [pooling follow-up](../experiments/pooling-filter-followup.md)).
- **No history input has shown a training benefit**, and the v7 shared-effect
  transport is an engineering proof, not a learning result.

## Where the code lives

| Part | Code |
| --- | --- |
| Network | `ActorCritic` in [`manabot/model/actor_critic.py`](../manabot/model/actor_critic.py) |
| Observation space and reference encoder | [`manabot/env/observation.py`](../manabot/env/observation.py) |
| Native encoder and continuation | [`managym/src/agent/observation_encoder.rs`](../managym/src/agent/observation_encoder.rs), [`continuation.rs`](../managym/src/agent/continuation.rs) |
| Shared effects | [`manabot/env/shared_effects.py`](../manabot/env/shared_effects.py), [`managym/src/agent/shared_effects.rs`](../managym/src/agent/shared_effects.rs) |
| Recent events | [`manabot/model/recent_events.py`](../manabot/model/recent_events.py) |
| Card rules tokens | [`managym/src/cards/tokens.rs`](../managym/src/cards/tokens.rs), [`manabot/model/semantic_cards.py`](../manabot/model/semantic_cards.py) |
| Capacity and architecture receipts | [`manabot/model/agent.md`](../manabot/model/agent.md#capacity-identity-and-accounting) |
