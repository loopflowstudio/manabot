# Optional public recent-event input

`AgentSpec(recent_events=True)` enables shared policy/value context for the
selected semantic pack. `Model(...)` and `with_recent_events(...)` select it
independently of capacity, value aggregation/output and learning rule.
`TrainingRegime` derives `observation.policy_history_version=1`; ordinary
construction outside recipes must provide that observation setting explicitly.
Generic policies without a semantic catalog reject history-on.

This comparison **resolves unexecuted plans only**:

```python
from manabot.infra.hypers import AgentSpec
from manabot.training.experiments import Case, Experiment, Model
from manabot.training.presets import ataraxos_mtg_v1

comparison = Experiment(
    "recent-history-unexecuted",
    ataraxos_mtg_v1(),
    cases=(
        Case("off", (Model(AgentSpec(recent_events=False)),)),
        Case("on", (Model(AgentSpec(recent_events=True)),)),
    ),
).resolve()
```

No training allocation or measured learning benefit follows. Actual history
comparisons remain separate from this software delivery and follow the
value-token priority. Frozen ETU-91/106 checkouts are unchanged.

## Native authority and bounded work

`Game::emit` updates a **derived public projection** at the same append that
owns the committed rules ledger. It captures card definitions, zones and owner
while the event happens, using `permanent_object_ref` for permanent targets.
It never reconstructs permanent slot allocation or retrospectively resolves
private card identities. There is no independent rules/event authority.

The cache retains at most 32 rows. Public target lists expand in native target
order. Filtering precedes retention: excluded events do not consume positions.
A zone move expires any old context links before a public arrival is added;
returning the physical card to the same zone cannot restore an old link.
Departed public definitions remain available, including token definitions.
Private arrivals produce no row. Source/target references without a public
identity at emission remain unavailable. Damage and counter-source payloads carry
only a physical CardId, so these sources retain a public definition and owner
but no causal zone or current-object link: a delayed effect must not bind to a
returned incarnation. Owner is explicit; controller
attribution is **not provided** by this contract.

Each event append inspects at most 128 cached references plus its own target
list. Observation creation only shares the immutable cache via `Arc`: history-off
observations do not walk it or the ledger. History-on encoding visits at most 32
rows and joins their links to the bounded visible-card table. No work grows with
prior game length. Cache maintenance adds bounded append work even with history
off; this is not a zero-overhead or measured-throughput claim. Clone and shared-page
forks share the cache until mutation; undo marks restore it. A new game starts
empty. Diagnostic state injection clears history rather than inventing a transition;
only subsequent real events enter its suffix. Draining either event queue does not erase policy history.

Both acting and fixed-viewer observations carry this shared native suffix.
The vector collector's attached event-buffer shape selects the native v1 encoder;
Python `ObservationSpace.encode` uses the same native projection for ordinary
checkpoint players. Legacy observation JSON stays unchanged; a JSON-only historical
snapshot cannot manufacture this suffix. Retain learning tensors or replay the
native trace to reconstruct it. The collector pauses at the same next learner observation
as before. The model itself has no retained memory or reset hook.

## Tensor and model contract

History-off retains the historical `[E,7]` transition-event tensor. History-on
uses `[E,12]`, with `1 <= E <= 32` and a binary `[E]` validity mask. The saved
observation version and world/input schema bind its meaning; historical false/zero
configuration fields remain omitted. Historical parameter keys, initialization
order and forward equations stay unchanged.

Each row is kind, signed amount, then source and target references. Each reference
is `(definition+1, owner_role, zone+1, context_index, availability)`:

- Definition zero and owner/zone zero mean unavailable. Owner roles are 1 for
  the viewer and 2 for the opponent. Player references have no definition/zone.
- Context indexes gather a currently visible card or player row; `-1` means
  unavailable. They are routing indexes, never learned numeric inputs.
- Availability is 0 for unknown, 1 for an exact current object/player, and 2
  for a retained public definition with no current context link (departed or
  lacking exact originating-incarnation metadata).

Admitted kinds are public arrival (0), damage (1), life change (2), spell cast
(3), spell resolved (4) and spell countered (5). Countering uses the counter
source and countered card as target. Resolution/counter facts follow the native
final zone movement, so their zone is the resulting zone, not a fabricated stack
incarnation. Cast and public-arrival rows can both describe the same spell.

Definition transport IDs index the **existing complete semantic catalog**; their
numeric values are not neural features. Both reference definitions and available
current object embeddings enter the event encoder. Kind, owner, zone and
availability are categorical; amount uses units of 20 life, and relative ordinal
age uses `log1p(age)`, newest zero. A nonlinear projection makes order observable;
valid rows are mean-pooled and added to shared object embeddings. The same event context
also conditions focus-free actions when attention is disabled. Empty history
preserves the history-off result under identical shared weights. Gradients
reach both heads, the event projection and shared semantic/object parameters.

Invalid padding is sanitized before validation, lookup and nonlinear operations;
it contributes neither age nor normalization. Empty windows give exactly zero
event context. Missing/malformed tensors fail explicitly. Ordinary strict loading
and parameter receipts admit history weights and reject contradictory metadata.

## Evidence and limits

The bounded fixtures exercise ordinary semantic collection without an optimizer,
raw export/reload, checkpoint-player decisions, fixed-viewer parity, hidden-world
resampling, reset, order/card sensitivity and finite gradients. Native debug
fixtures cover direct token creation, departed references, zone re-entry,
consistent presentation-ID renaming, bounded retention and shared-page/undo
forks. A 2,048-event excluded prefix leaves the same shared suffix in repeated
observations; this is a structural check, not a throughput benchmark.

This is a suffix of admitted public facts, not complete decision history. It omits
passes, private draws/movements, private reveals, trigger announcements and the
presentation-only combat/death/turn families. Ordinary combat damage still enters
through native DamageDealt. It does not encode exact timing gaps, cross-event
identity of departed duplicate copies, historical controller changes, beliefs or
recurrent state. Existing public zones have no face-down visibility model; adding
one requires revisiting admission. No arbitrary Magic/history coverage, training
benefit, strength, arena cohort or chapter outcome is established.

The software gate passed 264 affected Python checks (one unsupported
attention-off/value-token combination skipped), native debug unit and branch
contracts, Clippy and focused formatting/lint checks. A direct comparison with
base `ab1f61c4` preserved all state-dict bytes, architecture receipts and outputs
for four semantic history-off configurations (scalar/WDL × attention off/on).
These are correctness fixtures, not scientific training or evaluation evidence.
