# MTG-242: port onto current main — 2026-11-12

Jack Heart selected starting from main and porting the important existing work,
rather than reconciling the old implementation line by line. This supersedes the
old integration strategy, not the intended mana outcome or review-only boundary.

## Preserved sources

- Complete previous implementation and working notes: `e5b71f269e5ba4a7b4f75ba641b7ca59b95a470b`.
- Inspected main: `e5897c7abcba207b7224922d9363bd004b6b2eb8`.
- Historical measurement receipts: the old commit's `research/mana-risk-2026-11/`.
  Their timings describe those sources only and must not become current claims.

Keep this Task's branch/checkout and recorded history. Use Loopflow's managed
sync operations; establish main's implementation as the integration baseline
without deleting the old commits, resetting away contributions or launching a
second Task. Do not copy the old tree wholesale. Main's removed modules and
historical compatibility branches stay removed. Preserve these port instructions
locally through the sync even though main deleted old scratch/questions.md.

## Port in this order

1. Inventory main's existing payment/resource/completion features against the old
   acceptance tests. Port only missing behavior; keep main's exact source binding,
   shared completion, selected costs, layer invalidation, and current mana kinds
   (including turn-state restrictions such as Powerstone mana).
2. Replace the finite executable restriction vocabulary with typed predicates and
   explicit spending context in the current authority. Unknown restrictions must
   not become unrestricted. Preserve exact resource remainders and parent payment
   feasibility. Adapt bounded repeatable sources and post-cost variable production
   only where main does not already supply their behavior.
3. Port the strongest real-card and constructed Command/fork witnesses: restrictions
   and purposes, shared taps, once-turn/repeatable uses, variable output, nested
   Prism/Treasure, and actual suspended sacrifice/linked-return completion. Check
   cost commitment, delayed output and parent resumption exactly once. Reuse main's
   common helpers instead of restoring removed test/API surfaces.
4. Carry every new distinction through current native/Python projection, scalar/
   vector buffers, tokens, text, model and Academy. The old branch's v8/v9 numbering
   is historical, not a layout reservation on main. Prefer the current semantic
   record representation where it can express the full information. Do not revive
   state_records.py or old encoder forks. Validate checkpoint/world admission and
   do not silently reinterpret a saved model. If a layout/world-freeze decision is
   still necessary after inspecting the current record contract, specify the exact
   change and compatibility consequence for review; do not drop observations.
5. Port only optimizations justified on the current payment model. Use current
   profiles and main comparisons; fixed-payment fast paths must agree with the
   exact debug oracle. Keep full witness equality, legal choices and model
   class-label invariance. Compare current model joins before adding machinery.
6. Run current affected debug, projection/model and admission checks; rebuild the
   extension using current repository commands. Measure equivalent workloads
   against pinned current main, retaining tails and digests, and run Engine Budget.
   Old microbenchmarks prove neither a current speedup nor current correctness.

Fully explicit selection of every land and pool allocation remains separate work,
as clarified in this conversation. No merge, training experiment, artificial load
or speculative whole-engine rewrite is authorized. Publish a coherent reviewable
change with the current measured limits; do not waive the acceleration request.

## Launch state

Jack Heart explicitly requested "assign it" and "and start it" on 2026-11-12.
The Task was claimed for Jack Heart on this machine. Prior Graphs are stopped;
launch the main-first port through the code Graph, stopping at interactive review.

## Inventory on main `34980fdaa` — 2026-11-12

Baseline: `af7d4e073` merges main with main's tree exactly; old work stays in
history at `e5b71f269`.

Main already supplies (do not port): selected tap obligations with candidate
sets and counted taps (`Obligation::Tap`), once-turn/exhaust limited uses,
life and sacrifice reservations, class-aware pool remainders
(`payment_options`), the finite-resource registry guard including counted
taps, and shared completion: `advance_mana_upfront` commits taps/life/mana,
then runs sacrifice and `mana_suffix` as a child frame through
`run_before_decision` before resuming `ManaChoice` with the original parent.
Nested Prism/Treasure completion and fork isolation are witnessed in
`cards/tests/cards/roe.rs` (`prism_*`, `audit_prism_treasure_*`). Variable
outputs exist as `Effect::AddOneColorManaValue` (White Lotus Tile, Avatar
Kyoshi; tests in `tla.rs`) and `AddMana.amount`, read when offered
(`current_mana_output`). No current card's output depends on its own costs, so
post-cost variable output stays a documented limit, not ported.

Missing on main (port): `ManaUse` is still a 3-subtype bit mask
(`SPEND_SUBTYPES`), so a restriction naming any other subtype becomes an empty
mask that nothing can spend, and a purpose drops every other subtype. Powerstone
is a special boolean. Resolving costs, casts and activations are not distinct
contexts beyond masks.

### Port decisions

- `state/mana.rs`: `SpendPredicate` (cast subtype, activate-source subtype,
  cast a spell without a card type), `ManaRestriction::{Only, Except}` and
  `SpendingContext::{Resolving, Cast, Activate { zone }}` with an immutable
  shared characteristics snapshot. Powerstone is `Except({CastWithoutType(Artifact)})`.
- Observation data (`RestrictedManaData`) mirrors the typed restriction totally.
- Policy tensor: existing Lesson/Shrine/Ally columns and v10 Powerstone columns
  stay the frozen projection. A restriction those columns cannot represent is a
  typed refusal before buffers are written, never a silent drop. A fact-row
  encoding needs a new `FactKind`, which changes `FactEncoder.kind`'s parameter
  shape: that is a world-freeze decision left for review (see questions.md).

## Status — 2026-11-12

Implemented on the baseline: typed `SpendPredicate` / `ManaRestriction` /
`SpendingContext` replace `ManaUse` and `SPEND_SUBTYPES` in every consumer
(cards, casting and hand/graveyard/battlefield activation pricing, pool
spending, continuation, observation data, both encoders, docs). Encoders share
`unrepresentable_state`. Witnesses: state unit tests (any subtype, resolving
costs, exclusion breadth), `cr_106_mana::restrictions_on_any_subtype_pay_exactly_their_purposes`
(Ogre cast, Scarecrow source activation vs spell, Command/fork/stale checks),
Scarecrow Guide and Supportive Parents card tests, Python refusal/parity test.
The w17 content digest moved; every w17 corpus and pin was re-recorded per
`conformance/README.md` (same command counts and outcomes).

Checks: full debug `cargo test --workspace` 1173/1173; clippy all targets and
fmt clean; Python semantic, observation, buffer, belief, Academy (one flaky
watch test passes alone) and current-world admission suites pass.

Measurement (2026-11-12, this Mac, pinned main `34980fdaa` built from an
archive in a separate target; host load 18–93 from other processes, so wall
time is not evidence). Digests of every steps/search workload match main.
- First cut (purpose stamped in `finalize_characteristics`, cast snapshot
  allocated per pricing): five alternating step pairs, user CPU 3.49 → 3.69 s
  (+5.7%). `engine_budget.py sample` under load 39–68 also failed raw wall checks.
- Retained cut: purposes stamped once per layer build in `finish_object` and
  reused while unchanged; cast purpose shares the spell's definition Arc;
  `SpendingObject` hashes a precomputed digest. Retired instructions: steps
  62.81/62.78/62.83 G (main) vs 62.72/62.81/62.81 G; Search-64 1340.6 G vs
  1326.8 G (−1.0%). No speedup over main is claimed.

Sync — 2026-11-12: `599823943` merges main `88b71387e`. Main's changes since
`34980fdaa` touch only `vector_env_bindings.rs` and `__init__.pyi`, nothing in
payment. Pre-sync checkpoint `987915ddb` narrowed `SpendingObject` to subtypes
(types dropped, matching questions.md item 3) after the checks above; the
measurement predates it.

Check (post-sync, debug): `cargo test --workspace` filtered on
mana/restrict/prism/scarecrow/supportive/powerstone/lotus — all pass, 0 failed.

Remaining: CI Engine Budget on the PR (same-runner wall time). Not ported:
policy-input rows for other restrictions, general exclusions and post-cost
variable output (questions.md). Delete list: `ManaUse`, `SPEND_SUBTYPES`,
`Mana` mask helpers — removed; no exclusive fixtures remained.

## Concept review — 2026-11-12 (headless, against `599823943`)

Accepted model, as implemented: mana carries a `ManaRestriction` (`Only` a set
of `SpendPredicate`s over any subtype, or Powerstone's `ExceptNonartifactSpells`);
every cost carries the `SpendingContext` it is paid in (`Cast` announced spell,
`Activate { zone, source }`, `Resolving`). One authority (`allows`) serves
offers, witnesses and execution; observation data names each class whole.
No change to that model.

Mismatch found — the policy refusal is per class, but ambiguity is per pool.
`unrepresentable_state` accepts any class whose subtypes are a subset of
{Lesson, Shrine, Ally spells+sources}. The columns are total restricted mana
(`lesson_mana`) plus one overlapping sum per subtype, so four numbers per color
against up to seven admitted classes. Constructed counterexample (no current
card makes it): pool A = {Lesson only: 1, Shrine-or-Ally(+sources): 1}, pool
B = {Lesson-or-Shrine: 1, Ally(+sources): 1}. Both encode total 2, Lesson 1,
Shrine 1, Ally 1. A can cast a Lesson and a Shrine; B cannot. The guard calls
both exact, so the promise "refused, never dropped" fails for them.

Fixed after the review: `REPRESENTABLE_CLASSES` in
`observation_encoder.rs` is that closed list (Lesson; Lesson-or-Shrine;
Ally spells+sources; Powerstone via v10), shared by both encoders through
`unrepresentable_state`. Pool A is refused (its Shrine-or-Ally class is not
listed); pool B is two listed classes and still encodes, now uniquely: with
the list closed, Shrine column = Lesson-or-Shrine amount, Lesson column minus
that = Lesson-only amount. Native `restrictions_on_any_subtype_pay_exactly_their_purposes`
and `tests/semantic/test_restricted_mana_input.py` cover A, B and
Lesson-or-Ally. No layout or digest change.
`d0f92723a` makes the list typed `ManaRestriction` values (set equality, so
subtype order is irrelevant), Powerstone included.
Check (`d0f92723a`, debug): `cargo test --workspace -- mana restrict
observation_encoder` all pass, 0 failed; extension rebuilt from that source;
`tests/semantic/test_restricted_mana_input.py` 7 passed (2026-11-12).

Considered and kept: `SpellSnapshot` compares by definition pointer while
`SpendingObject` compares by content. It is a second identity rule for one
concept, but it was chosen for pricing cost, digests match main, and fork
clones share the Arc; a lost transposition merge across determinized worlds
is the worst consequence. Unify only if a profile shows content equality free.

Unresolved for review (unchanged; the brief routes it there): whether to add
`FactKind::RestrictedMana` rows so arbitrary restrictions reach the policy
(questions.md item 1). CLAUDE.md's open-vocabulary rule points to rows, but
no current card needs them and the kind's shape consequence for w17
checkpoints is a model decision. A general `Except` predicate set stays
unbuilt (questions.md item 2): it would change `RestrictedManaData`'s
published schema for no card.
