# MTG-508 · Coupled mana payments, fixed-tap bonus choices, owned non-mana activations

Three rules gaps left by MTG-500 (PR #639, open; this branch is synced onto it at `5e9156a791`). Each one makes a legal payment end in `AgentError("empty non-terminal action space")`, a debug panic, or `invalid ability index`. When this lands, every offered payment action leads to a completed payment, and no action that strands the cost is offered.

The designs come from MTG-500's ten concept reviews, revised by this Task's concept review (below): `7cb3655f1:scratch/plan.md` (designs 1, 4, 7 and the fixed-tap section) and `5e9156a791^:scratch/plan.md` (deferred findings). Nobody has approved them; they were "made for review" there and remain so.

## What a player sees

| Position (committed fixture) | Today | After |
|---|---|---|
| Mountain making `{U}{R}` beside a tapped-lands Heartbeat, casting Sokka (`fixed_mana_tap_can_open_a_bonus_choice`, `710163f05`) | `TapForMana` offered; taking it panics in debug (`an automatic tap's bonuses need no choice`) | Tap, then a blue/red bonus choice, then the cast completes |
| 4 life, Citadel + Henge, damage replacement adding life gained this turn (`fixed_mana_suffix_reads_life_through_damage_replacement`, `0d7bb6743`) | Henge-first offered, then empty decision at 6 life | Only Citadel-first offered; completes at 1 life |
| Trample-counter Mountain beside a trample-reading Heartbeat (`mana_trigger_reads_keyword_changed_by_another_activation`, `a90fd7109`) | Counter-first offered, then empty decision | Only the tap is offered first |
| Angel of Vitality + Raucous Audience + two Citadels at 28 / Henge at 22 / Canyon + Island at 25, casting Solemn Simulacrum (`tarnished_citadel_and_the_great_henge_change_another_mana_source`, `34092c814`) | Legal cast pruned, or a stranding order offered | Castable exactly when some order pays; only completing first steps offered |
| Non-mana activation whose static grant ends while its mana is paid (new) | `invalid ability index`, or a different ability resolves | The announced ability goes on the stack |

Ordinary boards, TLA Limited included, see no change in offers, order, digests of non-activation states, or speed.

## Source of truth

`PaymentModel` (`managym/src/flow/mana.rs`) stays the one feasibility authority. Every consumer already reaches it through `Game::payment_model` / `announced_model` and then `plan`, `plan_after_tap`, `payable`, `payable_after_tap` or `witness` (≈45 call sites in `mana.rs`, `mana/help.rs`, `mana/interchangeable.rs`, `mana/autotapper.rs`, `mana/preview.rs`). `cached_plan` (≈2582) is the choke point for the first four. The coupled regime is a second answer path behind that same surface, never a parallel API.

## Concept review (`scratch/concept-review.html`, pre-implementation, at `4a25a7a28f`)

All five questions were answered headless with the review's recommended option, **made for review, not settled**; `ship-review` should surface them. Each is folded into the passes below:

1. Candidate steps (activate, tap-for-mana, tap-for-cost, mana choice, replacement order) and reservations reach the coupled search through a structured query, not frozen `PaymentModel` edits (1a). Passed over: intercepting only `cached_plan` (1b), because `activation_payable` (`mana.rs:1741`), `after_tap`/`payable_after_tap` (≈2652), `plan_after_tap` (≈2502), `with_tapped` (`mana/help.rs:228`) and `after_mana_choice` (≈4778) mutate the frozen source list before `cached_plan` is reached, so `0d7bb6743`, `a90fd7109`, `34092c814` and delve-from-sacrifice would still strand.
2. The search keeps one mana pool (the fork's real `mana_pool`) and memoizes on a structural payment-state key (2a). Passed over: `deterministic_hash()` (2b): it serializes the full event ledger, pending triggers and `id_gen` (`state/hash.rs:384`), so transpositions never hit and every node pays O(match) JSON + BLAKE3; a separate abstract pool breaks `Condition::UnspentManaAtLeast`.
3. `ManaSource.fixed: Option<Mana>` stays for exact taps; a fixed tap whose bonus needs a choice becomes a `ManaActivation` (3a). Passed over: `FixedOutput::{Exact, Choice}` (3b), which touches ~38 exact-tap sites and duplicates `ManaActivation`'s multi-output logic.
4. One finalized `Box<ActivatedAbilityDefinition>` is captured at announcement across `PendingChoice::ActivationX`, `ActivatedTargeting` and `Operation::Activation`; priority scans stay borrow-only (4a). Passed over: priced cost plus unfinalized definition (4b): for non-mana abilities `finalize_characteristics` (`flow/statics.rs` ≈926) only prices `mana_cost`, so the pair is redundant.
5. `MATCH_STATE_HASH_VERSION` 35 → 36 (5a). Irreversible once released; see `scratch/questions.md`.

## Pass 1 · Fixed taps can suspend on a bonus choice

Prerequisite for pass 2: the fork search steps taps natively, and a fixed tap that opens a choice panics today.

- `tap_bonuses` (`mana.rs` ≈5142) evaluates the trigger predicate as the permanent will be once tapped (CR 106.12a): `CardPredicate.tapped` (`state/src/predicate.rs:65`) counts as satisfied. Boards whose layers read tap state are pass 2's Mutator case.
- `ManaSource.fixed: Option<Mana>` keeps meaning "one deterministic automatic output"; no change to the autotapper, `direct`, `fixed_sufficient`, `fixed_total` or automatic prefixes. A fixed tap whose bonuses need a choice gets `fixed: None` and a `ManaActivation` with `kind: ManaActivationKind::{FixedTap, Activated(usize)}` and `automatic: false`, emitting `PaymentStep::Fixed(permanent)` / `Action::TapForMana`, and included by `append_mana_floats`. Removes `fixed_source_mana`'s one-type assumption (≈5100).
- `tap_for_mana_action` (≈4388) takes the caller's frame and `commit_start` like `activate_mana_ability` (≈4831) and passes them to `resolve_mana_bonuses`; `perform_automatic_tap`'s `assert!(resolved.is_none())` becomes a suspension on the bonus `ManaChoice`, never running `commit_produced_mana` while suspended. Resume is the existing `ChooseMode` arm (`decision.rs` ≈2874).
- `PaymentPreview` (`mana/preview.rs:414`): hovering a `TapForMana` whose source has no exact fixed output returns `PaymentPreviewState::NeedsChoice` instead of calling `preview_tap`.
- No new `Decision` variant, action or hash change.

Shows: the `710163f05` fixture steps `TapForMana`, chooses each colour in turn on a clone, and Sokka reaches the stack with the right receipt origins; the automatic-mode cast of the same position completes; the hover preview reports `NeedsChoice`.

## Pass 2 · Coupled payment authority and fork search

The pass that could prove the plan wrong.

**Classifier (design 4).** `BoardArithmeticProfile`, cached per source seed and merged in `LayerBoard` (`flow/statics.rs` ≈316, `flow/layers.rs` ≈1488): a closed positive whitelist; anything unrecognised is stateful. Flags as in `7cb3655f1:scratch/plan.md` design 4, `ReplacementEffect` matched exhaustively (no `_` arm), and resolved `EffectRecords` in `state.effects` carrying `Modification::{ExileIfWouldDie, ExileIfWouldLeave}` (`state/src/effect.rs:110,117`) included. Per question each activation is `PaymentCoupling::{Invariant, Sensitive, Mutator}`. Subsumes `has_triggered_mana_abilities`. A question with no Mutator answers through today's pure path, untouched.

**Query.** `PaymentModel` keeps the pure answer and gains the context the coupled path needs: applied `SourceReservation`s and counter reservations, `Help` (convoke, improvise, `delve: Option<ObjectId>`, which `help_obligations` today pre-subtracts at the root and discards, `mana/help.rs:73`), and pending upfront activations (`CoupledObligation::CompleteUpfront { permanent, activation: Box<ManaUpfrontActivation> }`, plus any in-flight `OrderReplacement` frame). Candidate-step entry points (activate, tap-for-mana, tap-for-cost, mana choice, replacement order) replace the frozen edits listed in review answer 1: uncoupled, they delegate to today's pure methods; coupled, they step the candidate natively on the fork, then search. When coupled, also bypass `PaymentCache` (its key omits game state, ≈2609), `direct`, `fixed_sufficient` (≈2694), `determined_source_counts` (`mana/interchangeable.rs:55`), `automatic_units_within` (≈4141), `equivalent_mana_choices` (≈4560), `fixed_total`, `choice_tap_witness`, `useful_payment_taps` dedup and `SourceClassCache`.

**Search (design 1).**

- State: a `Game` fork whose real `players[p].mana_pool` is the search's pool (`mana_pool_mut` invalidates `LayerBoard` only when `reads.mana` is set), plus obligations, reservations and help. Enclosing decisions are cleared only after any pending `CompleteUpfront` or in-flight `OrderReplacement` has finished natively (`complete_mana_upfront` ≈4964; `choose_replacement` + `run_payment_frame`), so a grant that ended still completes from the owned activation. With them cleared, `payment_obligations` is empty and the `ChooseMode` validation (`decision.rs` ≈2533) cannot re-enter the search.
- At each state, with no pending upfront or order, ask the pure model with Mutators removed; if it pays, done. Otherwise branch on every Mutator and every Sensitive source; Invariant sources only to fund a pending Mutator's or Sensitive source's mana cost, one representative per `PurposeProjection::sources_equivalent` class (`mana.rs:432`, with `tap_candidate_sets`), never `same_payment_profile` (≈748, ignores tap-candidate membership).
- A step is native: `Decision::ManaUpfront` from `ManaUpfrontActivation::new`, `complete_mana_upfront`, suspensions on `ManaChoice` / bonus choice (payer OR) and `OrderReplacement` (payer OR, opponent AND). Produced mana is the pool's units past `unit_count()` at the step's start.
- After a step: `fork.payment_sources(player)`, re-apply reservations (sacrifice-reserved permanents stay until CR 601.2h), recompute delve payers from the fork's graveyard, reclassify, recurse.
- Memo: `CoupledMemoKey` hashed with `FastHasher`: per permanent (present, tapped, counters, activations used), life totals, turn histories (life gained/lost, permanents sacrificed/left, nonlands left, graveyard arrivals, damage pairs), graveyard and exile card sets, pool, obligations, reservations. Never events, priority or `id_gen`.
- No cap; termination is MTG-500's finite-resource contract (`mana_parts()`).
- Witness recorded as `PaymentStep`s so offers keep source order and action identity.
- `mana_suffix_life_delta`'s probe (≈3352) stays as the verified `LifeDelta` of Invariant and Sensitive steps.

`PaymentCache` gains test-visible counts of coupled searches, fork states and memo hits. Tests assert on counts, never time.

Shows: `0d7bb6743`, `a90fd7109` and `34092c814` flip from asserting the failure to stepping the completion and asserting the stranding first step is absent; two Citadels in either order hit the memo. Citadel/Room/Canyon + plain Heartbeat, Badgermole Cub or Air Nomad Legacy + Treasures, and Raucous Audience beside plain lands report zero coupled searches.

Stop rule, as in MTG-500: a counterexample that strands on a board the classifier calls uncoupled is committed as a fixture and widens "stateful"; one that strands inside the search is a search bug fixed there.

## Pass 3 · Payer replacement order, remaining boards, bounds

- **Design 7.** While the chooser has an enclosing payment, `Decision::OrderReplacement` offers (`decision.rs` ≈1949) and admits (≈2692) only orders the coupled authority completes, through the replacement-order entry point. An opponent's choice is never filtered. At priority all orders remain. One surviving order is taken by trivial-decision skipping.
- Acceptance boards from `7cb3655f1:scratch/plan.md` pass 4b: Doubling Season → Caradora while casting `{G}{G}` (only Caradora → Season offered); delve beside a non-token `{T}, Sacrifice: Add {G}` (and the same with an `ExileIfWouldDie` record, where delve gains nothing); summoning-sick creature with remove-haste-counter mana; a `tap_cost` predicate and an `ActivatedAbilitiesCostLess` predicate each beside a non-mana write; PreventAll reader before Citadel at 3 life; remove-a-+1/+1-counter mana beside Raucous Audience; Treasure beside a `SacrificedThisTurnAtLeast` output; White Lotus Tile / Avatar Kyoshi with a sacrifice source; a suspended `ManaUpfront` whose grant ended.
- `{15}` with 10 Forests, Angel, Audience and two Citadels: Forests classify Invariant, the answer is "unpayable", and the test pins the fork-state count (expected low hundreds; an estimate, not a measurement).

## Pass 4 · Owned non-mana activations

- Capture one finalized `Box<ActivatedAbilityDefinition>` at announcement (`activate_ability_action` / `begin_activation_targets`, `flow/play.rs` ≈1696, ≈2030) on `PendingChoice::ActivationX` (≈1760, ≈1862), `ActivatedTargeting` (`flow/game.rs:404`) and `Operation::Activation` (`flow/payment.rs:68`).
- Every later read uses the capture instead of `characteristics(..).activated_abilities.get(ability_index)`: ActivationX resume, `activation_target_requirement` and callers (`play.rs` ≈2103–2318), `cost_components` / `begin_payment` (`payment.rs` ≈811, ≈899), completion (≈1402 → `prepare_activated_ability`, `play.rs` ≈2358, taking the ability), `owed_reservation` / `announced_model` (`mana.rs` ≈3155, ≈3504), `action.rs` ≈1090/≈1259, `flow/interchangeable.rs` ≈943/≈1050, `agent/continuation.rs` ≈1188, `agent/effect_execution.rs` ≈656. `ability_index` stays for the stack object, use-limit record and observation frame.
- `prepare_card_activated_ability` (`play.rs` ≈2441) keeps an explicit check that the card is still in `ability.zone` ("activation source left its zone").
- Hypothetical feasibility checks in priority scans (`action.rs` ≈820/≈931, `mana.rs` ≈4483, `mana/preview.rs` ≈318) borrow `&ActivatedAbilityDefinition`; no clone per priority tick.
- `Operation::Activation` is hashed both through `PendingChoice` (`agent/env/snapshot.rs` ≈342) and inside `GameState.suspended_decision` (`SpendCompletion::Announced`, `FrameCompletion::{DiscardPayment, Paid}`). `MATCH_STATE_HASH_VERSION` 35 → 36.
- Regression: an activation granted by a static that depends on a mana source's state; paying its mana ends the grant; the announced ability reaches the stack with its targets; with a second ability on the source the right one resolves; same for an X activation. Checked live and through cold snapshot restore of a recorded object-bound Command.

Then: verify the drafted docs (`managym/README.md`, `docs/engine.md`) against the final implementation, `uv run --extra dev python scripts/refresh_academy_fixtures.py`, conformance fixtures, Engine Budget (`scripts/engine_budget.py record --reason` only if the digest moves for the version bump alone).

## Docs

Drafted in place before implementation (read these when building and reviewing):

- `managym/README.md` — owned non-mana activations, coupled `PaymentModel` fork search and `OrderReplacement` filtering, fixed-tap bonus choices in manual/automatic payment, and `needs_choice` on hovered fixed taps that open a bonus choice.
- `docs/engine.md` — `MATCH_STATE_HASH_VERSION` v35 → v36 and snapshot restore compatibility.

## Next

Nothing is built. Pass 1 is next: it delivers the fixed-tap bonus choice and teaches whether `ManaActivationKind::FixedTap` fits the existing `ManaActivation` search without touching exact-tap sites. Pass 2 is where the plan can still be proved wrong.

## Acceptance check

```sh
CARGO_BUILD_JOBS=2 cargo nextest run --manifest-path managym/Cargo.toml \
  -p managym -p managym-cards --test-threads 2 \
  -E 'test(manual_payment) | test(tarnished_citadel) | test(coupled) | test(owned_activation)'
```

Debug profile (CI's). Expected: all pass, including the five flipped fixtures, the fork-free TLA boards, the pinned `{15}` state count, and the grant-ends-mid-payment regression. Then the debug nextest for both touched crates.

## What must keep holding

- Native action identities, source order, the chooser, individual Heartbeat bonus decisions, spend restrictions, receipts, replay and action formats.
- `PaymentScan`'s debug rebuild check and `PaymentCache`'s debug recompute: probes and searches run on forks and leave `&self` untouched.
- Search clones append no decision-memory records and carry no recording sinks.
- No test asserts on time or builds a large board; the 14-permanent `{15}` board is the largest.
- Engine Budget: ordinary boards do no new work beyond reading one cached profile.

## Absent and error states

- Unpayable coupled question: the action is not offered (cast, activation, tap, mode, order). Never an empty decision.
- A fork step that errors or ends the game mid-payment: that branch is infeasible, except that reaching zero or negative life is allowed where the pure model allows it (signed `LifeBudget`).
- An opponent's `OrderReplacement` inside the search: the payment is feasible only if every order completes.
- Captured activation whose source left: unchanged from today (`activation source left its zone` for cards; a permanent's LKI path for the stack object).

## Exclusions

- No new card, mechanic, observation column or learning token. No Python or Academy surface change.
- No widening of the mana-ability contract or its suffix whitelist.
- No one-shot prevention shields (managym has none).
- The original Linux report's exact board stays unknown.

## What goes

- `fixed_source_mana`'s "no current fixed source produces more than one type" assumption and the `an automatic tap's bonuses need no choice` assert.
- The frozen candidate-step edits as coupled answers (`activation_payable`, `after_tap`, `plan_after_tap`, `with_tapped`, `after_mana_choice`); they survive only as the uncoupled path.
- `LayerBoard.has_triggered_mana_abilities` (`statics.rs` ≈335, `layers.rs` ≈460), replaced by the profile.
- The failure-asserting halves of the four fixtures (`unwrap_err` on `empty non-terminal action space`, the "frozen estimate" comments, the offline resolver walk in the fixed-tap test).
- `activation_ability(source, index)` reads after announcement (ActivationX included), and `invalid ability index` at `play.rs` ≈2382.
- Survives: `mana_suffix_life_delta`, `PaymentCache`, `SearchEquivalence`, `direct()`, every MTG-500 fixture's passing half.

## Stack

PR #639 lands first. This branch already contains it; after it merges, `lf sync main` rebases the stack. No security or privacy surface is touched; no security review needed.
