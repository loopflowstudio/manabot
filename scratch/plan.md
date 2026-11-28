# MTG-508 · Coupled mana payments, fixed-tap bonus choices, owned non-mana activations

Three rules gaps left by MTG-500 (PR #639, open; this branch is synced onto it at `5e9156a791`). Each one makes a legal payment end in `AgentError("empty non-terminal action space")`, a debug panic, or `invalid ability index`. When this lands, every offered payment action leads to a completed payment, and no action that strands the cost is offered.

The designs were settled across MTG-500's ten concept reviews and are carried here unchanged: `7cb3655f1:scratch/plan.md` (designs 1, 4, 7 and the fixed-tap section) and `5e9156a791^:scratch/plan.md` (deferred findings). Nobody has approved them; they were "made for review" there and remain so.

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

## Pass 1 · Fixed taps can suspend on a bonus choice

Prerequisite for pass 2: the fork search steps taps natively, and a fixed tap that opens a choice panics today.

- `tap_bonuses` (`mana.rs` ≈5142) evaluates the trigger predicate as the permanent will be once tapped (CR 106.12a). `CardPredicate.tapped` (`state/src/predicate.rs:65`) is the only tap-state field, so the planner matches with the tapped requirement satisfied; boards whose layers read tap state are pass 2's Mutator case.
- `ManaSource.fixed: Option<FixedOutput>` with `FixedOutput::{Exact(Mana), Choice(ManaChoices)}`. `fixed_source_mana` (≈5100) returns `Choice` where it returns `None` today (`outputs.len() > 1`).
- `tap_for_mana_action` (≈4388) takes the caller's frame the way `activate_mana_ability` does (≈4831), and `perform_automatic_tap`'s `assert!(resolved.is_none())` becomes a suspension on the `ManaChoice` that `resolve_mana_bonuses` already builds (`ManaProduction::Bonus`, `commit_start` included). Resume is the existing `ChooseMode` arm in `decision.rs` ≈2874.
- Only `Exact` sources enter automatic batches: `produce_mana`, `pay_mana_upfront`'s `PaymentStep::Fixed` prefix, `preview_tap` (`mana/preview.rs:414`), `fixed_total`, `direct()`.
- No new `Decision` variant, action or hash change.

Shows: the `710163f05` fixture steps `TapForMana`, chooses each colour in turn on a clone, and Sokka reaches the stack with the right receipt origins; the automatic-mode cast of the same position completes.

## Pass 2 · Board whitelist and the coupled fork search

The pass that could prove the plan wrong.

**Classifier (design 4).** `BoardArithmeticProfile`, cached per source seed and merged in `LayerBoard` (`flow/statics.rs` ≈316, `flow/layers.rs` ≈1488): a closed positive whitelist. Anything it does not recognise is stateful. Flags as listed in `7cb3655f1:scratch/plan.md` design 4, with `ReplacementEffect` matched exhaustively (no `_` arm). Per payment question each activation gets `PaymentCoupling::{Invariant, Sensitive, Mutator}`. `has_triggered_mana_abilities` is subsumed. A board with no Mutator answers through today's pure path, untouched.

**Search (design 1).** Reached from `cached_plan` only when the question has a Mutator. `PaymentModel` gains the `&Game` and payer it needs for this.

- State: a `Game` fork with `pending_choice` and `suspended_decision` cleared, plus the question's abstract pool, obligations and reservations. Verified: on such a fork `payment_obligations` is empty, so the `ChooseMode` validation in `decision.rs` ≈2533 (`mana_choice_completes_payment`) is vacuous and the search cannot re-enter itself.
- At each state, first ask the pure model with Mutator abilities removed. If it pays, done (handoff). Otherwise branch on each Mutator, and on Sensitive/Invariant sources only as far as funding a Mutator's own mana cost needs (one representative per `same_payment_profile` class).
- A Mutator step is native: build `Decision::ManaUpfront` from `ManaUpfrontActivation::new` and call `complete_mana_upfront` (≈4964), which already commits tap, life, counters, sacrifice and suffix and suspends on `ManaChoice` or `OrderReplacement`. Branch on each `ManaChoice` mode and each bonus choice (payer OR). `OrderReplacement`: payer OR, opponent AND. Mana produced is read as the fork pool's units past `unit_count()` at the step's start, the boundary `commit_produced_mana` uses.
- After a step: `fork.payment_sources(player)`, re-apply the caller's `SourceReservation`s (a sacrifice-reserved permanent stays until CR 601.2h), reclassify, recurse. A Mutator whose mana cost is still being funded stays a pending completion obligation; no pure handoff while one is pending.
- Memo key: `fork.state.deterministic_hash()` + pool + obligations + reservations. BLAKE3 over the serialized state is slow, and it runs only on coupled nodes.
- No cap. Termination is MTG-500's finite-resource contract (`mana_parts()`): every activation lowers the payer's unreserved permanents, untapped sources, use limits or counters.
- The witness path is recorded as `PaymentStep`s so offers keep source order and action identity.
- Shortcuts that assume a tap never changes another source defer to the authority when the question has a Mutator: `automatic_payability`'s `fixed_total`, `choice_tap_witness`, `useful_payment_taps` dedup, `SourceClassCache`, `direct()`.
- `mana_suffix_life_delta`'s probe (≈3352) stays as the verified `LifeDelta` of Invariant and Sensitive steps.

`PaymentCache` gains a test-visible count of coupled searches and fork states. Tests assert on those counts, never on elapsed time.

Shows: `0d7bb6743`, `a90fd7109` and `34092c814` flip from asserting the failure to stepping the completion and asserting the stranding first step is absent from the action space. Citadel/Room/Canyon + plain Heartbeat, Badgermole Cub or Air Nomad Legacy + Treasures, and Raucous Audience beside plain lands report zero coupled searches.

Stop rule, as in MTG-500: a new counterexample that strands on a board the classifier calls uncoupled is committed as a fixture and widens "stateful"; one that strands inside the search is a search bug fixed there.

## Pass 3 · Payer replacement order, remaining boards, bounds

- **Design 7.** While the chooser has an enclosing payment, `Decision::OrderReplacement` offers (`decision.rs` ≈1949) and admits (≈2692) only orders the authority completes. An opponent's choice is never filtered. At priority all orders remain. If one order survives, native trivial-decision skipping takes it.
- Acceptance boards from `7cb3655f1:scratch/plan.md` pass 4b: Doubling Season → Caradora while casting `{G}{G}` (only Caradora → Season offered); delve beside a non-token `{T}, Sacrifice: Add {G}`; summoning-sick creature with remove-haste-counter mana; a `tap_cost` predicate and an `ActivatedAbilitiesCostLess` predicate each beside a non-mana write; PreventAll reader before Citadel at 3 life; remove-a-+1/+1-counter mana beside Raucous Audience; Treasure beside a `SacrificedThisTurnAtLeast` output; White Lotus Tile / Avatar Kyoshi with a sacrifice source; a suspended `ManaUpfront` whose grant ended.
- `{15}` with 10 Forests, Angel, Audience and two Citadels: Forests classify Invariant, the answer is "unpayable", and the search visits a small fixed number of fork states (the test pins the count; expected in the low hundreds from two five-colour Citadels and one Audience).

## Pass 4 · Owned non-mana activations

- `ActivatedTargeting` (`flow/game.rs:404`) and `Operation::Activation` (`flow/payment.rs:68`) carry a boxed owned activation: the priced cost and the unfinalized `ActivatedAbilityDefinition` (already `Serialize`), captured once in `activate_ability_action` / `begin_activation_targets` (`flow/play.rs` ≈1696, ≈2030), the way `ManaUpfrontActivation::new` does for mana abilities.
- Every later read uses the capture instead of `characteristics(..).activated_abilities.get(ability_index)`: `activation_target_requirement` and its callers (`play.rs` ≈2103–2318), `cost_components` and `begin_payment` (`payment.rs` ≈811, ≈899), completion (`payment.rs` ≈1402 → `prepare_activated_ability`, `play.rs` ≈2358, which takes the ability instead of re-indexing), `owed_reservation` / `announced_model` (`mana.rs` ≈3155, ≈3504), `action.rs` ≈1090/≈1259, and the walkers in `flow/interchangeable.rs` ≈943/≈1050, `agent/continuation.rs` ≈1188, `agent/effect_execution.rs` ≈656. `ability_index` stays for the stack object, use-limit record and observation frame.
- Hypothetical operations built for feasibility (`action.rs` ≈820/≈931, `mana.rs` ≈4483, `mana/preview.rs` ≈318) capture the definition as it is when asked.
- `PendingChoice` is serialized into `state_digest` (`agent/env/snapshot.rs` ≈342), so digests taken mid-activation change. `MATCH_STATE_HASH_VERSION` goes 35 → 36 (chosen for review; see `scratch/questions.md`).
- Regression: an activation granted by a static that depends on a mana source's state; paying its mana ends the grant; the announced ability still reaches the stack with its targets, and with a second ability on the source the right one resolves. Checked live and through cold snapshot restore of a recorded object-bound Command.

Then: `managym/README.md` mana-payment section (coupled regime, fixed-tap choice, owned activations, v36 restore note beside the v35 one), `uv run --extra dev python scripts/refresh_academy_fixtures.py`, the conformance fixtures, and Engine Budget (`scripts/engine_budget.py record --reason` only if the digest moves for the version bump alone).

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
- `LayerBoard.has_triggered_mana_abilities` (`statics.rs` ≈335, `layers.rs` ≈460), replaced by the profile.
- The failure-asserting halves of the four fixtures (`unwrap_err` on `empty non-terminal action space`, the "frozen estimate" comments, the offline resolver walk in the fixed-tap test).
- `activation_ability(source, index)` reads after announcement, and `invalid ability index` at `play.rs` ≈2382.
- Survives: `mana_suffix_life_delta`, `PaymentCache`, `SearchEquivalence`, `direct()`, every MTG-500 fixture's passing half.

## Stack

PR #639 lands first. This branch already contains it; after it merges, `lf sync main` rebases the stack. No security or privacy surface is touched; no security review needed.
