# MTG-500 · Finish manual mana payment

A manual payment must return the payer’s next legal choice, finish the cast, or reach the game’s normal outcome. The reported stall held every game at the native step barrier.

Pass 1 (`1c521e300`) propagated empty, forced and automatic decision errors and applied the zero-life cost rule. The Room at −1 offers blue and red, isolated stepping poisons only the malformed slot under both skip settings, and all 23 focused debug tests passed.

Pass 2 (`a9359c03b`) carries signed life through mana, checks trailing fees and records source coupling. All 78 focused debug checks passed, including the four Heartbeat positions, the announced and Ward fees, and the read-only fork. The 512-game debug step benchmark was stopped at 4m41s before it emitted a digest. The release Engine Budget comparison remains in pass 5.

The first pass 2 attempt found the counterexample below (`34092c814`, test `tarnished_citadel_and_the_great_henge_change_another_mana_source` in `managym/cards/tests/cards/ody.rs`) and changed no production code. The concept review (`24f1ec2f3`, `scratch/concept-review.html`) replanned around it.

Pass 3 (`abfabece991`) owns the announced mana activation and its priced fee, reads dynamic output at completion, reserves paid taps from the owed cost, classifies Treasure against battlefield reads, and bumps the match-state hash to v35. Twelve focused debug checks passed, including live and snapshot-restored Ogre, Henge repricing, and conditional Treasure.

Pass 4's finite-resource contract (`ab008535c`) shares one mana-ability predicate, routes board-growing and counter-swapping abilities through the stack, and reserves the pending activation's counters and source presence. The 48 focused debug checks passed. Catalog parity held across printed, token and granted definitions, with suffixes on Tarnished Citadel and The Great Henge only.

Pass 4a (`494fb4c5e`) retains counter and sacrifice costs before computing mana output and walks the saved frame for obligations and reservations. All 56 focused debug checks passed across the mana and retained-batch fixtures, including live and restored replacement orders, counter and sacrifice outputs, and nested cast and resolution payments.

Pass 4 attempt 1 stopped at a counterexample (`0d7bb6743`, `fixed_mana_suffix_reads_life_through_damage_replacement` in `managym/src/flow/mana/tests.rs`) and changed no production code. At 4 life, Henge (+2) and Citadel (3 damage plus life gained this turn, through a constructed damage replacement) owe `{G}{G}{R}` then 2 life. Root probes certify both independent; Henge-first reaches an empty decision at 6 life (Citadel now deals 5). Citadel-first pays with 1 life left. The fourth concept review (`8182d5c876`, `scratch/concept-review.html`) widened this to a missing-read-channel finding.

Pass 4 attempt 2 reproduced `mana_trigger_reads_keyword_changed_by_another_activation` in debug (`a90fd7109`, `managym/src/flow/mana/tests.rs`). A constructed Heartbeat matches lands with trample. The Mountain can either tap or remove its trample counter to add `{G}`. While paying `{G}{R}{R}`, the counter-first offer reaches an empty decision because the later tap makes only `{R}`. Tap-first commits both red units and completes. Design 4's read graph would still certify both steps independent, so the stop rule sent pass 4 back to review.

The fifth concept review (`fc7f2f70e7`, `scratch/concept-review.html`) found four more channels the read graph misses (`tap_cost.predicate` `mana.rs` ≈3381; `ActivatedAbilitiesCostLess.predicate` `statics.rs` ≈990; delve graveyard growth from a non-token sacrifice `help.rs` ≈51; a summoning-sick creature removing its own haste counter `mana.rs` ≈3349) and concluded the blacklist premise is wrong. Design 4 below now holds its whitelist. Both Heartbeat of Spring (`chk.rs`) and Badgermole Cub (`tla.rs`) use a predicate with only `card_type` set, so the TLA acceptance boards stay on the whitelist.

## Pass 4 attempt 3 · repeatable mana effects

Attempt 3 (`de9787c04`, `repeatable_mana_suffix_grows_the_board_at_a_full_pool` in `managym/src/flow/mana/tests.rs`) shows that a constructed `{T}: Add {G}, create a Clue, untap this` is admitted as a mana ability. Three activations leave the same saturated pool and a ready source, add a Clue each time and change the match-state hash each time. No fork search over such effects terminates without a cap. `is_mana_suffix_effect` (`managym/state/src/card.rs:683`) accepts any untargeted effect except mana output, `Modal` and `If`. That contradicts `managym/README.md:669` ("Every registered mana ability must consume a finite resource"). The sixth review (`6f7b914a4`, `scratch/concept-review.html`) closes that contract as design 5 below.

Pass 4 attempt 4 (`da29dbbdc`, `mana_counter_cost_can_require_replacement_order` in `managym/src/flow/mana/tests.rs`) found an immediate-event panic in an activation admitted by design 5. Gray Ogre has a constructed `{T}, Put a +1/+1 counter on this: Add {G}`; Caradora adds one counter and a test-adjusted Doubling Season doubles counter costs. Stepping the offered action panics with `event requiring a choice needs a retained caller`. The committed fixture checks that the `CounterChange` batch asks the payer for `OrderReplacement` and that the two orders place three and four counters.

The seventh review (`4d5644302`, `scratch/concept-review.html`) traced the panic to two cost-execution models: non-mana activations retain counter and sacrifice costs in an `EffectFrame.batch` with `FrameCompletion::Activated` (`payment.rs` ≈1506), while `complete_mana_upfront` (`mana.rs` ≈5133) sends counters to `apply_immediate_event` (`proposed_event.rs` ≈1531), evaluates `upfront_mana_outputs` before `sacrifice_this` runs, and parks sacrifice and suffix behind `FrameCompletion::Decision(ManaChoice)`. It found three more defects on HEAD: dynamic sacrifice outputs count the source still on the battlefield; `payment_model::reserve` (≈3280) and `payment_obligations_owing` (≈3667) read only `suspended.decision`, so a suspended `OrderReplacement` hides every enclosing reservation and obligation; and live `OrderReplacement` offers every order during payment, so once the fork search certifies one order, the payer can still pick a stranding one (review case 2: Season → Caradora gives 3 counters, a 5/5 and only `{G}`).

## Why a frozen source list fails

Angel of Vitality is 4/4 at 25 life or more; Raucous Audience makes `{G}{G}` while its controller has a power-4 creature, else `{G}`. Casting Solemn Simulacrum (`{4}`):

- **28 life, two Tarnished Citadels.** Each Citadel alone leaves 25 life; both leave 22 and Audience drops to `{G}` (3 mana). Only Citadel → Audience → Citadel pays. Probing each source once from the root sees no change.
- **22 life, The Great Henge.** Henge gains 3 with Angel's replacement and makes `{G}{G}`; Audience then makes `{G}{G}`. The root total (2 + 1) prunes the legal cast before Henge is tried.
- **25 life, Sunbaked Canyon + Island.** No `mana_suffix()` anywhere; Canyon's ordinary `life_cost: 1` crosses the threshold. Audience must go first.
- **Sacrifice source + White Lotus Tile / Avatar Kyoshi.** Sacrificing changes a dynamic output.

Coupling is board-level: what a payment step writes (life, sacrifice, taps, counters, pool, effects) intersecting what the layer board reads (`LayerBoard.reads: ReadSet`, `dynamic_activations`, `has_triggered_mana_abilities`, `has_activated_cost_reduction` in `flow/layers.rs`, `flow/statics.rs`). The pure witness models the source list at the question’s root. Pass 4 rebuilds it after a coupled transition.

Colored convoke lives in `while_paying` (`flow/mana/help.rs`), wrapped by three outer callers and skipped by nested surfaces (`automatic_upfront_path`, `mana_choice_actions`/`mana_choice_completes_payment`, `mana_spend_choices`, `mana_tap_cost_actions`, `automatic_payment_action`); `automatic_payability`, `choice_tap_witness` and `useful_payment_taps` assume taps never change another source.

## Design · made for review

The run is headless; these follow the concept review's recommendations (1a, 2a, 3a, 4a) and are reversible internal engine work. Nobody has approved them. Earlier design 2 (probe each `mana_suffix()` once at the root, `LifeDelta`/`BoardChanging` on a frozen list) is refuted by the boards above and dropped.

1. **One feasibility authority with two regimes.** `PaymentScan` computes a `PaymentCoupling` once from source write flags (`life_cost > 0`, `mana_suffix()`, `sacrifice_source`, tap, counters, pool) against the layer board's reads, and from a suffix probe on an isolated fork (suspends, or mutates anything but life → coupled).
   - *Uncoupled:* the pure `PaymentModel` with a constant `LifeDelta(i32)` for fixed payer-local suffixes, full bounds, `PaymentCache`, `SearchEquivalence`, `direct()`. Covers ordinary TLA boards, including Citadel/Canyon/Room + Heartbeat.
   - *Coupled:* exact transitions on a `Game` fork. At each state, first ask the pure model to finish without coupled mutators; otherwise branch only on coupled mutators and affected sources (one representative per equivalence class of invariant sources when an in-flight activation's upfront cost needs funding), re-apply the caller's `SourceReservation`s on the fork (sacrifice-reserved permanents stay on the battlefield until CR 601.2h), rebuild sources, and hand off to the pure model once no coupled mutator remains. Memoize by canonical payment state. Static root bounds do not prune while a coupled mutator remains.
   - Shortcuts (`automatic_payability`'s fixed total, `choice_tap_witness`, `useful_payment_taps` dedup, `SourceClassCache`) defer to the authority on a coupled slice.
   - Passed over: re-deriving sources whenever life changes (misses shields, sacrifice, taps, counters); forking whenever any suffix/life/sacrifice source exists (Treasures and painlands are everywhere in TLA; kills the caches); plain forked DFS without pure handoff (10 Forests + coupled sources checking `{15}` explodes).
2. **Exact life.** Signed `LifeBudget { life: i32, cant_change: bool }`; activation `life_cost` checked against the running budget; post-mana life (announced `payment.life()`, cast `life_alternative`, resolution tail life) is a trailing `Obligation::PayLife(u32)` at completion (CR 601.2h).
3. **Colored convoke in the model.** `ManaSource.convoke_colors: ManaTypeSet`; the witness reduces an owed colored pip by tapping such a source; delete `while_paying`/`convoke_search`. Fallback if this exceeds a pass: one wrapper routing all eight surfaces through `while_paying`. Never defer it: with pass 1, an unfixed Guardian play is an `AgentError`.

4. **Closed positive whitelist, three roles** (fifth review 1a–4a, made for review; replaces design 4's read/write graph, passed over as 1b because every unwalked channel fails open).
   - `BoardArithmeticProfile` cached per `SourceSeed` and merged in `LayerBoard` over the battlefield and `functioning_command_objects(None)`: `has_conditional_or_count_statics` (condition, `LayerValue::Count`, battlefield/life/mana/tapped reads, `ActivatedAbilitiesCostLess`), `stateful_static_writes_non_pt`, `has_stateful_mana_triggers` (any triggered mana ability whose predicate is not `CardPredicate { card_type, ..default() }`; implement should also require a non-reading `who`/`output`), `has_damage_replacements` / `has_life_gain_replacements` / `has_counter_or_zone_replacements` from an exhaustive `ReplacementEffect` match with no `_` arm, `has_stateful_mana_abilities` (dynamic output, condition, cost reduction, `reduce_by_source_power`, `tap_cost`), `reads_mana_or_tapped`. Anything unrecognized is stateful.
   - Intrinsic `ActivationFacts` per `ManaSource` inside `PaymentScan` (suffix kind, verified `LifeDelta`, writes non-mana, writes counters, removes board rules, reads state); `PaymentCoupling { Invariant(LifeDelta), Sensitive(LifeDelta), Mutator }` is derived once per payment question (3a) from live sources, reservations, `help.delve` and suspended `ManaUpfrontActivation`, and `Obligation::Output` carries the role instead of `.life_delta()` (`payment_obligations_owing`, `mana.rs` ≈3516).
   - `board_sensitive` = stateful statics | stateful triggers | stateful mana | a replacement channel matched by a remaining step kind (2a) | delve. Mutator: any non-mana write while `board_sensitive`; always counter adds/removes, sacrificing a rule-bearing permanent, an unverified suffix; any tap while mana or tapped state is read. Sensitive: a state-reading step, a step sharing a permanent with a Mutator (or summoning-sick beside one), or any tap when a conditional static writes outside P/T (4a). Invariant otherwise.
   - Deletes `mana_activation_reads`, `payment_writes_couple` (`statics.rs` ≈366) and `has_triggered_mana_abilities`; `LayerBoard.reads` returns to cache invalidation only. No hash, replay or observation change (v35 stays).
   - Passed over: a single flat board boolean (2b, forks every Treasure beside Air Nomad Legacy or Badgermole); pre-baking coupling in `payment_sources()` (3b, blind to priority-time delve and reservations); every source Sensitive on any conditional static (4b, ~10× fork states on the 10-Forest board).

5. **Closed monotone mana-ability contract** (sixth review 1a–4a, made for review; reversible, with no change to the hash, tokens or replay).
   - One predicate on `ActivatedAbilityDefinition` (`mana_parts()`, `state/src/card.rs`) is used by `is_mana_ability`, `mana_choices`, `has_dynamic_mana_output`, `mana_suffix` and fixed entry mana. It requires three things. First, the costs are exactly those `ManaUpfrontActivation` pays: battlefield zone, and no loyalty, generic sacrifice, waterbend, exile/return/discard source, grantor cost, tap-power or station. Second, the ability consumes a finite resource: `sacrifice_source`, `tap_source`, a use limit, `tap_cost` > 0, or `remove_counters` > 0 with no `add_counters`. Third, every suffix is one of `GainLife`, `DrainLife`, or `LoseLife`/`DealDamageToPlayer` without a target (2a). Any other ability is an ordinary stack ability.
   - `Decision::ManaUpfront` reserves pending `remove_counters` the same way it reserves `tap_source` and `sacrifice_source` (`payment_model::reserve`, `mana.rs` ≈3209). Without that reservation, `{1}, Remove a counter: Add {G}{G}` on a permanent with one counter can nest without bound or spend the counter twice (3a).
   - The catalog parity test (`registered_mana_activations_consume_finite_payment_resources`, `tests.rs` ≈2439) widens to printed abilities, tokens, `GrantActivated` grants, dynamic outputs and suffixes. Any syntactic CR 605.1a mana ability the contract rejects fails CI.
   - Termination: each activation strictly lowers the payer's unreserved (permanents, untapped, use limits, counters) lexicographically. Choice and spend steps lower the owed pips or the continuation depth, and delve shrinks the graveyard. The fork search therefore needs no cap.
   - Passed over: runtime `AgentError` for non-monotone abilities (1b, two meanings of "mana ability" and fallible read paths); a capped fork search (1c, violates the no-truncation rule and still undecidable); a wider suffix whitelist (2b, no catalog card uses it); excluding bare `remove_counters` (3b, breaks `a90fd7109`).
   - The catalog audit in `ab008535c` confirmed the two suffixes and found no classification changes.

6. **Phase-accurate mana activation continuation** (seventh review 1a, 3a, 4a, made for review; reversible, adds a `FrameCompletion` variant under the branch's unreleased v35).
   - `complete_mana_upfront` pays taps and life immediately and captures `source`, `tapped` and the source's `chosen` entry choice as LKI. With counter or sacrifice costs it runs one cost `EffectFrame`: `CounterChange` (remove, then add) in `batch`, `sacrifice_this` in `queue` (3a, so departure LKI sees cost counters), completing with `FrameCompletion::ManaOutput(Box<PendingManaOutput { player, source, tapped, chosen, output, suffix, caller, resume }>)`. Without them it takes the current path with no allocation.
   - `complete_mana_output` evaluates `upfront_mana_outputs` after all costs and before the suffix, reading the captured `chosen` instead of `permanents[permanent].chosen` (CR 601.2h, 605.3b, 608.2h); the suffix then runs in its child frame before `ManaChoice`.
   - `payment_model::reserve`, `payment_obligations_owing`, `SuspendedResolution::continuation`, `FrameCompletion::then`, `run_payment_frame` and the exhaustive consumers (`interchangeable.rs`, `selection_sampling.rs`, `agent/effect_execution.rs`) walk `frame.completion` through `ManaOutput`'s `caller` and `resume`.
   - Passed over: pre-baking `ManaChoice` (1b, stale output across counter replacements and sacrifice); a post-cost stage on `Decision::ManaUpfront` (1c, a player-facing variant used as an internal transition); one simultaneous batch (3b, LKI misses cost counters); hash v36 (4b, no released v35).
7. **Payer `OrderReplacement` matches the search** (seventh review 2a, made for review; promoted from deferred findings). While `has_enclosing_payment(chooser)` holds, `Decision::OrderReplacement` (`decision.rs` ≈1918 offers, ≈2661 execution) offers and admits only orders the coupled authority completes; the fork search treats the payer's orders as OR and an opponent's as AND, and never filters an opponent's choice. Each step pushes one `ReplacementId` onto `pending.applied`, so with design 5 the search stays uncapped. Passed over: keeping it deferred (2b), which leaves an `AgentError` on a certified activation.

Preserve native action identities, source order, the chooser, individual Heartbeat bonus decisions, spend restrictions and receipts. Keep existing replay and action formats and version the canonical state hash when it changes. The `PaymentScan` debug rebuild check must keep holding: probes and coupled search run on forks and leave `&self` untouched.

## Remaining passes

4b. **Whitelist classifier, coupled regime and payer replacement filter** (designs 4, 1 and 7; `flow/layers.rs`, `flow/statics.rs`, `flow/mana.rs`, `flow/mana/help.rs`, `flow/decision.rs`). Counter-changing transitions stay in the fork, carrying live counter reservations into each rebuild. Live checks:
   - `0d7bb6743`: only Citadel-first offered; at 1 life the damage channel deactivates and Henge hands off to the pure model.
   - `a90fd7109`: only the Mountain tap is offered first; it commits both red units before the counter is removed for green.
   - Review case 2 while casting `{G}{G}`: only Caradora → Season is offered; at priority both orders remain.
   - Fifth review boards: delve spell beside a non-token `{T}, Sacrifice: Add {G}`; summoning-sick creature with remove-haste-counter mana and `{T}` mana (only tap-first); a `tap_cost` predicate and an `ActivatedAbilitiesCostLess` predicate each beside a non-mana write.
   - Reader before mutator: at 3 life a PreventAll permanent with `{T}, Sacrifice: Add {G}` beside Citadel, owing `{G}{R}` + 2 life; Citadel-first is offered.
   - Remove-a-+1/+1-counter mana beside Raucous Audience paying 3: only Audience-first. Treasure beside a `SacrificedThisTurnAtLeast`/graveyard-conditional output.
   - Suspended `ManaUpfront` whose grant ended, and one with a coupled suffix: no pure handoff while pending.
   - The frozen-list boards above, sacrifice with White Lotus Tile / Kyoshi, and a quick debug rejection of `{15}` with 10 Forests, Angel, Audience and two Citadels (Forests stay Invariant).
   - Ordinary TLA boards (Citadel/Room/Canyon + plain Heartbeat, Badgermole or Air Nomad Legacy + Treasures, Fated Firepower without Citadel) classify all-Invariant: no forks.
   - If a board outside the whitelist still strands, the fault is in the fork search, not a missing channel; fix it there.
5. **Colored convoke, collection check and acceptance.** Convoke in the model; the Guardian of Vitu-Ghazi board completes its cast. Restore a pre-activation snapshot into one slot of a two-slot native vector with skipping on; both slots return and match scalar play. Acceptance demo, debug nextest for touched crates, engine budget comparison, and the `managym/README.md` mana-payment section for the built design.

Pass 4 has stopped at a counterexample four times. If 4b stops again, split the coupled regime (4b) into its own Task and ship passes 1–4a with pass 5 (see `scratch/questions.md`).

## Reproduced on a TLA Limited board

Cast **Sokka, Bold Boomeranger** for {U}{R} with an empty pool, two untapped lands and **Heartbeat of Spring**. Activate **Tarnished Citadel** for colored mana first. Its ability deals 3 damage to its controller. Each land that produces mana triggers Heartbeat once. Citadel, Canyon and Heartbeat are in the TLE bonus sheet used in TLA Limited (`managym/cards/tests/fixtures/tle-bonus-sheet.jsonl`).

| Second land | Starting life | What the unpatched engine offers after Citadel | Correct result |
|---|---:|---|---|
| Airship Engine Room | 2 | Empty color decision at −1 life | Blue and red both complete the cast. The Room supplies the other color. State-based actions check for a loss after payment. |
| Airship Engine Room | 3 | Blue and red at 0 life | Same two choices. |
| Sunbaked Canyon | 3 | Empty color decision at 0 life | Only Canyon can go first. Pay 1 life, choose red, then use Citadel for blue. The cast completes and the payer loses normally. |
| Sunbaked Canyon | 5 | Blue | Either land can go first. The payer finishes at 1 life, with {U}{R} floating from Heartbeat and Sokka on the stack. |

The public 0.2.6 wheel and native checkout `b611abb3d7` both returned the two empty decisions with `skip_trivial=false`. On the wheel, enabling skipping made both failing steps exceed a two-second subprocess deadline. The adjacent controls returned. The original Linux report provides a sampled stack and no replay, seed or saved board. These positions reproduce the reported call path and stall independently. [Issue 628](https://github.com/loopflowstudio/manabot/issues/628)

Run the bounded release probe from the repository root. It prints the four mode lists in table order, `[]`, `[1, 3]`, `[]`, `[1]`.

```sh
uv run --no-project --python 3.13 --with managym==0.2.6 python scratch/repro_mana_payment.py
```

## Acceptance demo

```sh
cargo test --manifest-path managym/Cargo.toml -p managym-cards --test cards tarnished_citadel
```

The test exercises both branches of the Room’s blue/red choice and the legal orders for Canyon, checking taps, life, bonus mana, the spell and the subsequent state-based result. The low-life Room failure assertion runs before the skip-enabled case, so an old build reports the missing colors immediately. Paying a position never depends on worker count, elapsed time or a search budget.

## Concept review disposition

Second review (`24f1ec2f3`): 1a, 2a, 3a, 4a are design 1–3 above, made for review. Third review (`f5992e81f`): pass 3 implements 1a, 2a and 4a for review. 3a puts the non-mana sibling in deferred findings. Fourth review (`8182d5c876`): its read graph is superseded; its non-mana ownership and `OrderReplacement` items remain deferred. Fifth review (`fc7f2f70e7`): 1a–4a are design 4 above, made for review; its convoke item stays in pass 5 and its two separate outcomes match the deferred findings. The shield board was dropped after checking that managym currently has no consumable one-shot prevention shield. Sixth review (`6f7b914a4`): 1a–4a are design 5, implemented in `ab008535c`. Its convoke, non-mana ownership and `OrderReplacement` items match pass 5 and the deferred findings. Seventh review (`4d5644302`): 1a, 3a, 4a are design 6 (pass 4a) and 2a is design 7 (pass 4b), made for review; its convoke item matches pass 5 and non-mana ownership stays deferred.

## Deferred findings

- **Own announced non-mana activations** (third review choice 3a, made for review; 3b would fold it in here). `Operation::Activation` / `ActivatedTargeting` re-read `activated_abilities.get(ability_index)` in `prepare_activated_ability` (`flow/play.rs` ≈1871/1986) after mana is spent (`flow/payment.rs` ≈1413). A grant that ends while paying gives `AgentError("invalid ability index")` or shifts to another ability. Needs owned capture in `flow/play.rs` begin/pay, `flow/payment.rs`, and the hash; a separate Task.

## Docs

- `docs/engine.md` (`### Forced decisions`): `Game::tick` returns `Result<bool, AgentError>`; empty non-terminal spaces and failed forced/automatic actions error under both skip settings; isolated vector slots poison alone. Built in pass 1.
- `managym/README.md` describes the signed life budget and the order of fees. Pass 5 documents the completed authority and grouped help.
