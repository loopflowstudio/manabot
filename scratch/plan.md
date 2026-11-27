# MTG-500 · Finish manual mana payment

A manual payment must return the payer’s next legal choice, finish the cast, or reach the game’s normal outcome. The reported stall held every game at the native step barrier.

Pass 1 (`1c521e300`) propagated empty, forced and automatic decision errors and applied the zero-life cost rule. The Room at −1 offers blue and red, isolated stepping poisons only the malformed slot under both skip settings, and all 23 focused debug tests passed.

Pass 2 (`a9359c03b`) carries signed life through mana, checks trailing fees and records source coupling. All 78 focused debug checks passed, including the four Heartbeat positions, the announced and Ward fees, and the read-only fork. The 512-game debug step benchmark was stopped at 4m41s before it emitted a digest. The release Engine Budget comparison remains in pass 5.

The first pass 2 attempt found the counterexample below (`34092c814`, test `tarnished_citadel_and_the_great_henge_change_another_mana_source` in `managym/cards/tests/cards/ody.rs`) and changed no production code. The concept review (`24f1ec2f3`, `scratch/concept-review.html`) replanned around it.

## Pass 3 counterexample · an activated ability can disappear while its fee is paid

Pass 3 recorded the counterexample in `7124e6444`. The focused debug proof passed.

The constructed position `mana_upfront_fee_can_remove_the_activated_ability` in `managym/src/flow/mana/tests.rs` grants Gray Ogre `{1}{R}, {T}: Add {B}{B}` while its controller has unspent mana. Start with `{R}` and an untapped Mountain. Announce the activation and tap the Mountain. The native tap advances payment and spends `{1}{R}`. The final red unit removes the grant while the activation is in flight, so `complete_mana_upfront` panics with an empty ability list at index 0. The retained test taps and spends those same units on a fork and asserts that the activation remains pending, the live ability is gone, and the root digest is unchanged.

`Decision::ManaUpfront` retains the permanent and a live ability index. The payment model uses that lookup for reservations and saved obligations. Execution and the continuation projection also use it. Rebuilding sources on an exact fork leaves the activation with no authority to read. It must retain the announced cost and chosen ability through nested payment, then evaluate any dynamic output against the board at resolution. The source index still identifies the offered action and its provenance.

Realign pass 3 around an owned activation in `flow/decision.rs` before adding forked search in `flow/mana.rs`. Decide how the activation binds to snapshots and state hashes (`state/hash.rs`), since `ManaUpfront` is part of the canonical state. The acceptance check should execute the published PayCost on live and restored positions and reach the `{B}{B}` choice after the grant disappears. The forked search, colored convoke and collection checks remain pending.


## Why a frozen source list fails

Angel of Vitality is 4/4 at 25 life or more; Raucous Audience makes `{G}{G}` while its controller has a power-4 creature, else `{G}`. Casting Solemn Simulacrum (`{4}`):

- **28 life, two Tarnished Citadels.** Each Citadel alone leaves 25 life; both leave 22 and Audience drops to `{G}` (3 mana). Only Citadel → Audience → Citadel pays. Probing each source once from the root sees no change.
- **22 life, The Great Henge.** Henge gains 3 with Angel's replacement and makes `{G}{G}`; Audience then makes `{G}{G}`. The root total (2 + 1) prunes the legal cast before Henge is tried.
- **25 life, Sunbaked Canyon + Island.** No `mana_suffix()` anywhere; Canyon's ordinary `life_cost: 1` crosses the threshold. Audience must go first.
- **One-shot prevention shield + two Citadels.** The first hit consumes the shield; a root probe of each sees it unconsumed.
- **Sacrifice source + White Lotus Tile / Avatar Kyoshi.** Sacrificing changes a dynamic output.

Coupling is board-level: what a payment step writes (life, sacrifice, taps, counters, pool, effects) intersecting what the layer board reads (`LayerBoard.reads: StateReads`, `dynamic_activations`, `has_triggered_mana_abilities`, `has_activated_cost_reduction` in `flow/layers.rs`, `flow/statics.rs`). The pure witness models the source list at the question’s root. Pass 3 will rebuild it after a coupled transition.

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

Preserve native action identities, source order, the chooser, individual Heartbeat bonus decisions, spend restrictions and receipts. No changes to Rayon, GPU collection, card definitions or serialized formats. The `PaymentScan` debug rebuild check must keep holding: probes and coupled search run on forks and leave `&self` untouched.

## Remaining passes

3. **Coupled regime** (`flow/mana.rs`, `flow/layers.rs`, `flow/statics.rs`). The scan retains per-activation coupling and marks fixed taps separately. Layer and activation reads are summarized once. Keep ordinary Forest taps in the pure handoff on the Angel/Audience board. Forked search with pure-finish-first, mutator-only branching, reservation replay, memoization and shortcut deferral. Turn the Pass 2 test into the expected behavior: at 28 life only Citadel → Audience → Citadel is offered; at 22 life Henge-first is admitted; Canyon at 25 offers Audience first; the shield and sacrifice cases if a catalog card builds them (constructed positions otherwise); 10 Forests + coupled sources checking `{15}` returns quickly in debug.
4. **Colored convoke in the model.** Regression: the Guardian of Vitu-Ghazi board from `scratch/questions.md` completes its cast.
5. **Collection check and acceptance.** Restore a pre-activation snapshot into one slot of a two-slot native vector with skipping on; both slots return and match scalar play. Acceptance demo, debug nextest for touched crates, engine budget comparison. Rewrite the `managym/README.md` mana-payment section for the built design (the unbuilt description was removed in realign).

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

Second review (`24f1ec2f3`): choices 1a, 2a, 3a, 4a adopted above as made for review. Its invariants (reservation replay on forks, shortcut deferral, `PaymentScan` debug check) are in design 1. Filtering live `Decision::OrderReplacement` modes by payment completion is a separate outcome (deferred below). The first review's dispositions are superseded.

## Deferred findings

- **Restrict live `OrderReplacement` offers during payment** when one order strands the cost (`proposed_event.rs`). No current catalog card builds it; pass 1 turns any constructed case into an error rather than a spin.

## Docs

- `docs/engine.md` (`### Forced decisions`): `Game::tick` returns `Result<bool, AgentError>`; empty non-terminal spaces and failed forced/automatic actions error under both skip settings; isolated vector slots poison alone. Built in pass 1.
- `managym/README.md` describes the signed life budget and the order of fees. Pass 5 documents the completed authority and grouped help.
