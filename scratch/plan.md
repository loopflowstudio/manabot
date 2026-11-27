# MTG-500 · Finish manual mana payment

A manual payment must return the payer’s next legal choice, finish the cast, or reach the game’s normal outcome. The reported stall held every game at the native step barrier.

Pass 1 (`1c521e300`) propagated empty, forced and automatic decision errors and applied the zero-life cost rule. The Room at −1 offers blue and red, isolated stepping poisons only the malformed slot under both skip settings, and all 23 focused debug tests passed.

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

## Where the payment witness loses progress

Verified against the checkout (`b611abb3d7`); the concept review (`scratch/concept-review.html`) widened these from the reported call path.

- **Life budget.** `PaymentModel` holds `life: u32`, clamping negative life to 0, and `reserve_announced_payment` (`flow/mana.rs`) subtracts announced life eagerly with `saturating_sub`.
- **Effect-blind witness.** The payment model knows printed `life_cost` but not `mana_suffix()` effects. Only two catalog cards have one: Tarnished Citadel (3 damage, `cards/src/ody.rs`) and The Great Henge (+2 life, `cards/src/eld.rs`). Execution runs the suffix before exposing `Decision::ManaChoice`, so the model admits Citadel-first with a Canyon fee that is unaffordable once the damage lands. `payment_sources` also filters by current life, hiding Canyon even when Henge would gain life first.
- **Colored convoke outside the model.** `while_paying` (`flow/mana/help.rs:227`) wraps three outer admission calls (`flow/mana.rs:3390, 3629, 4063`); nested surfaces (`automatic_upfront_path`, `mana_choice_actions`/`mana_choice_completes_payment`, `mana_spend_choices`, `mana_tap_cost_actions`, `automatic_payment_action`) skip it. This is the Guardian of Vitu-Ghazi empty-upfront probe in `scratch/questions.md`.

## Design · made for review

The concept review recommended these and the run is headless; each is reversible internal engine work. Nobody has approved them.

1. **Exact life in the model (review 4a).** `PaymentModel` carries a signed `LifeBudget { life: i32, cant_change: bool }`. Post-mana life costs (announced `payment.life()`, cast `life_alternative`, resolution `Decision::Payment` tail life) become a trailing `Obligation::PayLife(u32)` checked at completion (CR 601.2h). `payment_sources` keeps non-life activation terms and the witness enforces `life_cost` against the running budget.
2. **Probe mana suffixes once (review 1a).** While building `payment_sources`, for an ability with a non-empty `mana_suffix()`, run the shared cost/suffix transition once on an isolated fork. A finished transition that changed only life records `ManaEffect::LifeDelta(i32)` on the activation and its output; a suspended one (replacement ordering) or one that changed anything else is `ManaEffect::BoardChanging` and is evaluated exactly on a fork, skipping caches and direct shortcuts only for that source. The probe shares the transition below offer validation. Prevention, replacements and `LifeTotalCantChange` come from the real damage path. Passed over: forking `Game` inside the witness DFS, which sees stale root state, breaks `PaymentCache` keys and clones in the priority hot path.
3. **Colored convoke in the model (review 2a).** `ManaSource` gains `convoke_colors: ManaTypeSet`; the witness reduces an owed colored pip by tapping such a source, and `while_paying`/`convoke_search` are deleted. Passed over: one wrapper routing all callers through `while_paying` (smaller now, keeps the unmemoized search beside the model). If 2a proves larger than a pass, fall back to the wrapper rather than defer: with pass 1 in place, a deferred convoke gap turns legal Guardian plays into errors.

Preserve native action identities, source order, the chooser, individual Heartbeat bonus decisions, spend restrictions and receipts. No changes to Rayon, GPU collection, card definitions or serialized formats. The `PaymentScan` debug check of `shared_payment_sources` against fresh `payment_sources` must keep holding (the probe leaves `&self` untouched).

## Remaining passes

2. **Exact payment witness.** Update the life budget and source probing in `managym/src/flow/mana.rs` and cast admission in `managym/src/flow/play.rs`. Tests in `managym/cards/tests/cards/ody.rs` beside the Citadel cases: the four table rows through normal cast, activation and mode submissions with skipping off first; a prevented Citadel hit with a payable Canyon fee; Citadel→Henge→Canyon at 2 life and Canyon+Henge paying {G}{G}{R} plus 2 life at 2 life (`eld.rs` if Henge-only); a fork offer read leaves the live game untouched.
3. **Colored convoke in the model.** Regression: the Guardian of Vitu-Ghazi board from `scratch/questions.md` completes its cast.
4. **Collection check and acceptance.** Restore a pre-activation snapshot into one slot of a two-slot native vector with skipping on; both slots return and match scalar play. Run the acceptance demo, debug-profile nextest for the touched crates, and the engine budget comparison.

## Acceptance demo

```sh
cargo test --manifest-path managym/Cargo.toml -p managym-cards --test cards tarnished_citadel
```

The test exercises both branches of the Room’s blue/red choice and the legal orders for Canyon, checking taps, life, bonus mana, the spell and the subsequent state-based result. The low-life Room failure assertion runs before the skip-enabled case, so an old build reports the missing colors immediately. Paying a position never depends on worker count, elapsed time or a search budget.

## Concept review disposition

All four review choices adopted above as made for review (1a, 2a with 2b fallback, 3a, 4a). Its `Game::play` hygiene note is in pass 1; the `PaymentScan` debug invariant is a constraint above. The review's replacement-ordering counterexample (Citadel + prevention + damage doubling) is covered by `BoardChanging`; no current card produces it, so no fixture is required.

## Docs

- `docs/engine.md` (`### Forced decisions`): updated `Game::tick` contract to document `Result<bool, AgentError>`, empty non-terminal action-space rejection under both `skip_trivial` settings, error propagation for forced/automatic actions, and isolated vector slot poisoning.
- `managym/README.md` (mana payment & `PaymentScan`/`PaymentCache` architecture): documented colored convoke inside `PaymentModel` (replacing the `while_paying` reference), CR 119.4 `can_pay_life` and signed `LifeBudget` with trailing `Obligation::PayLife`, isolated-fork `mana_suffix()` probing (`LifeDelta` / `BoardChanging`) with the Tarnished Citadel / Airship Engine Room / Sunbaked Canyon / The Great Henge examples, and direct-path/cache rules.
