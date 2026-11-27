# MTG-500 · Finish manual mana payment

A manual payment must return the payer’s next legal choice, finish the cast, or reach the game’s normal outcome. One game waiting for a mana color currently holds every other game at the native step barrier.

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

## Where progress stops

Verified against the checkout (`b611abb3d7`); the concept review (`scratch/concept-review.html`) widened these from the reported call path.

- **Raw life comparisons.** `check_activation_terms` (`flow/play.rs:1952`) and cast admission's `life_alternative` (`flow/play.rs:813`) compare `payable_life < fee as i32`, rejecting a zero fee at negative life. `Game::can_pay_life` (`flow/effects.rs`) is the rule (CR 119.4). `PaymentModel` holds `life: u32`, clamping negative life to 0, and `reserve_announced_payment` (`flow/mana.rs:3211`) subtracts announced life eagerly with `saturating_sub`.
- **Effect-blind witness.** The payment model knows printed `life_cost` but not `mana_suffix()` effects. Only two catalog cards have one: Tarnished Citadel (3 damage, `cards/src/ody.rs`) and The Great Henge (+2 life, `cards/src/eld.rs`). Execution runs the suffix before exposing `Decision::ManaChoice`, so the model admits Citadel-first with a Canyon fee that is unaffordable once the damage lands. `payment_sources` also filters by current life, hiding Canyon even when Henge would gain life first.
- **Colored convoke outside the model.** `while_paying` (`flow/mana/help.rs:227`) wraps three outer admission calls (`flow/mana.rs:3390, 3629, 4063`); nested surfaces (`automatic_upfront_path`, `mana_choice_actions`/`mana_choice_completes_payment`, `mana_spend_choices`, `mana_tap_cost_actions`, `automatic_payment_action`) skip it. This is the Guardian of Vitu-Ghazi empty-upfront probe in `scratch/questions.md`.
- **No-progress step.** `Game::tick` (`flow/tick.rs:113`) treats zero actions like one: with skipping on it loops forever, with skipping off it publishes a playerless space. A failed trivial action returns `true` (a game-over without a result), and `automatic_payment_action`'s witness failure panics through `.expect`. `Game::play` ignores step errors in a `while !is_game_over()` loop.

## Design · made for review

The concept review recommended these and the run is headless; each is reversible internal engine work. Nobody has approved them.

1. **Step invariant everywhere (review 3a).** `tick` returns `Result<bool, AgentError>`. Any non-terminal space with no actions is an `AgentError` before skipping or publishing; trivial and automatic-payment actions execute with `?`. Propagate through `finish_action_step`, `step` and `DecisionSurface::apply_search_action`; `Game::play` stops on error. Passed over: guarding only suspended mana decisions, which leaves `PendingChoice::Pay`, the fake game-over and the panic.
2. **Exact life in the model (review 4a).** `can_pay_life` at both `play.rs` sites. `PaymentModel` carries a signed `LifeBudget { life: i32, cant_change: bool }`. Post-mana life costs (announced `payment.life()`, cast `life_alternative`, resolution `Decision::Payment` tail life) become a trailing `Obligation::PayLife(u32)` checked at completion (CR 601.2h). `payment_sources` keeps non-life activation terms; the witness enforces `life_cost` against the running budget. Passed over: patching only the comparisons and keeping eager subtraction, which leaves Henge-then-Canyon wrong.
3. **Probe mana suffixes once (review 1a).** While building `payment_sources`, for an ability with a non-empty `mana_suffix()`, run the shared cost/suffix transition once on an isolated fork. A finished transition that changed only life records `ManaEffect::LifeDelta(i32)` on the activation and its output; a suspended one (replacement ordering) or one that changed anything else is `ManaEffect::BoardChanging` and is evaluated exactly on a fork, skipping caches and direct shortcuts only for that source. The probe shares the transition below offer validation. Prevention, replacements and `LifeTotalCantChange` come from the real damage path. Passed over: forking `Game` inside the witness DFS, which sees stale root state, breaks `PaymentCache` keys and clones in the priority hot path.
4. **Colored convoke in the model (review 2a).** `ManaSource` gains `convoke_colors: ManaTypeSet`; the witness reduces an owed colored pip by tapping such a source, and `while_paying`/`convoke_search` are deleted. Passed over: one wrapper routing all callers through `while_paying` (smaller now, keeps the unmemoized search beside the model). If 2a proves larger than a pass, fall back to the wrapper rather than defer: with pass 1 in place, a deferred convoke gap turns legal Guardian plays into errors.

Preserve native action identities, source order, the chooser, individual Heartbeat bonus decisions, spend restrictions and receipts. No changes to Rayon, GPU collection, card definitions or serialized formats. The `PaymentScan` debug check of `shared_payment_sources` against fresh `payment_sources` must keep holding (the probe leaves `&self` untouched).

## Remaining passes

1. **Step invariant and life fee rule.** Design items 1 and the two `can_pay_life` substitutions. Tests: a constructed empty non-terminal decision errors under both skip settings, strictly and in an isolated two-slot vector (only that slot poisoned, peer returns); the Room at 2 life offers blue and red. Teaches whether any existing test relied on the fake game-over or ignored step errors.
2. **Exact payment witness.** Design items 2 and 3. Tests in `managym/cards/tests/cards/ody.rs` beside the Citadel cases: the four table rows through normal cast, activation and mode submissions with skipping off first; a prevented Citadel hit with a payable Canyon fee; Citadel→Henge→Canyon at 2 life and Canyon+Henge paying {G}{G}{R} plus 2 life at 2 life (`eld.rs` if Henge-only); a fork offer read leaves the live game untouched.
3. **Colored convoke in the model.** Design item 4. Regression: the Guardian of Vitu-Ghazi board from `scratch/questions.md` completes its cast.
4. **Collection check and acceptance.** Restore a pre-activation snapshot into one slot of a two-slot native vector with skipping on; both slots return and match scalar play. Run the acceptance demo, debug-profile nextest for the touched crates, and the engine budget comparison.

## Acceptance demo

```sh
cargo test --manifest-path managym/Cargo.toml -p managym-cards --test cards tarnished_citadel
```

The test exercises both branches of the Room’s blue/red choice and the legal orders for Canyon, checking taps, life, bonus mana, the spell and the subsequent state-based result. The low-life Room failure assertion runs before the skip-enabled case, so an old build reports the missing colors immediately. Paying a position never depends on worker count, elapsed time or a search budget.

## Concept review disposition

All four review choices adopted above as made for review (1a, 2a with 2b fallback, 3a, 4a). Its `Game::play` hygiene note is in pass 1; the `PaymentScan` debug invariant is a constraint above. The review's replacement-ordering counterexample (Citadel + prevention + damage doubling) is covered by `BoardChanging`; no current card produces it, so no fixture is required.
